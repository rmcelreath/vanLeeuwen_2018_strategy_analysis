d <- read.delim( "data.txt" , stringsAsFactors=FALSE )

library(rethinking)

# code response
d$y <- ifelse( d$response == "Innovation" , 1 , NA )
d$y <- ifelse( d$response == "Majority" , 2 , d$y )
d$y <- ifelse( d$response == "Minority" , 3 , d$y )

# code predictors
d$male <- ifelse( d$sex=="Boy" , 1 , 0 )
d$age_z <- ( d$age - mean(d$age) )/sd(d$age)
d$culture_id <- coerce_index( d$culture )
d$order_maj1 <- ifelse( d$order=="Maj_first" , 1 , 0 )

dat_list <- list(
    N = nrow(d),
    N_cult = max(d$culture_id),
    y = d$y,
    male = d$male,
    age = d$age_z,
    age_sq = (d$age_z^2),
    omaj1 = d$order_maj1,
    cult = d$culture_id
)

# trinomial model with explicit social learning probabilities
# 6 strats:
# 1 Conform: follow majority (option 2)
# 2 Hipster: follow minority (option 3)
# 3 Unbiased: copy at random (3/4 chance option 2 or 1/4 option 3)
# 4 Maverick: choose option 1
# 5 Random: any option at random (1/3 each)
# 6 copy first demonstrated
# each strat implies probability of each of 3 options
# need to compute mixture over these to get likelihood of data

# basic model - quadratic age term
m1 <- stan( file="model_trinomial.stan" , data=dat_list , chains=1 , iter=800 , control=list(adapt_delta=0.95) )

# linear age term only
m0 <- stan( file="model_trinomial_agesimple.stan" , data=dat_list , chains=1 , iter=800 , control=list(adapt_delta=0.95) )

# everything a varying effect by culture
# very little data to estimate variance terms, so prior here is influential
# setting it to allow noticable variation among groups
dat_list$sigma_prior <- 2.0
m2 <- stan( file="model_trinomial_byculture.stan" , data=dat_list , chains=3 , iter=800 , control=list(adapt_delta=0.95) )

precis(m2,2) # lots of parameters
# show intercepts of each strategy by culture
plot( precis(m2,3,pars="alpha_k") )

compare(m0,m1,m2) # WAIC comparison
# WAIC plot
plot(compare(m0,m1,m2))

############
# now extract samples and draw proportions of strategies across age

# use m2 samples, but useful to compare m1 (total pooling) to see impact of pooling
post <- extract.samples(m2)

age_u <- sort(unique(dat_list$age))
age_u <- seq( from=min(age_u) , to=max(age_u) , length.out=20 )
na <- length(age_u)

ns <- dim(post$alpha)[1]
p <- array( 0 , dim=c(ns,na,6) ) # sample, age, strategy
p_cumu <- array( 0 , dim=c(ns,na,6) )

# devtools::install_github("karthik/wesanderson")
library(wesanderson)
#pal <- wes_palette("Darjeeling")
#pal <- wes_palette("FantasticFox")
pal <- wes_palette("Moonrise3")
pal[6] <- "#85D4C3"

blank(w=2)

par(mfrow=c(1,2))
# compute prob of each strat at each age for each sample
for ( male in 0:1 ) {

    for ( i in 1:ns ) {
        for ( j in 1:na ) {
            alpha_temp <- c( post$alpha[i,] , 0 )
            x <- age_u[j]
            logit_p <- with( post , alpha_temp + b_m[i,]*male + b_age[i,]*x + b_age2[i,]*x^2 )
            p[i,j,] <- exp(logit_p) / sum(exp(logit_p))
            # convert to cumulative prob, so easier to visualize
            p_cumu[i,j,] <- cumsum( p[i,j,] )
        }#j
    }#i

    # now plot as shaded polygons
    plot( NULL , xlim=range(age_u) , ylim=c(0,1) , xlab="age_z" , ylab="cumulative probability" )
    n_age <- length(age_u)
    mu <- apply( p_cumu , 2:3 , mean )
    for ( i in 1:6 ) {
        if ( i==1 ) bottom <- rep(0,n_age)
        else bottom <- mu[,i-1]
        polygon( c(age_u,age_u[n_age:1]) , c(mu[,i],bottom[n_age:1]) , col=pal[i] , border=NA )
    }#i
    mtext( ifelse( male==1 , "male" , "female" ) )

}#for male

#####
# additional plots

# density of conformist strength, shown as prob of copying majority
dens( 3^post$omega/(3^post$omega+1) , xlim=c(0.5,1) , adj=0.2 )
abline(v=3/4,lty=2)

# version with animation to show uncertainty
# press ENTER to advance animation; ctrl-c to escape
do <- TRUE
s <- 1
while ( do==TRUE ) {
    plot( NULL , xlim=range(age_u) , ylim=c(0,1) , xlab="age_z" , ylab="cumulative probability" )
    for ( i in 1:6 ) {
        if ( i==1 ) bottom <- rep(0,n_age)
        else bottom <- p_cumu[s,,i-1]
        polygon( c(age_u,age_u[n_age:1]) , c(p_cumu[s,,i],bottom[n_age:1]) , col=pal[i] , border=NA )
    }
    x <- readline("Enter for another plot:")
    s <- s + 1
}

# animation straight to gif
png(file="fantastic%02d.png", width=500, heigh=400)
  for ( s in 1:50 ){
    plot( NULL , xlim=range(age_u) , ylim=c(0,1) , xlab="age_z" , ylab="cumulative probability" )
    for ( i in 1:6 ) {
        if ( i==1 ) bottom <- rep(0,n_age)
        else bottom <- p_cumu[s,,i-1]
        polygon( c(age_u,age_u[n_age:1]) , c(p_cumu[s,,i],bottom[n_age:1]) , col=pal[i] , border=NA )
    }
  }
dev.off()
# convert pngs to one gif using ImageMagick
system("convert -delay 10 *.png fantastic.gif")


