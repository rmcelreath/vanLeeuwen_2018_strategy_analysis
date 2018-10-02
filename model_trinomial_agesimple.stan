// trinomial outcome model with 6 data-generating strategies
// strat 1: majority copying
// strat 2: minority copying
// strat 3: unbiased copying (only demonstrated options)
// strat 4: maverick (copy undemonstrated option)
// strat 5: random (1/3 chance each option)
// strat 6: first shown (copy first demonstrated option)
data{
    int N;
    int N_cult;
    int y[N];
    int male[N];
    real age[N];
    real age_sq[N];
    int omaj1[N];
    int cult[N];
}
transformed data{
    int N_strats;
    int N_options;
    N_strats = 6;
    N_options = 3;
}
parameters{
    vector[N_strats-1] alpha; // intercepts for each strategy (last is reference category)

    vector[N_strats] b_m;
    vector[N_strats] b_age;

    real<lower=0,upper=1> eps; // error rate
    real<lower=1> omega; // conformity strength parameter: Pr(maj) = M^omega/(M^omega + N^omega), where M=count of majority option and N=count of minority option
}
model{
    vector[N_strats] terms;
    vector[N_strats] logit_p;
    vector[N_strats] log_p;
    vector[N_strats] alpha_temp;

    // priors
    alpha ~ normal(0,4);
    b_m ~ normal(0,1);
    b_age ~ normal(0,1);

    eps ~ beta(1,100);
    omega ~ normal( 2 , 1 );

    for ( i in 1:(N_strats-1) ) alpha_temp[i] = alpha[i];
    alpha_temp[N_strats] = 0;

    // define mixture likelihood
    for ( i in 1:N ) {

        logit_p = alpha_temp + 
                  b_m*male[i] + 
                  b_age*age[i];
        log_p = log_softmax(logit_p); // log prob of each strategy
        
        if ( y[i]==1 ) { // unchosen option
            terms[1] = log_p[1] + log(eps);             // majority
            terms[2] = log_p[2] + log(eps) + log(0.5);             // minority
            terms[3] = log_p[3] + log(eps);         // unbiased copying
            terms[4] = log_p[4] + log1m(eps);       // Maverick
            terms[5] = log_p[5] + log(1.0/3.0);         // random choice
            terms[6] = log_p[6] + log(eps) + log(0.5);         // copy first demonstrated
        }
        if ( y[i]==2 ) { // majority option
            terms[1] = log_p[1] + log1m(eps) + omega*log(3) - log(3^omega + 1);       // majority
            terms[2] = log_p[2] + log(eps) + log(0.5);             // minority
            terms[3] = log_p[3] + log1m(eps) + log(0.75);        // unbiased copying
            terms[4] = log_p[4] + log(eps) + log(0.5);             // Maverick
            terms[5] = log_p[5] + log(1.0/3.0);         // random choice
            if ( omaj1[i]==1 )
                terms[6] = log_p[6] + log1m(eps);
            else
                terms[6] = log_p[6] + log(eps) + log(0.5);
        }
        if ( y[i]==3 ) { // minority option
            terms[1] = log_p[1] + log1m(eps) - log(3^omega + 1);             // majority
            terms[2] = log_p[2] + log1m(eps);       // minority
            terms[3] = log_p[3] + log1m(eps) + log(0.25);        // unbiased copying
            terms[4] = log_p[4] + log(eps) + log(0.5);             // Maverick
            terms[5] = log_p[5] + log(1.0/3.0);         // random choice
            if ( omaj1[i]==0 )
                terms[6] = log_p[6] + log1m(eps);
            else
                terms[6] = log_p[6] + log(eps) + log(0.5);
        }

        target += log_sum_exp( terms );
    }//i

}
generated quantities{
    vector[N] log_lik;

    {
        vector[N_strats] terms;
        vector[N_strats] logit_p;
        vector[N_strats] log_p;
        vector[N_strats] alpha_temp;

        for ( i in 1:(N_strats-1) ) alpha_temp[i] = alpha[i];
        alpha_temp[N_strats] = 0;

        for ( i in 1:N ) {

            logit_p = alpha_temp + 
                      b_m*male[i] + 
                      b_age*age[i];
            log_p = log_softmax(logit_p); // log prob of each strategy
            
            if ( y[i]==1 ) { // unchosen option
                terms[1] = log_p[1] + log(eps);             // majority
                terms[2] = log_p[2] + log(eps) + log(0.5);             // minority
                terms[3] = log_p[3] + log(eps);         // unbiased copying
                terms[4] = log_p[4] + log1m(eps);       // Maverick
                terms[5] = log_p[5] + log(1.0/3.0);         // random choice
                terms[6] = log_p[6] + log(eps) + log(0.5);         // copy first demonstrated
            }
            if ( y[i]==2 ) { // majority option
                terms[1] = log_p[1] + log1m(eps) + omega*log(3) - log(3^omega + 1);       // majority
                terms[2] = log_p[2] + log(eps) + log(0.5);             // minority
                terms[3] = log_p[3] + log1m(eps) + log(0.75);        // unbiased copying
                terms[4] = log_p[4] + log(eps) + log(0.5);             // Maverick
                terms[5] = log_p[5] + log(1.0/3.0);         // random choice
                if ( omaj1[i]==1 )
                    terms[6] = log_p[6] + log1m(eps);
                else
                    terms[6] = log_p[6] + log(eps) + log(0.5);
            }
            if ( y[i]==3 ) { // minority option
                terms[1] = log_p[1] + log1m(eps) - log(3^omega + 1);             // majority
                terms[2] = log_p[2] + log1m(eps);       // minority
                terms[3] = log_p[3] + log1m(eps) + log(0.25);        // unbiased copying
                terms[4] = log_p[4] + log(eps) + log(0.5);             // Maverick
                terms[5] = log_p[5] + log(1.0/3.0);         // random choice
                if ( omaj1[i]==0 )
                    terms[6] = log_p[6] + log1m(eps);
                else
                    terms[6] = log_p[6] + log(eps) + log(0.5);
            }

            log_lik[i] = log_sum_exp( terms );
        }//i
    }
}

