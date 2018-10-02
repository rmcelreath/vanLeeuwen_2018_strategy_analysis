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
    real sigma_prior;
}
transformed data{
    int N_strats;
    int N_options;
    int N_effects; // number of varying effects per culture
    N_strats = 6;
    N_options = 3;
    N_effects = (N_strats-1) + N_strats*3;
}
parameters{
    real<lower=0,upper=1> eps; // error rate
    real<lower=1> omega;

    // pooling parameters
    // means
    vector[N_strats-1] alpha;
    vector[N_strats] b_m;
    vector[N_strats] b_age;
    vector[N_strats] b_age2;
    // covariances
    vector<lower=0>[N_effects] Sigma;
    cholesky_factor_corr[N_effects] L_Rho;
    // individual culture effects
    matrix[N_effects,N_cult] z;
}
transformed parameters {
    vector[N_strats-1] alpha_k[N_cult];
    vector[N_strats] b_m_k[N_cult];
    vector[N_strats] b_age_k[N_cult];
    vector[N_strats] b_age2_k[N_cult];
    {
        matrix[N_cult,N_effects] v;
        v = (diag_pre_multiply(Sigma,L_Rho) * z)';
        for ( k in 1:N_cult ) {
            for ( i in 1:5 ) alpha_k[k,i] = alpha[i] + v[k,i];
            b_m_k[k] = b_m + v[k,6:11]';
            b_age_k[k] = b_age + v[k,12:17]';
            b_age2_k[k] = b_age2 + v[k,18:23]';
        }
    }
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
    b_age2 ~ normal(0,1);

    L_Rho ~ lkj_corr_cholesky(4);
    Sigma ~ normal( 0 , sigma_prior );
    to_vector(z) ~ normal(0,1);

    eps ~ beta(1,100);
    omega ~ normal( 2 , 1 );

    alpha_temp[N_strats] = 0;

    // define mixture likelihood
    for ( i in 1:N ) {
        int k;
        k = cult[i];

        for ( j in 1:(N_strats-1) ) alpha_temp[j] = alpha_k[k,j];

        logit_p = alpha_temp + 
                  b_m_k[k]*male[i] + 
                  b_age_k[k]*age[i] + b_age2_k[k]*age_sq[i];
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

        alpha_temp[N_strats] = 0;

        for ( i in 1:N ) {
            int k;
            k = cult[i];

            for ( j in 1:(N_strats-1) ) alpha_temp[j] = alpha_k[k,j];

            logit_p = alpha_temp + 
                  b_m_k[k]*male[i] + 
                  b_age_k[k]*age[i] + b_age2_k[k]*age_sq[i];
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

