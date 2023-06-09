data {
  int<lower = 0> n; // number of tested persons
  vector[2] mu;
  vector<lower = 0>[2] prev_beta;
  cov_matrix[2] Sigma;
}
parameters {
  real<lower = 0, upper = 1> prev;
  vector[2] logit_Se_Sp;
}
transformed parameters {
  real logit_Se = logit_Se_Sp[1];
  real logit_Sp = logit_Se_Sp[2];

  real<lower=0,upper=1> Se = inv_logit(logit_Se);
  real<lower=0,upper=1> Sp = inv_logit(logit_Sp);
  
  real<lower = 0, upper = 1> p;
  p = prev * Se + (1 - prev) * (1 - Sp);
}
model {
  logit_Se_Sp ~ multi_normal(mu, Sigma);
  prev ~ beta(prev_beta[1],prev_beta[2]);
}

generated quantities {
  int y_pred = binomial_rng(n, p);
}
