# Load required packages
library(rstan)
library(bayesplot)
library(ggplot2)
library(dplyr)

# Set options for parallel processing
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# ----------------------------------------------------------
# Step 1: Generate simulated biosimilar data
# ----------------------------------------------------------
set.seed(123)

# Parameters (bevacizumab HMW aggregates example)
J_b <- 6                    # number of biosimilar batches
n_per_batch <- 3           # replicates per batch
mu_r <- 1.50               # reference mean (%)
tau_r <- 0.14              # reference SD
mu_b_true <- 1.58          # true biosimilar mean
sigma_w <- 0.08            # within-batch SD
sigma_b <- 0.03            # between-batch SD

# Generate hierarchical data
mu_batch_true <- rnorm(J_b, mean = mu_b_true, sd = sigma_b)
y_data <- c()
batch_id <- c()

for (j in 1:J_b) {
  y_batch <- rnorm(n_per_batch, mean = mu_batch_true[j], sd = sigma_w)
  y_data <- c(y_data, y_batch)
  batch_id <- c(batch_id, rep(j, n_per_batch))
}

# Prepare data for Stan
stan_data <- list(
  J_b = J_b,
  N = length(y_data),
  n_per_batch = n_per_batch,
  batch = batch_id,
  y = y_data,
  mu_r = mu_r,
  tau_r = tau_r
)

# ----------------------------------------------------------
# Step 2: Define Stan model as string
# ----------------------------------------------------------
stan_model_code <- "
data {
  int<lower=1> J_b;              // number of biosimilar batches
  int<lower=1> N;                // total observations
  int<lower=1> n_per_batch;      // replicates per batch
  int<lower=1,upper=J_b> batch[N]; // batch assignment
  vector[N] y;                   // observations
  real mu_r;                     // reference mean
  real<lower=0> tau_r;           // reference sd
}

parameters {
  vector[J_b] mu_batch;          // batch means
  real mu_b;                     // biosimilar mean
  real<lower=0> sigma_w;         // within-batch sd
  real<lower=0> sigma_b;         // between-batch sd
  real<lower=0> psi;             // similarity tolerance
}

model {
  // Priors
  mu_b ~ normal(mu_r, psi);
  sigma_w ~ inv_gamma(2, 0.08);
  sigma_b ~ inv_gamma(2, 0.03);
  psi ~ inv_gamma(2, tau_r);
  
  // Hierarchy
  mu_batch ~ normal(mu_b, sigma_b);
  
  // Likelihood
  for (i in 1:N) {
    y[i] ~ normal(mu_batch[batch[i]], sigma_w);
  }
}

generated quantities {
  real diff = mu_b - mu_r;
  int<lower=0,upper=1> similar = fabs(diff) < 0.5 ? 1 : 0;
  
  // Posterior predictive for new batch
  real mu_new_batch = normal_rng(mu_b, sigma_b);
  real y_pred = normal_rng(mu_new_batch, sigma_w);
}
"

# ----------------------------------------------------------
# Step 3: Compile and run MCMC
# ----------------------------------------------------------
fit <- stan(
  model_code = stan_model_code,
  data = stan_data,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  thin = 1,
  seed = 456,
  control = list(adapt_delta = 0.95)
)

# ----------------------------------------------------------
# Step 4: Diagnostics
# ----------------------------------------------------------
print(fit, pars = c("mu_b", "mu_batch", "sigma_w", "sigma_b", "psi", "diff"))

# Check convergence
print(summary(fit)$summary[, c("Rhat", "n_eff")])

# Trace plots
mcmc_trace(as.array(fit), pars = c("mu_b", "sigma_b", "psi"))

# Density plots
mcmc_dens_overlay(as.array(fit), pars = c("mu_b", "diff"))

# Pairs plot for correlations
pairs(fit, pars = c("mu_b", "sigma_b", "psi"))

# ----------------------------------------------------------
# Step 5: Posterior analysis
# ----------------------------------------------------------
# Extract posterior samples
posterior <- extract(fit)

# Posterior mean and credible interval for mu_b
mu_b_post <- posterior$mu_b
cat("Posterior mean mu_b:", mean(mu_b_post), "\n")
cat("95% Credible Interval:", quantile(mu_b_post, c(0.025, 0.975)), "\n")

# Probability of similarity
diff_post <- posterior$diff
prob_similar <- mean(abs(diff_post) < 0.5)
cat("P(|mu_b - mu_r| < 0.5%):", prob_similar, "\n")

# Shrinkage analysis
cat("\nShrinkage Analysis:\n")
cat("Observed batch mean:", mean(y_data), "\n")
cat("Posterior mean mu_b:", mean(mu_b_post), "\n")
cat("Reference mean mu_r:", mu_r, "\n")
shrinkage <- (mean(y_data) - mean(mu_b_post)) / (mean(y_data) - mu_r)
cat("Shrinkage factor:", shrinkage, "\n")

# ----------------------------------------------------------
# Step 6: Visualization
# ----------------------------------------------------------
# Posterior distribution of mu_b
df_posterior <- data.frame(
  mu_b = mu_b_post,
  diff = diff_post
)

ggplot(df_posterior, aes(x = mu_b)) +
  geom_histogram(aes(y = ..density..), bins = 50, 
                 fill = "steelblue", alpha = 0.7) +
  geom_density(color = "darkblue", size = 1) +
  geom_vline(xintercept = mu_r, linetype = "dashed", 
             color = "red", size = 1) +
  geom_vline(xintercept = mean(mu_b_post), 
             color = "darkgreen", size = 1) +
  labs(title = "Posterior Distribution of Biosimilar Mean (mu_b)",
       x = "HMW Aggregates (%)", y = "Density") +
  theme_minimal()

# Difference from reference with equivalence margin
ggplot(df_posterior, aes(x = diff)) +
  geom_histogram(aes(y = ..density..), bins = 50, 
                 fill = "coral", alpha = 0.7) +
  geom_density(color = "darkred", size = 1) +
  geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", 
             color = "black", size = 1) +
  geom_vline(xintercept = 0, color = "blue", size = 1) +
  annotate("text", x = 0.4, y = max(density(diff_post)$y) * 0.9,
           label = paste0("P(similar) = ", round(prob_similar, 3))) +
  labs(title = "Posterior Distribution of Difference (mu_b - mu_r)",
       subtitle = "Dashed lines show equivalence margin (±0.5%)",
       x = "Difference (%)", y = "Density") +
  theme_minimal()

# Batch-level shrinkage
batch_means_data <- sapply(1:J_b, function(j) mean(y_data[batch_id == j]))
batch_means_post <- apply(posterior$mu_batch, 2, mean)

df_batches <- data.frame(
  batch = 1:J_b,
  observed = batch_means_data,
  posterior = batch_means_post,
  reference = mu_r
)

ggplot(df_batches, aes(x = batch)) +
  geom_point(aes(y = observed, color = "Observed"), size = 3) +
  geom_point(aes(y = posterior, color = "Posterior (shrunk)"), size = 3) +
  geom_hline(aes(yintercept = reference, color = "Reference"), 
             linetype = "dashed", size = 1) +
  geom_segment(aes(x = batch, xend = batch, y = observed, yend = posterior),
               arrow = arrow(length = unit(0.2, "cm")), alpha = 0.5) +
  scale_color_manual(values = c("Observed" = "red", 
                                "Posterior (shrunk)" = "blue",
                                "Reference" = "darkgreen")) +
  labs(title = "Hierarchical Shrinkage of Batch Means",
       subtitle = "Arrows show shrinkage toward reference",
       x = "Batch", y = "HMW Aggregates (%)", color = "Type") +
  theme_minimal() +
  theme(legend.position = "bottom")

# ----------------------------------------------------------
# Step 7: Model comparison (optional - with alternative priors)
# ----------------------------------------------------------
# Fit weakly informative model
stan_data_weak <- stan_data
stan_data_weak$tau_r <- 1.0  # Much larger prior uncertainty

fit_weak <- stan(
  model_code = stan_model_code,
  data = stan_data_weak,
  chains = 4,
  iter = 2000,
  warmup = 1000
)

# Compare LOO (Leave-One-Out Cross-Validation)
library(loo)
log_lik_1 <- extract_log_lik(fit, merge_chains = FALSE)
log_lik_2 <- extract_log_lik(fit_weak, merge_chains = FALSE)

loo_1 <- loo(log_lik_1, r_eff = relative_eff(log_lik_1))
loo_2 <- loo(log_lik_2, r_eff = relative_eff(log_lik_2))

print(loo_compare(loo_1, loo_2))