# =============================================================================
# HPV Vaccine Stability – Hierarchical Arrhenius Model in Stan (CmdStanPy)
# Fully non-centered parameterization, proper units, publication-ready
# =============================================================================

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import os

# -------------------------------
# 1. Generate synthetic data (same as before)
# -------------------------------
np.random.seed(42)
n_obs      = 400
n_lots     = 20
n_types    = 4
n_packages = 3

R_const  = 0.001987    # kcal/mol·K
Temp_ref = 298.0       # 25°C

time   = np.random.uniform(0, 60, n_obs)
tempC  = np.random.choice([5, 25, 40], size=n_obs)
tempK  = tempC + 273.15

lot_idx     = np.random.randint(1, n_lots + 1, n_obs)      # 1-based for Stan
type_idx    = np.random.randint(1, n_types + 1, n_obs)
package_idx = np.random.randint(1, n_packages + 1, n_obs)

# True parameters
true_alpha_type = np.array([100.5, 99.8, 101.2, 100.1])
true_k_type     = np.array([0.08, 0.12, 0.06, 0.10])
true_E_star     = 22.0
true_sigma      = 0.6

arr = np.exp(true_E_star / R_const * (1/Temp_ref - 1/tempK))

alpha_lot = true_alpha_type[type_idx-1] + np.random.normal(0, 0.8, n_obs)
k_lot     = true_k_type[type_idx-1]     + np.random.normal(0, 0.02, n_obs)
k_lot     = np.maximum(k_lot, 1e-6)

mu_true = (alpha_lot 
           - k_lot * arr * time 
           + np.random.normal(0, 0.5, n_obs) * (package_idx - 2))  # small package effect

y = np.random.normal(mu_true, true_sigma)

data_df = pd.DataFrame({
    "y": y,
    "time": time,
    "tempK": tempK,
    "arrhenius": arr,
    "lot": lot_idx,
    "type": type_idx,
    "package": package_idx
})

# -------------------------------
# 2. Stan model (save as string)
# -------------------------------
stan_code = """
data {
  int<lower=1> N;           // number of observations
  int<lower=1> n_lot;       
  int<lower=1> n_type;
  int<lower=1> n_pkg;
  vector[N] y;              // observed potency
  vector[N] time;           // months
  vector[N] tempK;          
  vector[N] arr;            // pre-computed exp(E*/R * (1/Tref - 1/T))
  array[N] int<lower=1,upper=n_lot> lot;
  array[N] int<lower=1,upper=n_type> type;
  array[N] int<lower=1,upper=n_pkg>  pkg;
  
  real<lower=0> R;          // gas constant in kcal/mol·K
  real T_ref;               // 298 K
}

parameters {
  // Hyperparameters
  vector[n_type] alpha_type_raw;
  vector<lower=0>[n_type] k_type_raw;
  real E_star;                              // shared activation energy (kcal/mol)
  
  // Hyper standard deviations
  real<lower=0> sd_alpha_lot;
  real<lower=0> sd_k_lot;
  
  // Non-centered lot effects
  vector[n_lot] z_alpha_lot;
  vector[n_lot] z_k_lot;
  
  // Package effects
  vector[n_pkg] alpha_pkg;
  
  real<lower=0> sigma;                      // residual SD
}

transformed parameters {
  vector[n_type] alpha_type = 100.0 + 5.0 * alpha_type_raw;
  vector[n_type] k_type     = 0.2  + 0.1  * k_type_raw;   // ensures positivity
}

model {
  // Priors
  alpha_type_raw ~ std_normal();
  k_type_raw     ~ std_normal();
  E_star         ~ normal(20, 10);
  
  sd_alpha_lot   ~ normal(0, 2);
  sd_k_lot       ~ normal(0, 1);
  
  z_alpha_lot    ~ std_normal();
  z_k_lot        ~ std_normal();
  
  alpha_pkg      ~ normal(0, 2);
  sigma          ~ normal(0, 2);
  
  // Likelihood
  vector[N] mu;
  for (i in 1:N) {
    real alpha = alpha_type[type[i]] + sd_alpha_lot * z_alpha_lot[lot[i]];
    real k     = k_type[type[i]]     + sd_k_lot     * z_k_lot[lot[i]];
    mu[i] = alpha + alpha_pkg[pkg[i]] - k * arr[i] * time[i];
  }
  
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  real<lower=0> E_star_kcal = E_star;   // already in kcal/mol
  
  for (i in 1:N) {
    real alpha = alpha_type[type[i]] + sd_alpha_lot * z_alpha_lot[lot[i]];
    real k     = k_type[type[i]]     + sd_k_lot     * z_k_lot[lot[i]];
    real mu_i  = alpha + alpha_pkg[pkg[i]] - k * arr[i] * time[i];
    y_rep[i]   = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(y[i] | mu_i, sigma);
  }
}
"""

# Write to temporary file
with open("arrhenius_stability.stan", "w") as f:
    f.write(stan_code)

# -------------------------------
# 3. Compile and sample with CmdStanPy
# -------------------------------
model = CmdStanModel(stan_file="arrhenius_stability.stan")
print(model)

stan_data = {
    "N": n_obs,
    "n_lot": n_lots,
    "n_type": n_types,
    "n_pkg": n_packages,
    "y": y,
    "time": time,
    "tempK": tempK,
    "arr": arr,
    "lot": lot_idx,
    "type": type_idx,
    "pkg": package_idx,
    "R": R_const,
    "T_ref": Temp_ref
}

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=2000,
    iter_sampling=2000,
    seed=42,
    show_progress=True,
    adapt_delta=0.95,
    max_treedepth=12
)

# -------------------------------
# 4. Convert to ArviZ and analyze (exactly like PyMC)
# -------------------------------
idata = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive="y_rep",
    log_likelihood="log_lik"
)

print(az.summary(idata, 
                 var_names=["E_star", "alpha_type", "k_type", 
                            "sd_alpha_lot", "sd_k_lot", "sigma", "alpha_pkg"],
                 hdi_prob=0.95))

# Posterior predictive check
az.plot_ppc(idata, data_pairs={"y": "y_rep"}, num_pp_samples=100)
plt.title("Posterior Predictive Check – HPV Vaccine Stability (Stan)")
plt.show()

# Trace plots
az.plot_trace(idata, var_names=["E_star", "k_type", "sigma"])
plt.show()

# Activation energy posterior
print("\nActivation Energy E* (kcal/mol):")
print(az.summary(idata, var_names=["E_star"], hdi_prob=0.95))

# Type-specific degradation rate at 25°C
k_type_25C = idata.posterior["k_type"].mean(dim=["chain", "draw"])
print("\nEstimated k_type at 25°C (per month):")
for i in range(n_types):
    print(f"HPV Type {i+1}: {k_type_25C[i].values:.4f}")

# Cleanup
os.remove("arrhenius_stability.stan")
for f in fit.runset.stdout_files: os.remove(f) if os.path.exists(f) else None
for f in fit.runset.stderr_files: os.remove(f) if os.path.exists(f) else None