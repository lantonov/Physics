# =============================================================================
# FULLY SELF-CONTAINED HPV Vaccine Stability Model in CmdStanPy
# Realistic synthetic data INCLUDED directly in the script
# =============================================================================

from cmdstanpy import CmdStanModel
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# 1. The exact same Stan model (saved as string)
# ------------------------------------------------------------------
stan_model = """
data {
  int<lower=1> N;
  int<lower=1> n_lot;
  int<lower=1> n_type;
  int<lower=1> n_pkg;
  
  vector[N] y;              // observed potency (% label claim)
  vector[N] time;           // storage time in months
  vector[N] tempK;          // storage temperature in Kelvin
  vector[N] arr;            // pre-computed Arrhenius factor exp(E*/R*(1/Tref-1/T))
  
  array[N] int<lower=1,upper=n_lot>   lot;
  array[N] int<lower=1,upper=n_type>  type;
  array[N] int<lower=1,upper=n_pkg>   pkg;
  
  real<lower=0> R_const;    // gas constant = 0.001987 kcal/mol·K
  real          T_ref;      // reference temperature 298 K (25°C)
}

parameters {
  vector[n_type] alpha_type_raw;
  vector<lower=0>[n_type] k_type_raw;
  real E_star;
  
  real<lower=0> sd_alpha_lot;
  real<lower=0> sd_k_lot;
  
  vector[n_lot] z_alpha_lot;
  vector[n_lot] z_k_lot;
  
  vector[n_pkg] alpha_pkg;
  
  real<lower=0> sigma;
}

transformed parameters {
  vector[n_type] alpha_type = 100.0 + 5.0 * alpha_type_raw;
  vector[n_type] k_type     = 0.15 + 0.1 * k_type_raw;   // positive degradation rates
}

model {
  alpha_type_raw ~ std_normal();
  k_type_raw     ~ std_normal();
  E_star         ~ normal(20, 10);
  
  sd_alpha_lot   ~ normal(0, 2);
  sd_k_lot       ~ normal(0, 1);
  
  z_alpha_lot ~ std_normal();
  z_k_lot     ~ std_normal();
  
  alpha_pkg   ~ normal(0, 2);
  sigma       ~ normal(0, 2);
  
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
  
  for (i in 1:N) {
    real alpha = alpha_type[type[i]] + sd_alpha_lot * z_alpha_lot[lot[i]];
    real k     = k_type[type[i]]     + sd_k_lot     * z_k_lot[lot[i]];
    real mu_i  = alpha + alpha_pkg[pkg[i]] - k * arr[i] * time[i];
    y_rep[i]   = normal_rng(mu_i, sigma);
    log_lik[i] = normal_lpdf(y[i] | mu_i, sigma);
  }
}
"""

# ------------------------------------------------------------------
# 2. Compile the model
# ------------------------------------------------------------------
model = CmdStanModel(stan_code=stan_model)

# ------------------------------------------------------------------
# 3. REALISTIC EXAMPLE DATA (directly in Python → passed to Stan)
# ------------------------------------------------------------------
np.random.seed(123)

N       = 400
n_lot   = 20
n_type  = 4
n_pkg   = 3

# True values (for simulation only – Stan will recover them)
true_alpha_type = [100.4, 99.9, 101.1, 100.2]
true_k_type     = [0.07, 0.11, 0.06, 0.09]   # per month at 25°C
true_E_star     = 23.5                        # kcal/mol
R_const         = 0.001987
T_ref           = 298.0

# Design
time  = np.random.uniform(0, 60, N)
tempC = np.random.choice([5, 25, 40], size=N, p=[0.4, 0.4, 0.2])
tempK = tempC + 273.15

lot   = np.random.randint(1, n_lot+1, N)
typ   = np.random.randint(1, n_type+1, N)
pkg   = np.random.randint(1, n_pkg+1, N)

# Pre-computed Arrhenius factor using the *true* E*
arr = np.exp(true_E_star / R_const * (1/T_ref - 1/tempK))

# Simulate potency
alpha = np.array(true_alpha_type)[typ-1] + np.random.normal(0, 0.8, N)
k     = np.array(true_k_type)[typ-1]     + np.random.normal(0, 0.015, N)
k     = np.maximum(k, 1e-8)

mu    = (alpha 
         + 0.4*(pkg-2)                                    # small package effect
         - k * arr * time)

y = np.random.normal(mu, 0.55)

# ------------------------------------------------------------------
# 4. Stan data dictionary – everything is here!
# ------------------------------------------------------------------
stan_data = {
    "N": N,
    "n_lot": n_lot,
    "n_type": n_type,
    "n_pkg": n_pkg,
    "y": y.tolist(),
    "time": time.tolist(),
    "tempK": tempK.tolist(),
    "arr": arr.tolist(),
    "lot": lot.tolist(),
    "type": typ.tolist(),
    "pkg": pkg.tolist(),
    "R_const": R_const,
    "T_ref": T_ref
}

# ------------------------------------------------------------------
# 5. Sample
# ------------------------------------------------------------------
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1500,
    iter_sampling=2000,
    seed=123,
    adapt_delta=0.95,
    max_treedepth=12
)

# ------------------------------------------------------------------
# 6. Results (ArviZ – identical workflow to PyMC)
# ------------------------------------------------------------------
idata = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive="y_rep",
    log_likelihood="log_lik"
)

print(az.summary(idata,
                 var_names=["E_star", "alpha_type", "k_type",
                            "sd_alpha_lot", "sd_k_lot", "sigma"],
                 hdi_prob=0.95).round(3))

# Recovery check
print("\nTrue E* was 23.5 → posterior 94% HDI covers it?")
print(az.summary(idata, var_names=["E_star"], hdi_prob=0.94))

# Posterior predictive check
az.plot_ppc(idata, num_pp_samples=80)
plt.title("PPC – Arrhenius Vaccine Stability Model (CmdStanPy)")
plt.show()

# Trace plot of the most important parameter
az.plot_trace(idata, var_names=["E_star", "k_type"])
plt.show()