import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, norm, invgamma, multivariate_normal
from scipy.special import erfinv  # For inverse normal CDF

# Set seed
np.random.seed(42)

# Parameters
n_lots = 5
n_impurities = 3

# True means (ppm)
mu_ref = np.array([2.0, 2.0, 2.0])
mu_bio = np.array([2.4, 2.4, 2.4])

# Shape alpha
alpha = np.full(n_impurities, 3.0)

# Rate beta
beta_ref = alpha / mu_ref
beta_bio = alpha / mu_bio

# Correlation matrix
R = np.array([[1.0, 0.6, 0.4],
              [0.6, 1.0, 0.5],
              [0.4, 0.5, 1.0]])

# Simulate Z ~ MVN(0, R)
Z_ref = multivariate_normal(mean=np.zeros(n_impurities), cov=R).rvs(n_lots)
Z_bio = multivariate_normal(mean=np.zeros(n_impurities), cov=R).rvs(n_lots)

# To uniform U = Phi(Z)
U_ref = norm.cdf(Z_ref)
U_bio = norm.cdf(Z_bio)

# Inverse CDF for Gamma
imp_ref = np.zeros((n_lots, n_impurities))
imp_bio = np.zeros((n_lots, n_impurities))
for k in range(n_impurities):
    imp_ref[:, k] = gamma.ppf(U_ref[:, k], a=alpha[k], scale=1/beta_ref[k])
    imp_bio[:, k] = gamma.ppf(U_bio[:, k], a=alpha[k], scale=1/beta_bio[k])

# Simple Bayesian analysis proxy (Metropolis for means, InvGamma for vars)
# Assume fixed alpha, estimate beta_j ~ Normal prior, lot var tau^2 ~ InvGamma
# For illustration; full NUTS would use Stan/PyMC

n_iter = 5000
burn_in = 1000

# Priors
mu_prior_mean = np.full(n_impurities, 2.0)
mu_prior_sd = np.full(n_impurities, 1.0)
tau_shape = 2.0
tau_rate = 0.1

# Initialize
mu_r = mu_prior_mean.copy()
mu_b = mu_prior_mean.copy() + 0.2  # Slight shift init
tau_sq = invgamma.rvs(tau_shape, scale=tau_rate, size=n_impurities)

# Log-likelihood function
def log_lik(imp, mu, tau_sq, alpha, n_lots):
    # Approximate lot effects as normal for proxy
    ll = 0
    for k in range(n_impurities):
        beta = alpha[k] / mu[k]
        ll += np.sum(gamma.logpdf(imp[:, k], a=alpha[k], scale=1/beta))
        # Prior on lot means (simplified)
        ll += norm.logpdf(mu[k], loc=mu_prior_mean[k], scale=mu_prior_sd[k])
        # Lot var contribution (pseudo)
        lot_var = np.var(imp[:, k])
        ll += invgamma.logpdf(tau_sq[k], a=tau_shape, scale=tau_rate + lot_var * n_lots / 2)
    return ll

# Metropolis steps
mu_r_chain = np.zeros((n_iter, n_impurities))
mu_b_chain = np.zeros((n_iter, n_impurities))
tau_sq_chain = np.zeros((n_iter, n_impurities))
accept_r = np.zeros(n_impurities)
accept_b = np.zeros(n_impurities)
accept_tau = np.zeros(n_impurities)

current_ll_r = log_lik(imp_ref, mu_r, tau_sq, alpha, n_lots)
current_ll_b = log_lik(imp_bio, mu_b, tau_sq, alpha, n_lots)

proposal_sd = 0.1  # Tuning param

for i in range(n_iter):
    # Propose for mu_r
    for k in range(n_impurities):
        mu_r_prop = mu_r.copy()
        mu_r_prop[k] += np.random.normal(0, proposal_sd)
        prop_ll = log_lik(imp_ref, mu_r_prop, tau_sq, alpha, n_lots)
        if np.log(np.random.rand()) < prop_ll - current_ll_r:
            mu_r = mu_r_prop
            current_ll_r = prop_ll
            accept_r[k] += 1
    mu_r_chain[i] = mu_r

    # Propose for mu_b
    for k in range(n_impurities):
        mu_b_prop = mu_b.copy()
        mu_b_prop[k] += np.random.normal(0, proposal_sd)
        prop_ll = log_lik(imp_bio, mu_b_prop, tau_sq, alpha, n_lots)
        if np.log(np.random.rand()) < prop_ll - current_ll_b:
            mu_b = mu_b_prop
            current_ll_b = prop_ll
            accept_b[k] += 1
    mu_b_chain[i] = mu_b

    # Update tau_sq (InvGamma proposal approximation)
    for k in range(n_impurities):
        tau_sq_prop = invgamma.rvs(tau_shape + n_lots/2, scale=tau_rate + np.var(imp_ref[:,k])/2 + np.var(imp_bio[:,k])/2)
        prop_ll_r = log_lik(imp_ref, mu_r, tau_sq_prop * np.ones_like(tau_sq), alpha, n_lots)
        prop_ll_b = log_lik(imp_bio, mu_b, tau_sq_prop * np.ones_like(tau_sq), alpha, n_lots)
        if np.log(np.random.rand()) < (prop_ll_r + prop_ll_b) - (current_ll_r + current_ll_b):
            tau_sq[k] = tau_sq_prop
            current_ll_r = prop_ll_r
            current_ll_b = prop_ll_b
            accept_tau[k] += 1
    tau_sq_chain[i] = tau_sq

# Burn-in
mu_r_chain = mu_r_chain[burn_in:]
mu_b_chain = mu_b_chain[burn_in:]
tau_sq_chain = tau_sq_chain[burn_in:]

# Summary
print("\nPosterior Means (Reference):", np.mean(mu_r_chain, axis=0).round(2))
print("Posterior Means (Biosimilar):", np.mean(mu_b_chain, axis=0).round(2))
print("Posterior Tau (SD):", np.sqrt(np.mean(tau_sq_chain, axis=0)).round(2))

# Visualizations
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for k in range(n_impurities):
    sns.histplot(mu_r_chain[:, k], kde=True, ax=axs[k], label='Reference', color='blue')
    sns.histplot(mu_b_chain[:, k], kde=True, ax=axs[k], label='Biosimilar', color='red')
    axs[k].set_title(f'Posterior for Mean Imp_{k+1}')
    axs[k].legend()
plt.tight_layout()
plt.show()

# Difference
delta_mu = mu_b_chain - mu_r_chain
print("\nMean Difference:", np.mean(delta_mu, axis=0).round(2))
print("95% CrI Lower:", np.percentile(delta_mu, 2.5, axis=0).round(2))
print("95% CrI Upper:", np.percentile(delta_mu, 97.5, axis=0).round(2))

# Similarity Prob
margin = 0.5  # ppm
prob_similar = np.mean(np.abs(np.mean(delta_mu, axis=0)) < margin)
print(f"P(|Delta mu| < {margin}): {prob_similar:.2f}")