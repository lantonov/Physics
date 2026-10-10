import pymc as pm
import numpy as np
import arviz as az
import pandas as pd

# Set a random seed for reproducibility, crucial for peer-review
np.random.seed(42)

print(f"Running on PyMC v{pm.__version__}")
print(f"Running on ArviZ v{az.__version__}")

# ---
## 1. Setup: Simulation Parameters (from Section 5.2)
# ---

# Define CQA names
cqa_names = ['Potency', '%HMW', 'FcRn_Binding']
k_dims = 3  # Number of CQAs

# A. Reference Product (Avastin) Parameters (Knowns)
mu_r = np.array([100.0, 1.5, 100.0])  # mu_r
sigma_r = np.array([5.0, 0.14, 3.0])
n_r = 56.0  # Number of reference lots

# B. Biosimilar (MVASI) "True" Parameters (for Data Simulation)
mu_b_true = np.array([102.0, 1.6, 101.0])  # "True" shifted mean
sigma_b = np.array([5.5, 0.15, 3.2])     # From sqrt([30.25, 0.0225, 10.24])
n_b = 13.0  # Number of biosimilar lots

# C. Equivalence Margins (Deltas)
# Tier 1: Potency (delta = 1.5 * sigma_r)
# Tier 2: %HMW (QR = 3 * sigma_r -> delta = 3 * 0.14 = 0.42)
# Tier 2: FcRn (QR = 3 * sigma_r -> delta = 3 * 3.0 = 9.0)
deltas = np.array([7.5, 0.42, 9.0])

# D. Construct Covariance Matrices (assuming rho=0.3 as per text)
rho = 0.3
R = np.array([[1.0, rho, rho],
              [rho, 1.0, rho],
              [rho, rho, 1.0]])

# Sigma_r (Reference Covariance)
D_r = np.diag(sigma_r)
Sigma_r = D_r @ R @ D_r

# Sigma_b (Biosimilar Covariance - for data generation and as "known" in model)
D_b = np.diag(sigma_b)
Sigma_b_true = D_b @ R @ D_b

print("--- Setup Complete ---")
print(f"Reference Mean (mu_r):\n{mu_r}\n")
print(f"Reference Cov (Sigma_r):\n{Sigma_r}\n")
print(f"Biosimilar True Mean (mu_b_true):\n{mu_b_true}\n")
print(f"Biosimilar True Cov (Sigma_b_true):\n{Sigma_b_true}\n")
print(f"Equivalence Margins (delta):\n{deltas}\n")

# ---
## 2. Data Generation
# ---
# Simulate the n_b=13 biosimilar lots from the "true" shifted distribution
Y_b_data = np.random.multivariate_normal(
    mean=mu_b_true,
    cov=Sigma_b_true,
    size=int(n_b)
)

print(f"Generated {Y_b_data.shape[0]} biosimilar lots (data points).")
print(f"Sample Mean (bar{Y_b_data.mean(axis=0)}).")

# ---
## 3. Define Priors (from Section 5.2)
# ---
# These are the three prior covariance matrices (Psi) for the mean mu_b
# Psi = MVN(mu_r, Psi)

# 1. Strongly Informative (a=1, full borrowing)
Psi_strong = Sigma_r / n_r

# 2. Weakly Informative (a=0.1, 10x variance inflation)
Psi_weak = (10.0 * Sigma_r) / n_r

# 3. Non-Informative (Jeffreys-like flat prior)
Psi_non = np.eye(k_dims) * 1e6  # Large variance = low information

# Store in a dictionary for a clean loop
priors = {
    "Strongly Informative": Psi_strong,
    "Weakly Informative": Psi_weak,
    "Non-Informative": Psi_non
}

# ---
## 4. Define and Run Bayesian Models (PyMC)
# ---

# This function defines the exact model from the rewrite:
# A multivariate-normal prior on the mean (mu_b)
# A multivariate-normal likelihood with a *known* covariance (Sigma_b_true)
# This is a conjugate model, but we use MCMC for illustrative power.

def build_bayesian_model(data, prior_mean, prior_cov, known_likelihood_cov):
    """
    Builds the PyMC model as defined in the case study.
    
    Model:
        mu_b ~ MVN(mu_r, Psi)
        Y_b | mu_b ~ MVN(mu_b, Sigma_b_known)
    """
    with pm.Model() as model:
        # Prior for the unknown biosimilar mean (mu_b)
        # mu_b ~ MVN(mu_r, Psi)
        mu_b = pm.MvNormal(
            'mu_b', 
            mu=prior_mean, 
            cov=prior_cov, 
            shape=k_dims
        )
        
        # Likelihood
        # The observed data Y_b is distributed around mu_b
        # with the *known* process covariance Sigma_b
        # Y_b ~ MVN(mu_b, Sigma_b_known)
        y_obs = pm.MvNormal(
            'y_obs',
            mu=mu_b,
            cov=known_likelihood_cov,
            observed=data
        )
        
    return model

# Dictionary to store results
results = {}

for name, psi_matrix in priors.items():
    print(f"\n--- Running Model: {name} ---")
    
    # 1. Build the model
    model = build_bayesian_model(
        data=Y_b_data,
        prior_mean=mu_r,
        prior_cov=psi_matrix,
        known_likelihood_cov=Sigma_b_true
    )
    
    # 2. Run the MCMC sampler (NUTS)
    # This is the modern, "peer-review" standard
    with model:
        idata = pm.sample(
            draws=4000, 
            tune=2000, 
            chains=4, 
            cores=1,
            target_accept=0.95
        )
    
    # 3. Store results
    results[name] = idata
    print(f"--- Model {name} Complete ---")


# ---
## 5. Posterior Analysis and Equivalence Calculation
# ---

print("\n\n" + "="*50)
print(" Bayesian Model Results")
print("="*50)

summary_frames = []

for name, idata in results.items():
    print(f"\n--- Analysis for: {name} Prior ---")
    
    # Get the posterior trace for mu_b (the estimated biosimilar mean)
    # This is a 2D array: (n_chains * n_draws, k_dims)
    posterior_mu_b = az.extract(idata, var_names=["mu_b"])
    
    # A. Calculate Posterior Statistics (Mean and SD)
    # This shows the "shrinkage" effect
    post_mean = posterior_mu_b.mean(axis=0)
    post_std = posterior_mu_b.std(axis=0)
    
    print(f"  Posterior Mean (est. mu_b): {np.round(post_mean, 2)}")
    print(f"  Posterior Std. Dev:       {np.round(post_std, 2)}")
    
    # B. Calculate Equivalence Probabilities
    prob_list = []
    joint_bool = np.full(posterior_mu_b.shape[0], True)

    for i in range(k_dims):
        # Calculate the difference between posterior draws and the reference mean
        diff = posterior_mu_b[:, i] - mu_r[i]
        
        # Find all draws that fall *inside* the equivalence margin
        is_equivalent_bool = (diff > -deltas[i]) & (diff < deltas[i])
        
        # The probability is the mean of this boolean array
        prob = is_equivalent_bool.mean()
        
        prob_list.append(prob)
        joint_bool = joint_bool & is_equivalent_bool

    # The joint probability is the % of MCMC draws where ALL CQAs
    # were simultaneously within their margins
    joint_prob = joint_bool.mean()

    print("\n  Equivalence Probabilities P(|mu_b - mu_r| < delta | data):")
    for i in range(k_dims):
        print(f"    {cqa_names[i]:<13}: {prob_list[i]:.4f} (Margin: +/- {deltas[i]})")
    
    print(f"    ---------------------------------")
    print(f"    JOINT PROBABILITY: {joint_prob:.4f}")
    
    # Store for final summary table
    summary_data = {
        'Prior': name,
        'Post. Mean (Potency)': post_mean[0],
        'Post. SD (Potency)': post_std[0],
        'P(Equiv) (Potency)': prob_list[0],
        'Post. Mean (%HMW)': post_mean[1],
        'Post. SD (%HMW)': post_std[1],
        'P(Equiv) (%HMW)': prob_list[1],
        'Post. Mean (FcRn)': post_mean[2],
        'Post. SD (FcRn)': post_std[2],
        'P(Equiv) (FcRn)': prob_list[2],
        'P(Joint Equiv)': joint_prob
    }
    summary_frames.append(pd.DataFrame([summary_data]))

# ---
## 6. Final Summary Table & Visualization
# ---

# Combine all results into a single "peer-review ready" table
final_summary = pd.concat(summary_frames).set_index('Prior')

print("\n\n" + "="*50)
print(" Final Comparison Summary")
print("="*50)
print(final_summary.to_markdown(floatfmt=".3f"))

# Visualize the shrinkage
# This plot clearly shows how the strongly informative prior
# "pulls" the posterior estimate away from the data and
# towards the reference mean (100).

print("\n\nVisualizing posterior 'shrinkage' for Potency (mu_r = 100):")
az.plot_forest(
    list(results.values()),
    model_names=list(results.keys()),
    var_names=["mu_b"],
    coords={'mu_b_dim_0': 0}, # Plot only the first CQA (Potency)
    hdi_prob=0.9, # Corresponds to the 90% CI for TOST
    figsize=(10, 4)
);