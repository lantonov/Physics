import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import jv

# ------------------------------------------------------------
# 1. Load the CSV visibility file
# ------------------------------------------------------------
csv_path = "C:\\Users\\lanto\\Documents\\SR1_M87_2017_095_hi_hops_netcal_StokesI.csv"
df = pd.read_csv(csv_path, comment="#", header=None)
df.columns = ["time", "T1", "T2", "u", "v", "amp", "phase_deg", "sigma"]

# Convert to complex visibilities
phase_rad = np.deg2rad(df["phase_deg"].values)
V = df["amp"].values * np.exp(1j * phase_rad)
sigma = df["sigma"].values
sigma[sigma == 0] = np.median(sigma[sigma > 0])

# Store u,v in their original units (likely Gλ)
# DO NOT scale by 1e-9 yet - keep baselines in native units
u_raw = df["u"].values
v_raw = df["v"].values

# Check the scale of baselines
print(f"u range: [{u_raw.min():.3e}, {u_raw.max():.3e}]")
print(f"v range: [{v_raw.min():.3e}, {v_raw.max():.3e}]")
print(f"Amplitude range: [{df['amp'].min():.4f}, {df['amp'].max():.4f}]")

# ------------------------------------------------------------
# 2. Define the mode-expanded ring model
# ------------------------------------------------------------
def ring_visibility(params, u, v, M=2):
    """
    params = [I0, R, Re(c1), Im(c1), Re(c2), Im(c2), ...]
    R is the ring radius in units matched to u,v
    I0 is total flux
    """
    I0 = params[0]
    R = params[1]
    
    # complex mode coefficients c_m
    c = []
    for m in range(1, M+1):
        re = params[2 + 2*(m-1)]
        im = params[3 + 2*(m-1)]
        c.append(re + 1j*im)
    
    rho = np.sqrt(u**2 + v**2)
    psi = np.arctan2(v, u)
    
    # Compute Bessel argument
    arg = 2*np.pi*R*rho
    
    # m=0 term with safe evaluation
    V_model = np.zeros_like(rho, dtype=np.complex128)
    
    # Check for valid arguments
    safe_mask = np.isfinite(arg) & (arg < 1e3)
    
    if np.any(safe_mask):
        V_model[safe_mask] = I0 * 2*np.pi*R * jv(0, arg[safe_mask])
        
        # higher modes
        for m in range(1, M+1):
            factor = (1j)**m
            V_model[safe_mask] += (c[m-1] * 2*np.pi*R * factor * 
                                    jv(m, arg[safe_mask]) * 
                                    np.exp(1j*m*psi[safe_mask]))
    
    return V_model

# ------------------------------------------------------------
# 3. Fit the model to the visibilities
# ------------------------------------------------------------
def residuals(params):
    V_model = ring_visibility(params, u_raw, v_raw, M=2)
    V_model = np.nan_to_num(V_model, nan=0.0, posinf=0.0, neginf=0.0)
    
    resid = np.concatenate([
        np.real(V_model - V) / sigma,
        np.imag(V_model - V) / sigma
    ])
    
    # Replace any remaining non-finite values
    resid = np.nan_to_num(resid, nan=1e10, posinf=1e10, neginf=-1e10)
    
    return resid

# Initial guess based on typical M87 parameters
# R should give arg ~ 1-10 for typical baselines
rho_typical = np.sqrt(np.median(u_raw**2 + v_raw**2))
R_init = 1.0 / rho_typical  # This gives 2πR*rho ~ 6.28 at median baseline

# I0 should be close to the mean amplitude
I0_init = np.median(df["amp"].values)

print(f"\nInitial guesses:")
print(f"I0 = {I0_init:.4f}")
print(f"R  = {R_init:.4e}")
print(f"Typical Bessel arg = {2*np.pi*R_init*rho_typical:.2f}")

# Initial parameter vector: I0, R, Re(c1), Im(c1), Re(c2), Im(c2)
p0 = np.array([I0_init, R_init, 0.01, 0.01, 0.01, 0.01])

# Test initial residuals
try:
    r0 = residuals(p0)
    print(f"Initial residual stats: min={np.min(r0):.4f}, max={np.max(r0):.4f}, finite={np.isfinite(r0).all()}")
except Exception as e:
    print(f"Error computing initial residuals: {e}")

# Set bounds to keep parameters physical
bounds_lower = [0.0, R_init*0.1, -1.0, -1.0, -1.0, -1.0]
bounds_upper = [I0_init*10, R_init*10, 1.0, 1.0, 1.0, 1.0]

print("\nRunning least squares fit...")
res = least_squares(residuals, p0, bounds=(bounds_lower, bounds_upper), 
                    verbose=2, max_nfev=1000)

params = res.x
I0, R = params[0], params[1]
c1 = params[2] + 1j*params[3]
c2 = params[4] + 1j*params[5]

print("\n" + "="*60)
print("FITTED PARAMETERS:")
print("="*60)
print(f"I0 = {I0:.6f} (total flux)")
print(f"R  = {R:.6e} (ring radius in baseline units)")
print(f"   → characteristic Bessel arg at median baseline: {2*np.pi*R*rho_typical:.2f}")

print("\nMODE COEFFICIENTS:")
print(f"c1 = {c1.real:.6f} + {c1.imag:.6f}i")
print(f"   |c1| = {abs(c1):.6f}")
print(f"   arg(c1) = {np.angle(c1, deg=True):.2f}°")

print(f"\nc2 = {c2.real:.6f} + {c2.imag:.6f}i")
print(f"   |c2| = {abs(c2):.6f}")
print(f"   arg(c2) = {np.angle(c2, deg=True):.2f}°")

print(f"\nFit quality:")
print(f"Cost function: {res.cost:.4f}")
print(f"Optimality: {res.optimality:.4e}")

# ------------------------------------------------------------
# 4. GR vs Glome discriminator
# ------------------------------------------------------------
print("\n" + "="*60)
print("GR vs GLOME INTERPRETATION:")
print("="*60)

m1_threshold = 0.05
m2_dominance = abs(c2) / (abs(c1) + 1e-10)

print(f"\nm=1 asymmetry amplitude: |c1| = {abs(c1):.4f}")
if abs(c1) > m1_threshold:
    print(f"  → m=1 is STRONG (>{m1_threshold}) → GR-like asymmetry detected")
    print(f"  → Suggests photon ring brightness asymmetry")
else:
    print(f"  → m=1 is SUPPRESSED (<{m1_threshold}) → Glome-like symmetry")
    print(f"  → Consistent with symmetric embedding space")

print(f"\nm=2 quadrupole amplitude: |c2| = {abs(c2):.4f}")
print(f"Ratio |c2|/|c1| = {m2_dominance:.2f}")

if abs(c2) > abs(c1):
    print(f"  → m=2 DOMINATES over m=1 → consistent with Glome prediction")
    print(f"  → Quadrupolar symmetry from higher-dimensional embedding")
else:
    print(f"  → m=2 does NOT dominate → more GR-like")
    print(f"  → Dipole asymmetry characteristic of Kerr spacetime")

print(f"\nPhase coherence check:")
print(f"arg(c1) = {np.angle(c1, deg=True):.1f}° (GR: variable with rotation)")
print(f"arg(c2) = {np.angle(c2, deg=True):.1f}° (Glome: should be phase-locked)")

print("\n" + "="*60)
print("MULTI-EPOCH ANALYSIS RECOMMENDATIONS:")
print("="*60)
print("For discrimination across observation epochs:")
print("  1. Track |c1|(t): GR predicts variability, Glome predicts stability")
print("  2. Track |c2|(t): Glome predicts constancy, GR allows evolution")
print("  3. Monitor arg(c2): Glome predicts phase-locking, GR allows drift")
print("  4. Compute ratio |c2|/|c1| vs time: Glome → stable & >1, GR → variable")
print("  5. Cross-correlate with jet activity: GR coupling expected, Glome decoupled")