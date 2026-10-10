# ------------------------------------------------------------
# Crescent fit with FREE Rout and wide offset bounds
# ------------------------------------------------------------

def crescent_visibility_free(params, u, v):
    """
    params = [I0, Rout, f, dalpha, dbeta]
    """
    I0, Rout, f, dalpha, dbeta = params
    Rin = f * Rout

    rho = np.sqrt(u**2 + v**2)
    phase_shift = np.exp(-2j * np.pi * (u * dalpha + v * dbeta))

    x_out = 2 * np.pi * Rout * rho
    x_in  = 2 * np.pi * Rin  * rho

    x_out = np.where(x_out == 0, 1e-20, x_out)
    x_in  = np.where(x_in  == 0, 1e-20, x_in)

    V_out = 2 * j1(x_out) / x_out
    V_in  = 2 * j1(x_in)  / x_in

    return I0 * (V_out - phase_shift * V_in)

def residuals_free(params):
    V_model = crescent_visibility_free(params, u, v)
    res = np.concatenate([
        (V_model.real - V.real) / sigma,
        (V_model.imag - V.imag) / sigma
    ])
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

# initial guesses
Rout_init = 20e-6 * np.pi / (180 * 3600)  # 20 μas
p0 = np.array([I0_init, Rout_init, 0.5, 0.0, 0.0])

# bounds:
# Rout allowed from 5–50 μas
Rout_min = 5e-6  * np.pi / (180 * 3600)
Rout_max = 50e-6 * np.pi / (180 * 3600)

bounds_lower = [0.0, Rout_min, 0.1, -2e-9, -2e-9]
bounds_upper = [I0_init*10, Rout_max, 0.99,  2e-9,  2e-9]

print("Running crescent fit with FREE R_out and wide offset bounds...")
res = least_squares(residuals_free, p0, bounds=(bounds_lower, bounds_upper),
                    verbose=2, max_nfev=800)

I0, Rout, f, dalpha, dbeta = res.x
Rin = f * Rout

print("\n=== Crescent Fit Results (Free R_out, Wide Offsets) ===")
print(f"I0      = {I0:.4f} Jy")
print(f"R_out   = {Rout:.3e} rad")
print(f"R_in    = {Rin:.3e} rad (f = {f:.3f})")
print(f"offset  = ({dalpha:.3e}, {dbeta:.3e}) rad")
print(f"cost    = {res.cost:.3e}")
print(f"optimality = {res.optimality:.3e}")
