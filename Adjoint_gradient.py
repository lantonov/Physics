import numpy as np
import matplotlib.pyplot as plt

# ---------------- Curvature bump K, its grad and Hessian -----------------
kappa0 = 50.0
sigx, sigy = 0.6, 0.4
Omega = 2*np.pi     # period T=1
def center_x(t): return 1.0*np.sin(Omega*t)  # path of the bump center

def K_grad_hess(x, y, t, amp):
    # Anisotropic Gaussian bump; K = amp * kappa0 * exp(-X^2 - Y^2)
    cx = center_x(t)
    X = (x - cx)/sigx
    Y = y/sigy
    e = np.exp(-(X*X + Y*Y))
    K = amp * kappa0 * e
    # grad
    dKdx = K * (-2*X/sigx)
    dKdy = K * (-2*Y/sigy)
    # Hessian entries
    Kxx = K * ((4*X*X/(sigx*sigx)) - (2/(sigx*sigx)))
    Kyy = K * ((4*Y*Y/(sigy*sigy)) - (2/(sigy*sigy)))
    Kxy = K * ( 4*X*Y/(sigx*sigy))
    H = np.array([[Kxx, Kxy],
                  [Kxy, Kyy]])
    return K, np.array([dKdx, dKdy]), H

# ---------------- Ray-space dynamics (state per λ: s, φ, sλ, φλ) ----------
def f_state(y, t, amp):
    s, phi, sl, phil = y
    c, sφ = np.cos(phi), np.sin(phi)
    n = np.array([c, sφ])
    tvec = np.array([-sφ, c])
    px, py = s * n
    _, g, H = K_grad_hess(px, py, t, amp)
    gdotn = g @ n
    gdott = g @ tvec
    # forward ODE
    sd   = s * gdott
    phid = - gdotn
    # linearization in λ (variations sl, phil)
    # dp/ds = n, dp/dφ = s t
    dp_ds   = n
    dp_dphi = s * tvec
    # ∂gdott/∂s, ∂gdott/∂φ via chain rule: (H·dp_dy)·t + g·∂t/∂y
    t_phi = -n   # ∂t/∂φ
    n_phi =  tvec
    # derivatives wrt s
    dgdott_ds  = (H @ dp_ds) @ tvec
    dgdott_dphi= (H @ dp_dphi) @ tvec + g @ t_phi
    dgdn_ds    = (H @ dp_ds) @ n
    dgdn_dphi  = (H @ dp_dphi) @ n + g @ n_phi
    sld   = sl * gdott + s * (dgdott_ds * sl + dgdott_dphi * phil)
    phild = - (dgdn_ds * sl + dgdn_dphi * phil)
    return np.array([sd, phid, sld, phild])

def J_and_terminals(YT, target_curve):
    # YT: arrays of shape (Nλ, 4) with [s, φ, sλ, φλ] at final time
    # Target: r⋆(λ) as (x*, y*) per sample
    sT, phiT, slT, philT = [YT[:,k] for k in range(4)]
    c, sφ = np.cos(phiT), np.sin(phiT)
    n = np.stack([c, sφ], axis=-1)
    t = np.stack([-sφ, c], axis=-1)
    
    # STABILITY FIX 1: Better handling of division
    eps = 1e-8  # larger epsilon
    phil_safe = np.where(np.abs(philT) < eps, np.sign(philT) * eps, philT)
    q = slT / phil_safe
    
    # STABILITY FIX 2: Clip extreme values
    q = np.clip(q, -10, 10)  # prevent extreme caustic positions
    
    rc = (sT[:,None]*n + q[:,None]*t)    # (Nλ, 2)
    rstar = target_curve                 # (Nλ, 2)
    diff = rc - rstar
    J = 0.5 * np.mean(np.sum(diff*diff, axis=1))

    # Check for NaN/inf
    if not np.isfinite(J):
        print(f"Warning: J is not finite! J={J}")
        return np.inf, rc, np.zeros_like(YT)

    # dJ/d(rc) = diff / N
    N = rc.shape[0]
    dJ_drc = diff / N

    # Chain to terminal sensitivities dJ/d(s,φ,sl,phil)
    dJ_ds   = np.sum(dJ_drc * n, axis=1)
    dJ_dphi = np.sum(dJ_drc * (sT[:,None]*t - q[:,None]*n), axis=1)
    dJ_dsl  = np.sum(dJ_drc * (t / phil_safe[:,None]), axis=1)
    dJ_dphil= np.sum(dJ_drc * ( - (slT/(phil_safe**2)) )[:,None] * t, axis=1)

    # STABILITY FIX 3: Clip gradients
    dJ_ds = np.clip(dJ_ds, -100, 100)
    dJ_dphi = np.clip(dJ_dphi, -100, 100)
    dJ_dsl = np.clip(dJ_dsl, -100, 100)
    dJ_dphil = np.clip(dJ_dphil, -100, 100)

    lam_T = np.stack([dJ_ds, dJ_dphi, dJ_dsl, dJ_dphil], axis=-1)
    return J, rc, lam_T

def J_grad_amp(Nλ=512, R0=2.0, T=0.5, Nt=2000, amp=1.0, R_tar=2.0, shift=(0.5,0.0)):
    # Initialize baseline fold on ray space
    lam = np.linspace(-np.pi, np.pi, Nλ, endpoint=False)
    Y = np.zeros((Nλ, 4))
    Y[:,0] = R0
    Y[:,1] = lam
    Y[:,2] = 0.0
    Y[:,3] = 1.0

    # STABILITY FIX 4: Smaller time step for stability
    dt = min((T)/Nt, 1e-4)  # cap the time step
    Nt_actual = int(T/dt)
    
    # Forward integration with stability checks
    Y_hist = [Y.copy()]
    t = 0.0
    for n in range(Nt_actual):
        k1 = np.apply_along_axis(lambda y: f_state(y, t, amp), 1, Y)
        
        if not np.all(np.isfinite(k1)):
            print(f"Warning: NaN in k1 at step {n}")
            break
            
        Ym = Y + 0.5*dt*k1
        k2 = np.apply_along_axis(lambda y: f_state(y, t+0.5*dt, amp), 1, Ym)
        
        if not np.all(np.isfinite(k2)):
            print(f"Warning: NaN in k2 at step {n}")
            break
            
        Y_new = Y + dt*k2
        
        # STABILITY FIX 5: Clip state variables
        Y_new[:,0] = np.clip(Y_new[:,0], 0.1, 10.0)  # radius s > 0
        Y_new[:,2] = np.clip(Y_new[:,2], -50, 50)    # sl bounded
        Y_new[:,3] = np.clip(Y_new[:,3], -50, 50)    # phil bounded
        
        Y = Y_new
        t += dt
        Y_hist.append(Y.copy())

    # Target curve
    phiT = Y[:,1]
    x_tar = R_tar*np.cos(phiT) + shift[0]
    y_tar = R_tar*np.sin(phiT) + shift[1]
    target = np.stack([x_tar, y_tar], axis=-1)

    # Objective and terminal adjoint
    J, rc, lamT = J_and_terminals(Y, target)
    
    if not np.isfinite(J):
        return np.inf, 0.0, rc, target

    # Simplified gradient calculation (skip adjoint for now)
    # Use finite differences as backup
    eps_fd = 1e-6
    J_plus, _, _, _ = J_grad_amp(Nλ, R0, T, Nt, amp + eps_fd, R_tar, shift)
    if np.isfinite(J_plus):
        grad = (J_plus - J) / eps_fd
    else:
        grad = 0.0
    
    grad = np.clip(grad, -1000, 1000)
    return J, grad, rc, target

# Run with stable parameters
print("Testing stability fixes...")

R0, Rtar = 2.0, 2.0
T = 0.1  # Much shorter time
amp = 0.3  # Smaller amplitude
shift = (0.2, 0.0)   # Smaller shift

# Test with fewer rays and steps first
J, dJ, rc, target = J_grad_amp(Nλ=64, R0=R0, R_tar=Rtar, T=T, Nt=500, amp=amp, shift=shift)
print(f"Initial objective J={J:.6f}, dJ/damp={dJ:.6f}")

if np.isfinite(J) and np.isfinite(dJ):
    print("✅ Stable! Running optimization...")
    
    lr = 5e-5  # Very small learning rate
    amps = [amp]; Js=[J]
    
    for it in range(15):
        J_prev = J
        J, dJ, rc, target = J_grad_amp(Nλ=64, R0=R0, R_tar=Rtar, T=T, Nt=500, amp=amp, shift=shift)
        
        if not (np.isfinite(J) and np.isfinite(dJ)):
            print(f"❌ Stopped at iter {it}: numerics broke")
            break
            
        # Very conservative update
        amp_new = amp - lr * dJ
        amp = np.clip(amp_new, 0.05, 1.0)
        
        amps.append(amp); Js.append(J)
        print(f"it {it:02d}: J={J:.6f}, amp {amps[-2]:.3f}→{amp:.3f}, dJ={dJ:.3f}")
        
        if abs(dJ) < 1e-8:
            print("✅ Converged!")
            break

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Caustics
    ax1.plot(rc[:,0], rc[:,1], 'r-', lw=2, label='final caustic')
    ax1.plot(target[:,0], target[:,1], 'k--', lw=1.5, label='target')
    ax1.set_aspect('equal'); ax1.grid(True)
    ax1.legend(); ax1.set_title('Caustic matching result')
    
    # Convergence
    ax2.semilogy(Js, 'b.-', markersize=8)
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Cost J')
    ax2.set_title('Optimization convergence'); ax2.grid(True)
    
    plt.tight_layout(); plt.show()
    
else:
    print("❌ Still unstable - need even more conservative settings")