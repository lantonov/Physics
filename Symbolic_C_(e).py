# Requires: sympy
import sympy as sp

# Symbols
e,a,GM,c,R,theta,u,eta = sp.symbols('e a GM c R theta u eta')
pi = sp.pi

# Hopf expansion coefficients
a1 = -sp.Rational(1,2)
a2 = sp.Rational(3,8)
a3 = -sp.Rational(5,16)

# Angular momentum per unit mass
ell2 = GM*a*(1 - e**2)

# --- corrected potential with explicit R factors ---
# U(1/u)/m = GM*( R*u + a1*(1/(R*u)) + a2*(1/(R**3 * u**3)) + a3*(1/(R**5 * u**5)) )
U = GM*( R*u + a1*(1/(R*u)) + a2*(1/(R**3 * u**3)) + a3*(1/(R**5 * u**5)) )

# Binet forcing (exact)
dU_du = sp.diff(U, u)
Delta_f = - (1/(ell2 * u**2)) * dU_du
Delta_f = sp.simplify(Delta_f)

# Extract coefficients of u**(-4), u**(-6), u**(-8)
c4 = sp.simplify( sp.together(Delta_f).coeff(u, -4) )
c6 = sp.simplify( sp.together(Delta_f).coeff(u, -6) )
c8 = sp.simplify( sp.together(Delta_f).coeff(u, -8) )

# Unperturbed orbit u0(theta)
u0 = (1/(a*(1 - e**2))) * (1 + e*sp.cos(theta))

# Helper: compute integral I_n = -(1/e) * c_n * ∫ sinθ * u0^{-n} dθ
def secular_integral_coeff(cn, n):
    factor = (a*(1-e**2))**n
    integrand = sp.sin(theta) * (1+e*sp.cos(theta))**(-n)
    S = sp.integrate(integrand, (theta, 0, 2*pi))
    return sp.simplify( (-1/e) * cn * factor * S )

I4 = secular_integral_coeff(c4, 4)
I6 = secular_integral_coeff(c6, 6)
I8 = secular_integral_coeff(c8, 8)

# First-order
Delta_phi_1 = sp.simplify(I4)
# Direct second-order from c8
Delta_phi_A = sp.simplify(I8)
# Single iteration from c6
Delta_phi_C = sp.simplify(I6)

# Second-order iteration: solve (d^2+1) δu = RHS where RHS = c4 * u0^{-4}
RHS = sp.simplify(c4 * (u0**(-4)))
# Fourier coefficients A_k = (1/pi) ∫ RHS cos(kθ) dθ for k=0..3
A0 = (1/pi) * sp.integrate(RHS, (theta,0,2*pi))
A1 = (1/pi) * sp.integrate(RHS*sp.cos(theta), (theta,0,2*pi))
A2 = (1/pi) * sp.integrate(RHS*sp.cos(2*theta), (theta,0,2*pi))
A3 = (1/pi) * sp.integrate(RHS*sp.cos(3*theta), (theta,0,2*pi))
# Particular solution δu = sum_{k!=1} A_k cos(kθ)/(1-k^2). For k=1 resonance, omit secular term.
delta_u = sp.simplify(A0 + A2*sp.cos(2*theta)/(1-4) + A3*sp.cos(3*theta)/(1-9))

# d/du Δf1 evaluated on u0
dDf1_du = sp.simplify(sp.diff(c4*u**(-4), u).subs(u, u0))
Delta_phi_B = sp.simplify( (-1/e) * sp.integrate(delta_u * dDf1_du * sp.sin(theta), (theta, 0, 2*pi)) )

Delta_phi_total = sp.simplify(Delta_phi_1 + Delta_phi_A + Delta_phi_B + Delta_phi_C)

# Expand in eta = GM/(a*c**2)
expr_eta = sp.simplify( Delta_phi_total.subs(GM, eta*a*c**2) )
series_eta = sp.series(expr_eta, eta, 0, 3).removeO()
C2_expr = sp.simplify( (series_eta - 6*pi*eta) / (pi*eta**2) )
C2_simpl = sp.simplify(sp.together(C2_expr))

# Numeric evaluation
C2_numeric = sp.N(C2_simpl.subs(e, 0.2056), 12)

print('c4 =', c4)
print('c6 =', c6)
print('c8 =', c8)
print('Delta_phi_1 =', sp.simplify(Delta_phi_1))
print('Delta_phi_A =', sp.simplify(Delta_phi_A))
print('Delta_phi_B =', sp.simplify(Delta_phi_B))
print('Delta_phi_C =', sp.simplify(Delta_phi_C))
print('C2(e) =', C2_simpl)
print('C2(0.2056) =', C2_numeric)
