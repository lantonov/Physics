# Requires: sympy
import sympy as sp
sp.init_printing(use_unicode=True)

# Symbols
e,a,GM,c,R,theta,u,eta = sp.symbols('e a GM c R theta u eta')
pi = sp.pi

# Hopf expansion coefficients
a1 = -sp.Rational(1,2)
a2 = sp.Rational(3,8)
a3 = -sp.Rational(5,16)

# Angular momentum per unit mass
ell2 = GM*a*(1 - e**2)

# Correct pulled-back potential per unit mass (include R factors)
U = GM*( R*u + a1*(1/(R*u)) + a2*(1/(R**3 * u**3)) + a3*(1/(R**5 * u**5)) )

# Exact Binet forcing
dU_du = sp.diff(U, u)
Delta_f = - (1/(ell2 * u**2)) * dU_du
Delta_f = sp.simplify(Delta_f)

# Extract coefficients of u**(-4), u**(-6), u**(-8)
c4 = sp.simplify( sp.together(Delta_f).coeff(u, -4) )
c6 = sp.simplify( sp.together(Delta_f).coeff(u, -6) )
c8 = sp.simplify( sp.together(Delta_f).coeff(u, -8) )

# Unperturbed orbit u0(theta)
u0 = (1/(a*(1 - e**2))) * (1 + e*sp.cos(theta))

# Closed forms for K2,K3 and J2,J3 (exact)
K2 = 2*pi/(1-e**2)**(sp.Rational(3,2))
K3 = pi*(2+e**2)/(1-e**2)**(sp.Rational(5,2))
J2 = -2*pi*e/(1-e**2)**(sp.Rational(3,2))
J3 = -3*pi*e/(1-e**2)**(sp.Rational(5,2))

# Derive K4,K5 and J4,J5 by differentiation identities:
# d/de K_n = -n * J_{n+1}
# so J4 = -1/3 dK3/de, J5 = -1/4 dK4/de etc. We'll compute K4,K5 by integrating relations or by differentiating known K2/K3.
K3_expr = sp.simplify(K3)
# compute J4 from dK3/de = -3 J4  => J4 = -1/3 dK3/de
J4 = sp.simplify(-sp.diff(K3_expr, e)/3)
# compute K4 by using relation between K4 and J4: d/de K4 = -4 J5, but we can obtain K4 by direct integral formula:
# Use symbolic integration for K4 and K5 via residue substitution (z = exp(i theta)) to avoid guesswork.
z = sp.symbols('z')
# helper to compute K_n via contour substitution: K_n = ∮ 2^n z^{n-1} / (e z^2 + 2 z + e)^n dz / i
def K_via_residue(n):
    # build integrand as rational function in z
    z = sp.symbols('z')
    D = e*z**2 + 2*z + e
    integrand = 2**n * z**(n-1) / D**n
    # residue at the root inside unit circle z_plus
    z_plus = (-1 + sp.sqrt(1-e**2))/e
    # compute residue of integrand at z_plus
    res = sp.residue(integrand, z, z_plus)
    return sp.simplify(2*sp.pi*res)
K4 = sp.simplify(K_via_residue(4))
K5 = sp.simplify(K_via_residue(5))
# compute J4,J5 via differentiation identities: J_n = -1/(n-1) dK_{n-1}/de
J4_check = sp.simplify(-sp.diff(K3_expr, e)/3)  # same as above
J5 = sp.simplify(-sp.diff(K4, e)/4)

# Now compute the secular integrals S_n = ∫ sinθ (1+e cosθ)^{-n} dθ for n=4,6,8
# We'll compute them symbolically by direct integration (SymPy can handle these definite integrals)
S4 = sp.integrate(sp.sin(theta)*(1+e*sp.cos(theta))**(-4), (theta, 0, 2*pi))
S6 = sp.integrate(sp.sin(theta)*(1+e*sp.cos(theta))**(-6), (theta, 0, 2*pi))
S8 = sp.integrate(sp.sin(theta)*(1+e*sp.cos(theta))**(-8), (theta, 0, 2*pi))

# secular contributions I_n = -(1/e) * c_n * (a*(1-e^2))^n * S_n
def I_n_from_cn(cn, n, S_n):
    factor = (a*(1-e**2))**n
    return sp.simplify( (-1/e) * cn * factor * S_n )

I4 = sp.simplify(I_n_from_cn(c4, 4, S4))
I6 = sp.simplify(I_n_from_cn(c6, 6, S6))
I8 = sp.simplify(I_n_from_cn(c8, 8, S8))

# First-order
Delta_phi_1 = sp.simplify(I4)
Delta_phi_C = sp.simplify(I6)
Delta_phi_A = sp.simplify(I8)

# Second-order iteration: compute Fourier coefficients A_k for RHS = c4 * u0^{-4}
RHS = sp.simplify(c4 * u0**(-4))
A0 = (1/pi) * sp.integrate(RHS, (theta,0,2*pi))
A1 = (1/pi) * sp.integrate(RHS*sp.cos(theta), (theta,0,2*pi))
A2 = (1/pi) * sp.integrate(RHS*sp.cos(2*theta), (theta,0,2*pi))
A3 = (1/pi) * sp.integrate(RHS*sp.cos(3*theta), (theta,0,2*pi))

# particular solution excluding k=1 resonance
delta_u = sp.simplify(A0 + A2*sp.cos(2*theta)/(1-4) + A3*sp.cos(3*theta)/(1-9))

# d/du Δf1 evaluated on u0
dDf1_du = sp.simplify(sp.diff(c4*u**(-4), u).subs(u, u0))
Delta_phi_B = sp.simplify( (-1/e) * sp.integrate(delta_u * dDf1_du * sp.sin(theta), (theta, 0, 2*pi)) )

# Total second-order contribution
Delta_phi_total = sp.simplify(Delta_phi_1 + Delta_phi_A + Delta_phi_B + Delta_phi_C)

# Expand in eta = GM/(a*c**2)
expr_eta = sp.simplify( Delta_phi_total.subs(GM, eta*a*c**2) )
series_eta = sp.series(expr_eta, eta, 0, 3).removeO()
C2_expr = sp.simplify( (series_eta - 6*pi*eta) / (pi*eta**2) )
C2_simpl = sp.simplify(sp.together(sp.factor(C2_expr)))

# Expand numerator and denominator to get explicit polynomials
P = sp.simplify(sp.factor(sp.together(sp.simplify(sp.factor(sp.expand(sp.together(sp.simplify(sp.factor(sp.simplify(sp.together(sp.simplify(sp.factor(sp.simplify(sp.together(C2_simpl)))))))))))))))
# The above is defensive; the final rational function is C2_simpl; extract numerator/denominator
P_poly = sp.simplify(sp.factor(sp.together(sp.simplify(sp.factor(sp.simplify(sp.together(sp.simplify(sp.factor(sp.simplify(sp.together(C2_simpl))))))))))).as_numer_denom()[0]
Q_poly = sp.simplify(sp.factor(sp.together(sp.simplify(sp.factor(sp.simplify(sp.together(sp.simplify(sp.factor(sp.simplify(sp.together(C2_simpl))))))))))).as_numer_denom()[1]

# Numeric evaluation
C2_numeric = sp.N(C2_simpl.subs(e, 0.2056), 12)

# Print results
print('c4 =', sp.simplify(c4))
print('c6 =', sp.simplify(c6))
print('c8 =', sp.simplify(c8))
print('\\nDelta_phi_1 =', sp.simplify(Delta_phi_1))
print('\\nDelta_phi_C =', sp.simplify(Delta_phi_C))
print('\\nDelta_phi_A =', sp.simplify(Delta_phi_A))
print('\\nDelta_phi_B =', sp.simplify(Delta_phi_B))
print('\\nC2(e) (rational) =', C2_simpl)
print('\\nP(e) =', sp.expand(P_poly))
print('\\nQ(e) =', sp.expand(Q_poly))
print('\\nC2(0.2056) =', C2_numeric)
