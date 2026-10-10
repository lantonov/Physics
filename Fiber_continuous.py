from sympy import *
from sympy.ntheory.continued_fraction import continued_fraction_iterator, continued_fraction_convergents

# Define the irrational frequency ratio numerically (from previous message)
omega_ratio = 18031.22

# Compute the continued fraction expansion up to a reasonable depth
# Since continued_fraction_iterator doesn't have max_terms parameter,
# we need to manually limit the iteration
cf_iter = []
cf_generator = continued_fraction_iterator(omega_ratio)
max_terms = 10

for i, term in enumerate(cf_generator):
    if i >= max_terms:
        break
    cf_iter.append(term)

# Compute the convergents using the correct approach
convergents = list(continued_fraction_convergents(cf_iter))

print("Continued fraction terms:", cf_iter)
print("Convergents:", convergents)