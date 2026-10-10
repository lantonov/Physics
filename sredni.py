a = 5
b = 2

gm  = (a*b)**0.5
am  = (a + b)/2
hm  = 2*a*b/(a+b)
chm = (a**2 + b**2)/(a + b)

from numpy import pi
from scipy.special import ellipk

agm = 0.25*pi*(a + b)/ellipk((a - b)**2/(a + b)**2)
print(gm,am,hm,chm,agm)