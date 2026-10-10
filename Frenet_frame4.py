import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example curve on S^3.
def gamma(s):
    # The original curve: here q = [cos(s), sin(s)/sqrt(2), sin(s)/sqrt(2), 0]
    # Note: q0 = cos(s) will oscillate, so expect some points to be handled
    # by the north chart (q0>0) and some by the south (q0<0).
    return np.array([np.cos(s), np.sin(s)/np.sqrt(2), np.sin(s)/np.sqrt(2), 0])

# Multi-chart gnomonic projection.
def multi_chart_gnomonic_proj(q, epsilon=1e-8):
    if q[0] >= epsilon:
        proj = q[1:] / q[0]
        chart = 'north'
    elif q[0] <= -epsilon:
        proj = q[1:] / -q[0]
        chart = 'south'
    else:
        proj = np.array([np.nan, np.nan, np.nan])
        chart = 'undefined'
    return proj, chart

# Sample parameter values.
s_vals = np.linspace(0, 4*np.pi, 500)
proj_points = []
charts_used = []

for s in s_vals:
    q = gamma(s)
    proj, chart = multi_chart_gnomonic_proj(q)
    proj_points.append(proj)
    charts_used.append(chart)
proj_points = np.array(proj_points)

# Plot projected points, color-coding the charts.
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Separate indices for north and south chart points.
north_inds = [i for i, ch in enumerate(charts_used) if ch == 'north']
south_inds = [i for i, ch in enumerate(charts_used) if ch == 'south']

ax.plot(proj_points[north_inds,0], proj_points[north_inds,1], proj_points[north_inds,2],
        'b-', label='Gnomonic (North chart)')
ax.plot(proj_points[south_inds,0], proj_points[south_inds,1], proj_points[south_inds,2],
        'r-', label='Gnomonic (South chart)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Multi-Chart Gnomonic Projection of a Curve on S^3')
ax.legend()
plt.show()
