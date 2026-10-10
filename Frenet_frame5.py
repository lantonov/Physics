import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the curve on S^3.
def gamma(s):
    return np.array([np.cos(s), np.sin(s)/np.sqrt(2), np.sin(s)/np.sqrt(2), 0])

# Hard multi-chart stereographic projection
def multi_chart_stereographic(q, epsilon=1e-8):
    denom_N = 1 - q[0]
    denom_S = 1 + q[0]
    if np.abs(denom_N) < epsilon:
         return q[1:] / denom_S, 'south'
    elif np.abs(denom_S) < epsilon:
         return q[1:] / denom_N, 'north'
    if np.abs(denom_N) >= np.abs(denom_S):
         return q[1:] / denom_N, 'north'
    else:
         return q[1:] / denom_S, 'south'

# Smooth atlas function for stereographic projection.
def atlas_stereographic(q, north_thresh=0.8, south_thresh=-0.8):
    q0 = q[0]
    # Compute each projection when possible.
    if np.abs(1 - q0) > 1e-8:
        xN = q[1:] / (1 - q0)
    else:
        xN = np.array([np.nan, np.nan, np.nan])
    if np.abs(1 + q0) > 1e-8:
        xS = q[1:] / (1 + q0)
    else:
        xS = np.array([np.nan, np.nan, np.nan])
        
    if q0 >= north_thresh:
        return xN, 'north'
    elif q0 <= south_thresh:
        return xS, 'south'
    else:
        weight = (q0 - south_thresh) / (north_thresh - south_thresh)
        blended = weight * xN + (1 - weight) * xS
        return blended, 'blended'

# Sample the curve.
s_vals = np.linspace(0, 4*np.pi, 500)
proj_hard = []
charts_hard = []
proj_atlas = []
charts_atlas = []
for s in s_vals:
    q = gamma(s)
    p_hard, chard = multi_chart_stereographic(q)
    p_atlas, chara = atlas_stereographic(q)
    proj_hard.append(p_hard)
    charts_hard.append(chard)
    proj_atlas.append(p_atlas)
    charts_atlas.append(chara)
proj_hard = np.array(proj_hard)
proj_atlas = np.array(proj_atlas)

# Plot the two projections side-by-side.
fig = plt.figure(figsize=(14,6))

# Hard multi-chart projection.
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(proj_hard[:,0], proj_hard[:,1], proj_hard[:,2], 'b-', label='Hard multi-chart')
ax1.set_title('Hard Chart Switching')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.legend()

# Smooth atlas projection.
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(proj_atlas[:,0], proj_atlas[:,1], proj_atlas[:,2], 'r-', label='Atlas (blended)')
ax2.set_title('Smooth Atlas Transition')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.legend()

plt.show()
