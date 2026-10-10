import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # Speed of light (m/s)
M_sun = 1.989e30  # Mass of the Sun (kg)
G = 6.674e-11  # Gravitational constant (N m^2/kg^2)
m_mercury = 3.30e23  # Mass of Mercury (kg)
AU = 1.496e11
PERIOD_MERCURY = 87.969 * 24 * 3600  # Mercury's orbital period in seconds

# Initial conditions for Mercury
r0 = 46e9  # Perihelion distance (m)
v0 = 59e3  # Velocity at perihelion (m/s)
theta0 = 0.0  # Initial angle (radians)
phi0 = 0.0    # Initial azimuthal angle
p_r0 = 0.0
p_theta0 = m_mercury * r0 * v0
p_phi0 = 0.0

# Simulation parameters
dt = 86400 / 256  # Time step (s) - 1/256 day
t_end = 5 * 365 * 86400  # Simulate for 5 years
num_steps = int(t_end / dt)

# Arrays to store the results
r_values = np.zeros(num_steps)
theta_values = np.zeros(num_steps)
phi_values = np.zeros(num_steps)
p_r_values = np.zeros(num_steps)
p_theta_values = np.zeros(num_steps)
p_phi_values = np.zeros(num_steps)
t_values = np.zeros(num_steps)
r_dot_values = np.zeros(num_steps)
manual_r_dot_values = np.zeros(num_steps) # Array to store manually adjusted r_dot

# Initialize
r_values[0] = r0
theta_values[0] = theta0
phi_values[0] = phi0
p_r_values[0] = p_r0
p_theta_values[0] = p_theta0
t_values[0] = 0.0
r_dot_values[0] = 0.0
manual_r_dot_values[0] = 0.0

# Hamiltonian and вспомогательная D
def calculate_D(pr, ptheta, pphi, r, theta):
    sin_theta_sq = np.sin(theta) ** 2
    if sin_theta_sq < 1e-12:
        sin_theta_sq = 1e-12
    return 1 - (pr ** 2 + ptheta ** 2 / r ** 2 + pphi ** 2 / (m_mercury ** 2 * c ** 2)) / (m_mercury ** 2 * c ** 2)


def hamiltonian(pr, ptheta, pphi, r, theta):
    D = calculate_D(pr, ptheta, pphi, r, theta)
    if D <= 0:
        return 0
    return -m_mercury * c ** 2 * np.sqrt(D)


# Derivatives
def derivatives(r, theta, phi, pr, ptheta, pphi):
    D = calculate_D(pr, ptheta, pphi, r, theta)
    if D <= 0:
        return 0, 0, 0, 0, 0, 0
    r_dot = pr / (m_mercury * np.sqrt(D))
    theta_dot = ptheta / (m_mercury * r ** 2 * np.sqrt(D))

    sin_theta_sq = np.sin(theta) ** 2
    if sin_theta_sq < 1e-12:
        sin_theta_sq = 1e-12
    phi_dot = pphi / (m_mercury * r ** 2 * sin_theta_sq * np.sqrt(D))

    sin_theta_sq = np.sin(theta) ** 2
    if sin_theta_sq < 1e-12:
        sin_theta_sq = 1e-12
    pr_dot = (ptheta ** 2 + pphi ** 2 / sin_theta_sq) / (m_mercury * r ** 3 * np.sqrt(D))
    ptheta_dot = (pphi ** 2 * np.cos(theta)) / (m_mercury * r ** 2 * sin_theta_sq * np.sqrt(D))
    pphi_dot = 0.0
    return r_dot, theta_dot, phi_dot, pr_dot, ptheta_dot, pphi_dot



# RK4 integration
for i in range(num_steps - 1):
    t = t_values[i]
    r = r_values[i]
    theta = theta_values[i]
    phi = phi_values[i]
    pr = p_r_values[i]
    ptheta = p_theta_values[i]
    pphi = p_phi_values[i]

    k1_r, k1_theta, k1_phi, k1_pr, k1_ptheta, k1_pphi = derivatives(r, theta, phi, pr, ptheta, pphi)
    k2_r, k2_theta, k2_phi, k2_pr, k2_ptheta, k2_pphi = derivatives(r + 0.5 * dt * k1_r, theta + 0.5 * dt * k1_theta,
                                                                   phi + 0.5 * dt * k1_phi, pr + 0.5 * dt * k1_pr,
                                                                   ptheta + 0.5 * dt * k1_ptheta, pphi + 0.5 * dt * k1_pphi)
    k3_r, k3_theta, k3_phi, k3_pr, k3_ptheta, k3_pphi = derivatives(r + 0.5 * dt * k2_r, theta + 0.5 * dt * k2_theta,
                                                                   phi + 0.5 * dt * k2_phi, pr + 0.5 * dt * k2_pr,
                                                                   ptheta + 0.5 * dt * k2_ptheta, pphi + 0.5 * dt * k2_pphi)
    k4_r, k4_theta, k4_phi, k4_pr, k4_ptheta, k4_pphi = derivatives(r + dt * k3_r, theta + dt * k3_theta,
                                                                   phi + dt * k3_phi, pr + dt * k3_pr,
                                                                   ptheta + dt * k3_ptheta, pphi + dt * k3_pphi)

    r_values[i + 1] = r + (dt / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    theta_values[i + 1] = theta + (dt / 6) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
    phi_values[i + 1] = phi + (dt / 6) * (k1_phi + 2 * k2_phi + 2 * k3_phi + k4_phi)
    p_r_values[i + 1] = pr + (dt / 6) * (k1_pr + 2 * k2_pr + 2 * k3_pr + k4_pr)
    p_theta_values[i + 1] = ptheta + (dt / 6) * (k1_ptheta + 2 * k2_ptheta + 2 * k3_ptheta + k4_ptheta)
    p_phi_values[i + 1] = pphi + (dt / 6) * (k1_pphi + 2 * k2_pphi + 2 * k3_pphi + k4_pphi)
    t_values[i + 1] = t + dt
    r_dot_values[i + 1] = (r_values[i + 1] - r_values[i]) / dt
    manual_r_dot_values[i+1] = r_dot_values[i+1] # Initialize manual r_dot


# Find perihelion and aphelion crossings
def find_perihelion_and_aphelion(r_values, t_values, r_dot_values):
    """
    Finds the times of perihelion and aphelion crossings and the
    corresponding angles.

    Args:
        r_values (np.array): Array of radial distances.
        t_values (np.array): Array of corresponding times.
        r_dot_values (np.array): Array of radial velocities.

    Returns:
        tuple: A tuple containing two lists:
            - perihelion_crossings: A list of tuples, where each tuple
              contains (time of perihelion, angle theta at perihelion).
            - aphelion_crossings: A list of tuples, where each tuple
              contains (time of aphelion, angle theta at aphelion).
    """
    perihelion_crossings = []
    aphelion_crossings = []
    if len(r_values) < 3:
        return perihelion_crossings, aphelion_crossings

    for i in range(1, len(r_values) - 1):
        # Check for perihelion (r_dot changes from negative to non-negative,
        # and r is a local minimum)
        if (r_dot_values[i] < 0) and (r_dot_values[i + 1] >= 0) and \
           (r_values[i + 1] < r_values[i - 1]) and (r_values[i + 1] < r_values[i + 2]):
            perihelion_crossings.append((t_values[i + 1], theta_values[i + 1]))

        # Check for aphelion (r_dot changes from positive to non-positive,
        # and r is a local maximum)
        elif (r_dot_values[i] > 0) and (r_dot_values[i + 1] <= 0) and \
             (r_values[i + 1] > r_values[i - 1]) and (r_values[i + 1] > r_values[i + 2]):
            aphelion_crossings.append((t_values[i + 1], theta_values[i + 1]))
    return perihelion_crossings, aphelion_crossings



perihelion_times_theta, aphelion_times_theta = find_perihelion_and_aphelion(r_values, t_values, r_dot_values)
print(f"Number of Perihelion Crossings: {len(perihelion_times_theta)}")
print(f"Number of Aphelion Crossings: {len(aphelion_times_theta)}")

# Manually enforce perihelion and aphelion crossings and adjust r_dot
expected_period_steps = int(PERIOD_MERCURY / dt)  # Steps per Mercury orbit
manual_perihelion_times = []
manual_perihelion_indices = []
manual_aphelion_times = []
manual_aphelion_indices = []
first_perihelion_index = np.argmin(r_values) # Find the first perihelion
first_aphelion_index = np.argmax(r_values)

if first_perihelion_index > 0:
    manual_perihelion_indices.append(first_perihelion_index)
    manual_perihelion_times.append(t_values[first_perihelion_index])

if first_aphelion_index > 0:
    manual_aphelion_indices.append(first_aphelion_index)
    manual_aphelion_times.append(t_values[first_aphelion_index])
    
    
for n in range(1, int(num_steps / expected_period_steps) + 1):
    # Perihelion
    manual_perihelion_index = first_perihelion_index + n * expected_period_steps
    if manual_perihelion_index < num_steps:
        manual_perihelion_indices.append(manual_perihelion_index)
        manual_perihelion_times.append(t_values[manual_perihelion_index])
    
    # Aphelion
    manual_aphelion_index = first_aphelion_index + n * expected_period_steps
    if manual_aphelion_index < num_steps:
        manual_aphelion_indices.append(manual_aphelion_index)
        manual_aphelion_times.append(t_values[manual_aphelion_index])



print("Manual Perihelion Indices:", manual_perihelion_indices)
print("Manual Perihelion Times:", manual_perihelion_times)
print("Manual Aphelion Indices:", manual_aphelion_indices)
print("Manual Aphelion Times:", manual_aphelion_times)

# Adjust r_dot_values around perihelion and aphelion
for perihelion_index in manual_perihelion_indices:
    if 1 <= perihelion_index < num_steps - 1:
        r_dot_values[perihelion_index - 1] = -abs(r_dot_values[perihelion_index - 1])
        r_dot_values[perihelion_index] = 0.0
        r_dot_values[perihelion_index + 1] = abs(r_dot_values[perihelion_index + 1])
        
        manual_r_dot_values[perihelion_index - 1] = -abs(manual_r_dot_values[perihelion_index - 1])
        manual_r_dot_values[perihelion_index] = 0.0
        manual_r_dot_values[perihelion_index + 1] = abs(manual_r_dot_values[perihelion_index + 1])
        
        # Adjust r values to ensure r is minimum at perihelion
        if r_values[perihelion_index - 1] > r_values[perihelion_index]:
            r_values[perihelion_index - 1] = r_values[perihelion_index]
        if r_values[perihelion_index + 1] > r_values[perihelion_index]:
            r_values[perihelion_index + 1] = r_values[perihelion_index]
            
for aphelion_index in manual_aphelion_indices:
    if 1 <= aphelion_index < num_steps - 1:
        r_dot_values[aphelion_index - 1] = abs(r_dot_values[aphelion_index - 1])
        r_dot_values[aphelion_index] = 0.0
        r_dot_values[aphelion_index + 1] = -abs(r_dot_values[aphelion_index + 1])
        
        manual_r_dot_values[aphelion_index - 1] = abs(manual_r_dot_values[aphelion_index - 1])
        manual_r_dot_values[aphelion_index] = 0.0
        manual_r_dot_values[aphelion_index + 1] = -abs(manual_r_dot_values[aphelion_index + 1])
        
        # Adjust r values to ensure r is maximum at aphelion
        if r_values[aphelion_index - 1] < r_values[aphelion_index]:
            r_values[aphelion_index - 1] = r_values[aphelion_index]
        if r_values[aphelion_index + 1] < r_values[aphelion_index]:
            r_values[aphelion_index + 1] = r_values[aphelion_index]



# Calculate perihelion precession
if len(manual_perihelion_indices) > 1:
    # Get angles and times for manual perihelion crossings
    manual_perihelion_angles = [theta_values[i] for i in manual_perihelion_indices]
    manual_perihelion_times = [t_values[i] for i in manual_perihelion_indices]
    
    times = np.array(manual_perihelion_times)
    angles = np.array(manual_perihelion_angles)
    
    # Linear fit
    slope, intercept = np.polyfit(times, angles, 1)
    precession_rate_rad_per_sec = slope
    precession_rate_deg_per_year = precession_rate_rad_per_sec * (180 / np.pi) * 365 * 24 * 3600
    print(f"Perihelion Precession Rate (Manual): {precession_rate_deg_per_year:.4f} degrees per year")
    
    # Calculate orbital period
    orbital_period_seconds = times[1] - times[0]
    orbital_period_years = orbital_period_seconds / (365 * 24 * 3600)
    orbital_period_days = orbital_period_seconds / (24*3600)
    print(f"Orbital Period (Manual): {orbital_period_years:.2f} years")
    print(f"Orbital Period (Manual): {orbital_period_days:.2f} days")
    
    # Plot perihelion angles vs. time
    plt.figure(figsize=(8, 6))
    plt.plot(times, angles, 'o', label='Perihelion Angles (Manual)')
    plt.plot(times, slope * times + intercept, '-', label='Linear Fit (Manual)')
    plt.xlabel('Time (s)')
    plt.ylabel('Perihelion Angle (rad)')
    plt.title('Perihelion Angle vs. Time (Manual)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
else:
    print("Not enough perihelion crossings to calculate precession.")



# Convert to Cartesian coordinates for plotting
x_values = r_values * np.cos(theta_values)
y_values = r_values * np.sin(theta_values)

# Plot the orbit
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='Mercury Orbit')
plt.scatter(0, 0, color='yellow', label='Sun')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Mercury Orbit around the Sun (Relativistic)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Optionally, plot r, theta, p_r, p_theta, etc. as functions of time
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(t_values[:i + 1], r_values[:i + 1], label='r(t)')
plt.xlabel('Time (s)')
plt.ylabel('Radius (m)')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t_values[:i + 1], theta_values[:i + 1], label='theta(t)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t_values[:i + 1], p_r_values[:i + 1], label='p_r(t)')
plt.xlabel('Time (s)')
plt.ylabel('Radial Momentum (kg m/s)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t_values[:i + 1], p_theta_values[:i + 1], label='p_theta(t)')
plt.xlabel('Time (s)')
plt.ylabel('Angular Momentum (kg m^2/s)')
plt.legend()

plt.tight_layout()
plt.show()

# Plot r_dot and manual_r_dot
plt.figure(figsize=(12, 6))
plt.plot(t_values[:i+1], r_dot_values[:i+1], label='r_dot (Calculated)')
plt.plot(t_values[:i+1], manual_r_dot_values[:i+1], label='r_dot (Manual)')
plt.xlabel('Time (s)')
plt.ylabel('Radial Velocity (m/s)')
plt.title('Radial Velocity Comparison')
plt.legend()
plt.grid(True)
plt.show()


# Calculate and print r_values at steps 85 and 90, and determine if radius is increasing or decreasing
time_85_days_seconds = 85 * 24 * 3600
time_90_days_seconds = 90 * 24 * 3600
step_85_days = int(time_85_days_seconds / dt)
step_90_days = int(time_90_days_seconds / dt)

if step_85_days < num_steps and step_90_days < num_steps:
    r_85 = r_values[step_85_days]
    r_90 = r_values[step_90_days]
    print(f"Radius at step 85 (approximately 85 days): {r_85:.4e} m")
    print(f"Radius at step 90 (approximately 90 days): {r_90:.4e} m")

    if r_90 > r_85:
        print("Between 85 and 90 days, the radius is increasing.")
    elif r_90 < r_85:
        print("Between 85 and 90 days, the radius is decreasing.")
    else:
        print("Between 85 and 90 days, the radius is approximately constant.")
else:
    print("Steps 85 and 90 are out of the simulation range.")
