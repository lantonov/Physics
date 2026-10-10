import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
import numpy as np

# Updated simulation with potential toggle and energy overlay
def compute_potential_energy(r, model, v0=1):
    if model == "log":
        return 0.5 * v0**2 * np.log(r**2)
    elif model == "newton":
        return -1 / r
    return np.zeros_like(r)

def compute_kinetic_energy(r, dr_dt):
    return 0.5 * m * dr_dt**2

def simulate_orbit(v0, delta_r, gamma, t, model="log", return_energy=False):
    r = 1 + delta_r * np.exp(-gamma * t) * np.cos(v0 * t)
    theta = v0 * t
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if return_energy:
        dr_dt = -delta_r * np.exp(-gamma * t) * (v0 * np.sin(v0 * t) + gamma * np.cos(v0 * t))
        T = compute_kinetic_energy(r, dr_dt)
        U = compute_potential_energy(r, model, v0)
        E = T + U
        return x, y, r, E, T, U
    return x, y, r

# Parameters
t = np.linspace(0, 2*np.pi, 800)
m = 1.0
init_v0 = 200
init_delta_r = 0.2
init_gamma = 0.05
init_model = "log"

# Plot setup
fig, (ax_orbit, ax_energy) = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(bottom=0.35)
x, y, r, E, T, U = simulate_orbit(init_v0, init_delta_r, init_gamma, t, init_model, True)
orbit_line, = ax_orbit.plot(x, y)
ax_orbit.set_title("Galaxy Orbit")
ax_orbit.axis("equal")
ax_orbit.set_xlim(-2, 2)
ax_orbit.set_ylim(-2, 2)
ax_orbit.set_xlabel("x")
ax_orbit.set_ylabel("y")

energy_line, = ax_energy.plot(t, E, label="Total E")
T_line, = ax_energy.plot(t, T, label="Kinetic")
U_line, = ax_energy.plot(t, U, label="Potential")
ax_energy.set_ylabel("Energy")
ax_energy.set_xlabel("Time")
ax_energy.legend()
ax_energy.grid(True)

# Sliders and toggles
ax_v0 = plt.axes([0.2, 0.25, 0.65, 0.03])
ax_delta_r = plt.axes([0.2, 0.2, 0.65, 0.03])
ax_gamma = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_pot = plt.axes([0.025, 0.65, 0.15, 0.1])
ax_check = plt.axes([0.025, 0.5, 0.15, 0.1])

s_v0 = Slider(ax_v0, 'v0', 100, 300, valinit=init_v0)
s_delta_r = Slider(ax_delta_r, 'δr', 0.01, 0.5, valinit=init_delta_r)
s_gamma = Slider(ax_gamma, 'γ', 0.0, 0.2, valinit=init_gamma)
radio_pot = RadioButtons(ax_pot, ('log', 'newton'), active=0)
check_energy = CheckButtons(ax_check, ['Show Energy'], [True])

# Update function
def update(val):
    v0 = s_v0.val
    delta_r = s_delta_r.val
    gamma = s_gamma.val
    model = radio_pot.value_selected
    show_E = check_energy.get_status()[0]
    x, y, r, E, T, U = simulate_orbit(v0, delta_r, gamma, t, model, True)
    orbit_line.set_xdata(x)
    orbit_line.set_ydata(y)
    if show_E:
        energy_line.set_ydata(E)
        T_line.set_ydata(T)
        U_line.set_ydata(U)
        ax_energy.set_ylim(np.min([E, T, U]) - 0.2, np.max([E, T, U]) + 0.2)
    else:
        energy_line.set_ydata(np.full_like(t, np.nan))
        T_line.set_ydata(np.full_like(t, np.nan))
        U_line.set_ydata(np.full_like(t, np.nan))
    fig.canvas.draw_idle()

# Connect widgets
s_v0.on_changed(update)
s_delta_r.on_changed(update)
s_gamma.on_changed(update)
radio_pot.on_clicked(update)
check_energy.on_clicked(update)

plt.show()
