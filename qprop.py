import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################################################################
#                             DRAG & ATMOSPHERE                              #
##############################################################################

def calculate_drag_perturbation(state_vector):
    """
    Return an effective drag coefficient (constant, typical Cd ~ 2.2).
    """
    return 2.2

def calculate_atmospheric_density(altitude):
    """
    Calculate atmospheric density based on a piecewise exponential model
    with variable scale heights.
    """
    h = altitude / 1000.0
    if h < 0:  # below Earth's surface
        return 0.0

    if h < 25:
        rho0, H = 1.225, 7.249
        rho = rho0 * np.exp(-h / H)
    elif h < 30:
        rho0, H = 3.899e-2, 6.349
        rho = rho0 * np.exp(-(h - 25) / H)
    elif h < 40:
        rho0, H = 1.774e-2, 6.682
        rho = rho0 * np.exp(-(h - 30) / H)
    elif h < 50:
        rho0, H = 3.972e-3, 7.554
        rho = rho0 * np.exp(-(h - 40) / H)
    elif h < 60:
        rho0, H = 1.057e-3, 8.382
        rho = rho0 * np.exp(-(h - 50) / H)
    elif h < 70:
        rho0, H = 3.206e-4, 7.714
        rho = rho0 * np.exp(-(h - 60) / H)
    elif h < 80:
        rho0, H = 8.770e-5, 6.549
        rho = rho0 * np.exp(-(h - 70) / H)
    elif h < 90:
        rho0, H = 1.905e-5, 5.799
        rho = rho0 * np.exp(-(h - 80) / H)
    elif h < 100:
        rho0, H = 3.396e-6, 5.382
        rho = rho0 * np.exp(-(h - 90) / H)
    elif h < 110:
        rho0, H = 5.297e-7, 5.877
        rho = rho0 * np.exp(-(h - 100) / H)
    elif h < 120:
        rho0, H = 9.661e-8, 6.396
        rho = rho0 * np.exp(-(h - 110) / H)
    elif h < 130:
        rho0, H = 2.438e-8, 7.054
        rho = rho0 * np.exp(-(h - 120) / H)
    elif h < 140:
        rho0, H = 8.484e-9, 8.131
        rho = rho0 * np.exp(-(h - 130) / H)
    elif h < 150:
        rho0, H = 3.845e-9, 9.492
        rho = rho0 * np.exp(-(h - 140) / H)
    elif h < 160:
        rho0, H = 2.07e-9, 11.06
        rho = rho0 * np.exp(-(h - 150) / H)
    elif h < 180:
        rho0, H = 5.464e-10, 16.08
        rho = rho0 * np.exp(-(h - 160) / H)
    elif h < 200:
        rho0, H = 2.789e-10, 22.33
        rho = rho0 * np.exp(-(h - 180) / H)
    elif h < 250:
        rho0, H = 7.248e-11, 29.74
        rho = rho0 * np.exp(-(h - 200) / H)
    elif h < 300:
        rho0, H = 2.418e-11, 37.105
        rho = rho0 * np.exp(-(h - 250) / H)
    elif h < 350:
        rho0, H = 9.518e-12, 45.546
        rho = rho0 * np.exp(-(h - 300) / H)
    elif h < 400:
        rho0, H = 3.725e-12, 53.628
        rho = rho0 * np.exp(-(h - 350) / H)
    elif h < 450:
        rho0, H = 1.585e-12, 53.298
        rho = rho0 * np.exp(-(h - 400) / H)
    elif h < 500:
        rho0, H = 6.967e-13, 58.515
        rho = rho0 * np.exp(-(h - 450) / H)
    elif h < 600:
        rho0, H = 1.454e-13, 60.828
        rho = rho0 * np.exp(-(h - 500) / H)
    elif h < 700:
        rho0, H = 3.614e-14, 63.822
        rho = rho0 * np.exp(-(h - 600) / H)
    elif h < 800:
        rho0, H = 1.170e-14, 71.835
        rho = rho0 * np.exp(-(h - 700) / H)
    elif h < 900:
        rho0, H = 5.245e-15, 88.667
        rho = rho0 * np.exp(-(h - 800) / H)
    elif h <= 1000:
        rho0, H = 3.019e-15, 124.64
        rho = rho0 * np.exp(-(h - 900) / H)
    else:
        rho = 0.0

    return rho

##############################################################################
#                        ORBITAL STATE DERIVATIVES                           #
##############################################################################

def calculate_state_derivatives(state_vector, effective_Cd):
    """
    Returns the 6D state derivative for [x, y, z, vx, vy, vz].
    Includes gravity (with J2) and drag.
    """
    x, y, z, vx, vy, vz = state_vector
    derivatives = np.zeros(6)

    # Position derivatives
    derivatives[0:3] = [vx, vy, vz]

    # Constants
    G  = 6.67430e-11
    M  = 5.972e24
    Re = 6378137.0
    J2 = 1.08263e-3

    r = np.sqrt(x**2 + y**2 + z**2)
    r2 = r*r
    # Direction cosines
    x_r = x / r
    y_r = y / r
    z_r = z / r

    # Grav + J2
    factor = G * M / r2
    coeff  = 1.5 * J2 * (Re**2) / r2
    common = 5 * z_r**2 - 1

    a_x = -factor * (x_r + coeff * x_r * common)
    a_y = -factor * (y_r + coeff * y_r * common)
    a_z = -factor * (z_r + coeff * z_r * (5 * z_r**2 - 3))

    # Drag
    altitude = r - 6371000.0  # Earth's mean radius (m)
    rho = calculate_atmospheric_density(altitude)
    if altitude < 0:
        rho = 0.0

    A = 4.0     # Cross-sectional area (m^2)
    m = 500.0   # Mass (kg)

    v_rel = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v_rel) + 1e-10

    drag_accel = -0.5 * effective_Cd * A * rho * v_mag * v_rel / m

    derivatives[3] = a_x + drag_accel[0]
    derivatives[4] = a_y + drag_accel[1]
    derivatives[5] = a_z + drag_accel[2]

    return derivatives

def calculate_drag_force(state_vector):
    """
    Return the instantaneous drag force vector [Fx, Fy, Fz] in Newtons.
    """
    x, y, z, vx, vy, vz = state_vector
    r = np.sqrt(x**2 + y**2 + z**2)
    altitude = r - 6371000.0
    rho = calculate_atmospheric_density(altitude)
    if altitude < 0:
        rho = 0.0

    cd = calculate_drag_perturbation(state_vector)
    A  = 4.0
    m  = 500.0

    v_rel = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v_rel) + 1e-10
    drag_accel = -0.5 * cd * A * rho * v_mag * v_rel / m
    drag_force = drag_accel * m  # F = m*a

    return drag_force

##############################################################################
#                              ODE WRAPPER                                   #
##############################################################################

def dynamics(t, state):
    """
    ODE function for solve_ivp. 
    """
    cd = calculate_drag_perturbation(state)
    return calculate_state_derivatives(state, cd)

def run_simulation(initial_classical_state, time_steps):
    """
    Solve the ODE with 8(9) integrator (DOP853). 
    Returns array of shape (n_steps, 6).
    """
    sol = solve_ivp(fun=dynamics,
                    t_span=(time_steps[0], time_steps[-1]),
                    y0=initial_classical_state,
                    t_eval=time_steps,
                    method='DOP853')
    return sol.y.T  # shape -> (n_steps, 6)

##############################################################################
#                                MAIN CODE                                   #
##############################################################################

# Initial conditions
initial_positions = np.array([3280100, -2958420, 5170410])   # (m)
initial_velocities = np.array([3.62125e3, 6.58049e3, 1.46809e3])  # (m/s)
initial_classical_state = np.concatenate((initial_positions, initial_velocities))

# Time steps: 1-second increments for a day
time_steps = np.arange(0, 60 * 60*24, 1.0)  # 86400 seconds total

# Solve the entire orbit
trajectory = run_simulation(initial_classical_state, time_steps)
positions   = trajectory[:, :3]
velocities  = trajectory[:, 3:]

# Save data (optional)
np.savetxt(
    'orbit_positions.dat',
    positions,
    header='X Y Z',
    fmt='%.6e',
    delimiter=' ',
    comments=''
)

# Check for NaN/Inf
if np.isnan(positions).any() or np.isinf(positions).any():
    print("Warning: positions contain NaN or Inf values.")

##############################################################################
#                    REAL-TIME PLOTTING WITH VELOCITY & DRAG                 #
##############################################################################

plt.ion()  # Turn on interactive mode for live updating
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Real-Time Orbital Trajectory with Velocity & Drag Vectors")

# Draw Earth (wireframe or surface)
r_earth = 6371000.0
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = r_earth * np.outer(np.cos(u), np.sin(v))
y_sphere = r_earth * np.outer(np.sin(u), np.sin(v))
z_sphere = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=0.2)

# Prepare line objects and quivers for real-time updates
orbit_line, = ax.plot([], [], [], 'r-', label='Trajectory', linewidth=2)
pos_vector = None
vel_vector = None
drag_vector = None

# Axis bounds
max_radius = np.max(np.linalg.norm(positions, axis=1))
ax.set_xlim(-max_radius, max_radius)
ax.set_ylim(-max_radius, max_radius)
ax.set_zlim(-max_radius, max_radius)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()

# Scales for arrow visibility (tweak as needed)
vel_scale  = 1e-3  # velocity is large, so scale down
drag_scale = 1e-5  # drag force is usually small, so might scale up or down

for i in range(len(positions)):
    x, y, z = positions[i]
    vx, vy, vz = velocities[i]

    # Position from t=0 up to current index
    orbit_line.set_data(positions[:i+1, 0], positions[:i+1, 1])
    orbit_line.set_3d_properties(positions[:i+1, 2])

    # Remove old quivers each time so we can draw new ones
    if pos_vector: 
        pos_vector.remove()
    if vel_vector: 
        vel_vector.remove()
    if drag_vector:
        drag_vector.remove()

    # 1) Position vector: from Earth's center to current position
    pos_vector = ax.quiver(
        0, 0, 0,        # start
        x, y, z,        # direction
        color='g',
        arrow_length_ratio=0.0,  # no arrow tip (it's a line)
        linewidth=0.01
    )

    # 2) Velocity vector: from current position, scaled
    vel_vector = ax.quiver(
        x, y, z,                   # start at satellite
        vx * vel_scale,            # scaled velocity
        vy * vel_scale,
        vz * vel_scale,
        color='r',
        arrow_length_ratio=0.2
    )

    # 3) Drag force vector: from current position, scaled
    drag_vec = calculate_drag_force(trajectory[i])  # [Fx, Fy, Fz]
    drag_vector = ax.quiver(
        x, y, z,
        drag_vec[0] * drag_scale,
        drag_vec[1] * drag_scale,
        drag_vec[2] * drag_scale,
        color='b',
        arrow_length_ratio=0.2
    )

    # Redraw
    plt.draw()
    plt.pause(0.001)
    # plt.pause(1)

# Keep final plot open
plt.ioff()
plt.show()