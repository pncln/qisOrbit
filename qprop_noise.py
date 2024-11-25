from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import numpy as np
from scipy.optimize import minimize
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

from qiskit_ibm_runtime import QiskitRuntimeService
# QiskitRuntimeService.save_account(channel="ibm_quantum", token="f86c0fffd56ce4afc7791a1ed24ae5f9a2eda284f797526dbe5668d20583e3dfd32dd2f5bf56ce07e2996675edb9aa25f922ecc0f605b6494ba12bbe363afe87")
service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")
noise_model = NoiseModel.from_backend(backend)

sim = AerSimulator(noise_model=noise_model)
# --------------------------
# Simulating using sampler
#---------------------------
from qiskit_aer.primitives import SamplerV2

##################### DRAG ########################################################

def calculate_drag_perturbation(state_vector):
    """
    Calculate effective drag coefficient based on quantum measurements.

    Args:
        state_vector: [x, y, z, vx, vy, vz] numpy array
    """
    x, y, z, vx, vy, vz = state_vector

    # Quantum simulation as before
    qc_drag = QuantumCircuit(3)
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # Define drag components (same as your code)
    drag_components = np.array([
        0.6,    # baseline
        0.4,    # z component
        0.4,    # y component
        0.2,    # y,z interaction
        0.3,    # x component
        0.2,    # x,z interaction
        0.2,    # x,y interaction
        0.1     # x,y,z interaction
    ])

    drag_components = drag_components / np.sqrt(np.sum(drag_components**2))

    state_drag = Statevector(drag_components)
    qc_drag.initialize(state_drag, [0, 1, 2])
    qc_drag.measure_all()

    sim = AerSimulator(noise_model=noise_model)
    qc_drag_t = transpile(qc_drag, sim)
    sampler = SamplerV2()
    result = sampler.run([qc_drag_t], shots=2000).result()

    counts = result[0].data.meas.get_counts()

    # Calculate effective drag coefficient based on probabilities
    total_shots = sum(counts.values())
    effective_Cd = 0.0

    # Base drag coefficient (nominal value)
    Cd_nominal = 2.2  # Typical value for satellites

    for state, count in counts.items():
        prob = count / total_shots
        # Modify Cd based on state probabilities
        # You can define how each state affects the drag coefficient
        if state == '000':
            # Baseline effect
            effective_Cd += Cd_nominal * prob * 1.0
        elif state == '001':
            # Z component increases Cd
            effective_Cd += Cd_nominal * prob * 1.1
        elif state == '010':
            # Y component increases Cd
            effective_Cd += Cd_nominal * prob * 1.1
        elif state == '011':
            # Y-Z interaction further increases Cd
            effective_Cd += Cd_nominal * prob * 1.2
        elif state == '100':
            # X component increases Cd
            effective_Cd += Cd_nominal * prob * 1.1
        elif state == '101':
            # X-Z interaction
            effective_Cd += Cd_nominal * prob * 1.2
        elif state == '110':
            # X-Y interaction
            effective_Cd += Cd_nominal * prob * 1.2
        elif state == '111':
            # X-Y-Z interaction (maximum effect)
            effective_Cd += Cd_nominal * prob * 1.3

    return effective_Cd


##############################################################################


############### ORBITAL STATE DERIVATIVES ####################################

def calculate_state_derivatives(state_vector, effective_Cd):
    x, y, z, vx, vy, vz = state_vector

    # Initialize the derivatives array
    derivatives = np.zeros(6)

    # Position derivatives (velocities)
    derivatives[0:3] = [vx, vy, vz]

    # Constants
    G = 6.67430e-11       # Gravitational constant (m^3 kg^-1 s^-2)
    M = 5.972e24          # Earth's mass (kg)
    R_e = 6378137.0       # Earth's equatorial radius (m)
    J2 = 1.08263e-3       # Earth's second zonal harmonic (dimensionless)

    # Compute common terms
    r = np.sqrt(x**2 + y**2 + z**2)         # Magnitude of position vector
    r2 = r ** 2                             # Square of distance
    r5 = r ** 5                             # r to the fifth power
    z2 = z ** 2                             # Square of z-component

    # Direction cosines
    x_r = x / r
    y_r = y / r
    z_r = z / r

    # Gravitational acceleration with J2 perturbation
    factor = G * M / r2
    coeff = 1.5 * J2 * (R_e ** 2) / r2
    common_term = 5 * z_r ** 2 - 1

    a_x = -factor * (x_r + coeff * x_r * common_term)
    a_y = -factor * (y_r + coeff * y_r * common_term)
    a_z = -factor * (z_r + coeff * z_r * (5 * z_r ** 2 - 3))

    # Atmospheric density
    altitude = r - 6371000.0  # Earth's mean radius in meters
    rho = calculate_atmospheric_density(altitude)

    # Ensure non-negative atmospheric density
    if altitude < 0:
        rho = 0.0

    # Drag acceleration
    A = 4.0       # Cross-sectional area (m^2)
    m = 500.0     # Mass of satellite (kg)

    v_rel = np.array([vx, vy, vz])  # Assuming atmosphere is stationary
    v_rel_mag = np.linalg.norm(v_rel) + 1e-10  # Avoid division by zero

    drag_acceleration = (-0.5 * effective_Cd * A * rho * v_rel_mag * v_rel) / m

    # Total accelerations
    derivatives[3] = a_x + drag_acceleration[0]
    derivatives[4] = a_y + drag_acceleration[1]
    derivatives[5] = a_z + drag_acceleration[2]

    return derivatives

def calculate_atmospheric_density(altitude):
    """
    Calculate atmospheric density based on a piecewise exponential model with variable scale heights.

    Args:
        altitude (float): Altitude above Earth's surface in meters.

    Returns:
        float: Atmospheric density in kg/m^3.
    """
    # Convert altitude to kilometers
    h = altitude / 1000.0

    if h < 0:
        return 0.0

    # Define scale heights (H) and base densities (rho0) for different altitude ranges
    # Data sourced from the U.S. Standard Atmosphere 1976 model
    if h < 25:
        rho0 = 2.3450e-1                  # kg/m^3 at 12.500 meters (took average between 25k and sea level)
        H = 7.249                     # Scale height in km
        rho = rho0 * np.exp(-h / H)
    elif h < 30:
        rho0 = 2.1717e-2
        H = 6.349
        rho = rho0 * np.exp(-(h - 25) / H)
    elif h < 40:
        rho0 = 6.7052e-3
        H = 6.682
        rho = rho0 * np.exp(-(h - 30) / H)
    elif h < 50:
        rho0 = 1.5357e-3
        H = 7.554
        rho = rho0 * np.exp(-(h - 40) / H)
    elif h < 60:
        rho0 = 4.6376e-4
        H = 8.382
        rho = rho0 * np.exp(-(h - 50) / H)
    elif h < 70:
        rho0 = 1.3323e-4
        H = 7.714
        rho = rho0 * np.exp(-(h - 60) / H)
    elif h < 80:
        rho0 = 3.2589e-5
        H = 6.549
        rho = rho0 * np.exp(-(h - 70) / H)
    elif h < 90:
        rho0 = 6.7099e-6
        H = 5.799
        rho = rho0 * np.exp(-(h - 80) / H)
    elif h < 100:
        rho0 = 1.137e-6
        H = 5.382
        rho = rho0 * np.exp(-(h - 90) / H)
    elif h < 110:
        rho0 = 1.898e-7
        H = 5.877
        rho = rho0 * np.exp(-(h - 100) / H)
    elif h < 120:
        rho0 = 3.501e-8
        H = 6.396
        rho = rho0 * np.exp(-(h - 110) / H)
    elif h < 130:
        rho0 = 1.054e-8
        H = 7.054
        rho = rho0 * np.exp(-(h - 120) / H)
    elif h < 140:
        rho0 = 4.461e-9
        H = 8.131
        rho = rho0 * np.exp(-(h - 130) / H)
    elif h < 150:
        rho0 = 2.270e-9
        H = 9.492
        rho = rho0 * np.exp(-(h - 140) / H)
    elif h < 160:
        rho0 = 1.294e-9
        H = 11.06
        rho = rho0 * np.exp(-(h - 150) / H)
    elif h < 180:
        rho0 = 6.380e-10
        H = 16.08
        rho = rho0 * np.exp(-(h - 160) / H)
    elif h < 200:
        rho0 = 5.174e-10
        H = 22.33
        rho = rho0 * np.exp(-(h - 180) / H)
    elif h < 250:
        rho0 = 9.665e-11
        H = 29.74
        rho = rho0 * np.exp(-(h - 200) / H)
    elif h < 300:
        rho0 = 2.718e-11
        H = 37.105
        rho = rho0 * np.exp(-(h - 250) / H)
    elif h < 350:
        rho0 = 9.521e-12
        H = 45.546
        rho = rho0 * np.exp(-(h - 300) / H)
    elif h < 400:
        rho0 = 3.655e-12
        H = 53.628
        rho = rho0 * np.exp(-(h - 350) / H)
    elif h < 450:
        rho0 = 5.725e-12
        H = 53.298
        rho = rho0 * np.exp(-(h - 400) / H)
    elif h < 500:
        rho0 = 6.708e-13
        H = 58.515
        rho = rho0 * np.exp(-(h - 450) / H)
    elif h < 600:
        rho0 = 1.946e-13
        H = 60.828
        rho = rho0 * np.exp(-(h - 500) / H)
    elif h < 700:
        rho0 = 4.663e-14
        H = 63.822
        rho = rho0 * np.exp(-(h - 600) / H)
    elif h < 800:
        rho0 = 1.460e-14
        H = 71.835
        rho = rho0 * np.exp(-(h - 700) / H)
    elif h < 900:
        rho0 = 6.387e-14
        H = 88.667
        rho = rho0 * np.exp(-(h - 800) / H)
    elif h <= 1000: 
        rho0 = 3.635e-15 #950km
        H = 124.64
        rho = rho0 * np.exp(-(h - 900) / H)
    else:
        # For altitudes above 1000 km, density is negligible
        rho = 0.0

    return rho

##############################################################################

def binary_encode_state_vector(classical_state):
    """
    Encode classical state vector into a quantum state using binary encoding.

    Args:
        classical_state (numpy array): [x, y, z, vx, vy, vz]

    Returns:
        QuantumCircuit: Quantum circuit representing the encoded state.
    """
    # Define the number of bits for binary representation
    num_bits = 32  # Adjust based on desired precision
    num_vars = 6  # Number of state variables
    num_qubits = num_bits * num_vars

    qc = QuantumCircuit(num_qubits)
    
    # Define min and max values for normalization
    min_values = np.array([-1e7, -1e7, -1e7, -1e4, -1e4, -1e4])  # Adjust as needed
    max_values = np.array([1e7, 1e7, 1e7, 1e4, 1e4, 1e4])        # Adjust as needed

    # Normalize and encode each classical value
    for i, value in enumerate(classical_state):
        # Normalize value to [0, 1)
        normalized_value = (value - min_values[i]) / (max_values[i] - min_values[i])

        # Convert normalized value to integer
        integer_value = int(normalized_value * (2 ** num_bits - 1))

        # Convert integer to binary string
        binary_string = format(integer_value, f'0{num_bits}b')

        # Encode binary string onto qubits using X gates
        for j, bit in enumerate(reversed(binary_string)):
            if bit == '1':
                qc.x(i * num_bits + j)

    return qc

def extract_classical_state_from_binary(qc, expected_state):
    """
    Extract classical state variables from the quantum state using binary decoding.
    
    Args:
        qc (QuantumCircuit): Quantum circuit representing the encoded state.
    
    Returns:
        numpy array: Classical state vector [x, y, z, vx, vy, vz]
    """
    num_vars = 6
    num_bits = 32  # Set to match the encoding function
    num_qubits = num_vars * num_bits

    # Add measurement to the circuit
    qc_measure = qc.copy()
    qc_measure.measure_all()

    # Use AerSimulator for sampling
    simulator = AerSimulator(noise_model=noise_model, method='matrix_product_state')
    compiled_circuit = transpile(qc_measure, optimization_level=0)
    result = simulator.run(compiled_circuit, shots=1).result()
    counts = result.get_counts()

    if not counts:
        raise ValueError("No measurement results obtained from the quantum circuit.")

    # Get the measured bitstring
    measured_state = list(counts.keys())[0]
    measured_state = measured_state.replace(' ', '')  # Remove spaces if any

    # Debugging output
    print(f"Measured state (before reversing): {measured_state}")

    # The measured_state may need to be reversed due to endianness
    measured_state = measured_state[::-1]

    # Debugging output
    print(f"Measured state (after reversing): {measured_state}")

    classical_state = []

    # Define min and max values for de-normalization
    min_values = np.array([-1e7, -1e7, -1e7, -1e4, -1e4, -1e4])
    max_values = np.array([1e7, 1e7, 1e7, 1e4, 1e4, 1e4])

    # Check if the measured_state has the expected length
    expected_length = num_vars * num_bits
    if len(measured_state) < expected_length:
        raise ValueError(f"Measured state length ({len(measured_state)}) is less than expected ({expected_length}).")

    # Split the bitstring and decode each variable
    for i in range(num_vars):
        # Extract bits for the current variable
        bit_start = i * num_bits
        bit_end = bit_start + num_bits
        binary_string = measured_state[bit_start:bit_end][::-1]  # Reverse bits

        # Debugging statement
        print(f"Variable {i}: binary_string='{binary_string}'")

        # Convert binary string to integer
        integer_value = int(binary_string, 2)
        
        # Normalize back to [0, 1)
        normalized_value = integer_value / (2 ** num_bits - 1)
        
        # De-normalize to original value
        value = normalized_value * (max_values[i] - min_values[i]) + min_values[i]
        classical_state.append(value)

    error = np.abs(classical_state - expected_state)

    return np.array(classical_state), error
    
def run_variational_simulation(initial_classical_state, time_steps):
    """
    Run the variational simulation over the specified time steps using RK4 integration.
    Records the error at each time step.

    Args:
        initial_classical_state (numpy array): Initial [x, y, z, vx, vy, vz]
        time_steps (numpy array): Array of time steps

    Returns:
        tuple: (trajectory, errors)
    """
    num_bits = 32  # Number of bits per variable
    num_vars = 6  # Number of variables

    # Store the trajectory and errors
    trajectory = []
    errors = []

    # Encode the initial classical state into a quantum circuit
    qc = binary_encode_state_vector(initial_classical_state)
    # Extract the classical state from the quantum circuit and compute error
    classical_state, error = extract_classical_state_from_binary(qc, initial_classical_state)
    trajectory.append(classical_state)
    errors.append(error)

    # Loop over time steps
    for t_idx in range(1, len(time_steps)):
        dt = time_steps[t_idx] - time_steps[t_idx - 1]

        # Get the current state
        current_state = trajectory[-1]

        # Runge-Kutta 4th Order Method

        # Compute k1
        effective_Cd_k1 = calculate_drag_perturbation(current_state)
        k1 = calculate_state_derivatives(current_state, effective_Cd_k1)

        # Compute k2
        state_k2 = current_state + 0.5 * dt * k1
        effective_Cd_k2 = calculate_drag_perturbation(state_k2)
        k2 = calculate_state_derivatives(state_k2, effective_Cd_k2)

        # Compute k3
        state_k3 = current_state + 0.5 * dt * k2
        effective_Cd_k3 = calculate_drag_perturbation(state_k3)
        k3 = calculate_state_derivatives(state_k3, effective_Cd_k3)

        # Compute k4
        state_k4 = current_state + dt * k3
        effective_Cd_k4 = calculate_drag_perturbation(state_k4)
        k4 = calculate_state_derivatives(state_k4, effective_Cd_k4)

        # Compute the next state
        next_state = current_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Encode the updated state into a quantum circuit
        qc = binary_encode_state_vector(next_state)

        # Extract the classical state from the quantum circuit and compute error
        classical_state, error = extract_classical_state_from_binary(qc, next_state)
        trajectory.append(classical_state)
        errors.append(error)

    return np.array(trajectory), np.array(errors)



# Define initial classical state vector
initial_positions = np.array([3280100, -2958420, 5170410])  # Positions in meters
initial_velocities = np.array([3.62125*1000, 6.58049*1000, 1.46809*1000])  # Velocities in meters per second
initial_classical_state = np.concatenate((initial_positions, initial_velocities))

# Define time steps
duration = 80  # Total simulation time in seconds (e.g., 1 hour)
num_steps = 40
time_steps = np.linspace(0, duration, num_steps)

# Run the variational simulation
trajectory, errors = run_variational_simulation(initial_classical_state, time_steps)

positions = trajectory[:, :3]

# Run checks
if np.isnan(positions).any() or np.isinf(positions).any():
    print("Warning: positions array contains NaN or Inf values.")

print("Positions array:", positions)

# Plot the trajectory
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
r_earth = 6371000  # Earth radius in meters
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r_earth * np.outer(np.cos(u), np.sin(v))
y = r_earth * np.outer(np.sin(u), np.sin(v))
z = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='blue', alpha=0.3)

ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
        'r-', label='Orbit', linewidth=2)
ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
           color='green', s=100, label='Start')

max_range = np.max(np.abs(positions))
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Orbital Trajectory Around Earth')
ax.legend()

ax.set_box_aspect([1, 1, 1])

plt.show()

# # Run the variational simulation
# trajectory, errors = run_variational_simulation(initial_classical_state, time_steps)

# positions = trajectory[:, :3]

# Compute total error as the norm of the error vector at each time step
total_errors = np.linalg.norm(errors, axis=1)

# Plot the error over time
# plt.figure(figsize=(10, 6))
# plt.plot(time_steps, total_errors, label='Total Error')
# plt.xlabel('Time (s)')
# plt.ylabel('Error Magnitude')
# plt.title('Accumulation of Quantum Circuit Noise Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()