from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sim = AerSimulator()
# --------------------------
# Simulating using estimator
#---------------------------
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer.primitives import SamplerV2

# Define your state vectors
V_1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Example vector
V_2 = np.array([0, 1])                         # Example vector
R_1 = np.array([1/np.sqrt(2), -1/np.sqrt(2)])  # Example vector
R_2 = np.array([1, 0])                         # Example vector

# ========================== SATELLITE ===============================

# Create quantum circuits for each state vector
qc_V1 = QuantumCircuit(1)
qc_V2 = QuantumCircuit(1)
qc_R1 = QuantumCircuit(1)
qc_R2 = QuantumCircuit(1)

# Convert state vectors to Statevector objects
state_V1 = Statevector(V_1)
state_V2 = Statevector(V_2)
state_R1 = Statevector(R_1)
state_R2 = Statevector(R_2)

# Initialize the quantum circuits with the state vectors
qc_V1.initialize(state_V1, 0)
qc_V2.initialize(state_V2, 0)
qc_R1.initialize(state_R1, 0)
qc_R2.initialize(state_R2, 0)

# Optional: Add measurement if you want to measure the states
qc_V1.measure_all()
qc_V2.measure_all()
qc_R1.measure_all()
qc_R2.measure_all()

# Transpile the circuits
qc_V1_t = transpile(qc_V1, sim)
qc_V2_t = transpile(qc_V2, sim)
qc_R1_t = transpile(qc_R1, sim)
qc_R2_t = transpile(qc_R2, sim)

# Using the sampler from your existing code
sampler = SamplerV2()
job = sampler.run([qc_V1_t, qc_V2_t, qc_R1_t, qc_R2_t], shots=128)
results = job.result()

# Print results for each circuit
for i, result in enumerate(results):
    print(f"Circuit {i} counts: {result.data.meas.get_counts()}")

# ========================== SATELLITE END ============================

# ================== PERTURBATION ===============================

# Define 3D force components (x, y, z) using 3 qubits
# Each force vector is represented as an 8-dimensional state vector (2^3)

# Solar radiation pressure with x,y,z components
solar_radiation = np.array([
    0.5,    # |000⟩ - baseline
    0.5,    # |001⟩ - z component
    0.5,    # |010⟩ - y component
    0.25,   # |011⟩ - y,z interaction
    0.25,   # |100⟩ - x component
    0.25,   # |101⟩ - x,z interaction
    0.25,   # |110⟩ - x,y interaction
    0.1     # |111⟩ - x,y,z interaction
]) / np.sqrt(sum(np.array([0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.1])**2))

# Atmospheric drag with directional components
atm_drag = np.array([
    0.6,    # stronger drag effect in primary direction
    0.4,    # z component
    0.4,    # y component
    0.2,    # y,z interaction
    0.3,    # x component
    0.2,    # x,z interaction
    0.2,    # x,y interaction
    0.1     # x,y,z interaction
]) / np.sqrt(sum(np.array([0.6, 0.4, 0.4, 0.2, 0.3, 0.2, 0.2, 0.1])**2))

# Create 3-qubit quantum circuits
qc_solar = QuantumCircuit(3)
qc_drag = QuantumCircuit(3)

# Convert to Statevector objects and initialize
state_solar = Statevector(solar_radiation)
state_drag = Statevector(atm_drag)

qc_solar.initialize(state_solar, [0, 1, 2])
qc_drag.initialize(state_drag, [0, 1, 2])

# Add measurements for all qubits
qc_solar.measure_all()
qc_drag.measure_all()

# Transpile and run
sim = AerSimulator()
qc_solar_t = transpile(qc_solar, sim)
qc_drag_t = transpile(qc_drag, sim)

# Run with more shots to get better statistics
sampler = SamplerV2()
job = sampler.run([qc_solar_t, qc_drag_t], shots=2000)
results = job.result()

# Print results with force component interpretation
force_names = ['Solar Radiation', 'Atmospheric Drag']
for i, result in enumerate(results):
    print(f"\n{force_names[i]} Force Components:")
    counts = result.data.meas.get_counts()
    for state, count in counts.items():
        print(f"State |{state}⟩: {count} counts - ", end="")
        if state == '000': print("Baseline")
        elif state == '001': print("Z component")
        elif state == '010': print("Y component")
        elif state == '011': print("Y-Z interaction")
        elif state == '100': print("X component")
        elif state == '101': print("X-Z interaction")
        elif state == '110': print("X-Y interaction")
        elif state == '111': print("X-Y-Z interaction")

# =================== PERTURBATION END ==========================


# ======================= Combined perturbation =================================

# This code:

# Creates different superposition states representing combined perturbation effects
# Uses 4 qubits to capture more complex interactions
# Implements specific quantum circuits for different force combinations
# Uses Hadamard gates to create superpositions
# Adds controlled-X gates to model interactions between forces
# Includes phase relationships using RZ gates
# Measures and displays the probability distribution of different combined states

# Create a 4-qubit circuit to represent combined perturbations
qc_combined = QuantumCircuit(4)

# Apply Hadamard gates to create superposition of all forces
qc_combined.h(range(4))

# Create specific superpositions for different combined effects

# Solar radiation + Atmospheric drag superposition
qc_solar_drag = QuantumCircuit(4)
qc_solar_drag.h(0)  # Superposition for solar radiation
qc_solar_drag.x(1)  # Set drag effect
qc_solar_drag.h(2)  # Superposition for interaction
qc_solar_drag.h(3)  # Additional interaction terms

# J2 + Third body superposition
qc_j2_third = QuantumCircuit(4)
qc_j2_third.h(0)
qc_j2_third.h(1)
qc_j2_third.cx(0, 2)  # Entangle J2 with interaction qubit
qc_j2_third.cx(1, 3)  # Entangle third-body with interaction qubit

# Create a superposition of all perturbations
qc_all_forces = QuantumCircuit(4)
# Create equal superposition
qc_all_forces.h(range(4))
# Add phase relationships
qc_all_forces.rz(np.pi/4, 0)
qc_all_forces.rz(np.pi/3, 1)
qc_all_forces.cx(0, 2)
qc_all_forces.cx(1, 3)

# Add measurements
for qc in [qc_combined, qc_solar_drag, qc_j2_third, qc_all_forces]:
    qc.measure_all()

# Transpile circuits
sim = AerSimulator()
circuits_t = [transpile(qc, sim) for qc in [qc_combined, qc_solar_drag, qc_j2_third, qc_all_forces]]

# Run the sampler
sampler = SamplerV2()
job = sampler.run(circuits_t, shots=2000)
results = job.result()

# Print results with interpretations
labels = ["Combined Effects", "Solar-Drag", "J2-Third Body", "All Forces"]
for i, result in enumerate(results):
    print(f"\n{labels[i]} Superposition Measurements:")
    counts = result.data.meas.get_counts()
    # Sort by count frequency
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    for state, count in sorted_counts.items():
        probability = count/2000
        print(f"State |{state}⟩: {probability:.3f} probability")

# ====================== Combined perturbation end ===========================

def interpret_quantum_state(state):
    # Dictionary to map binary states to physical meanings
    force_interpretations = {
        '000000': 'No perturbation',
        '000001': 'Solar radiation (weak)',
        '000010': 'Atmospheric drag (weak)',
        '000011': 'Solar radiation + Atmospheric drag',
        '000100': 'J2 effect (weak)',
        '000101': 'J2 + Solar radiation',
        '000110': 'J2 + Atmospheric drag',
        '000111': 'Combined orbital forces',
        '001000': 'Third body effect',
        '001001': 'Third body + Solar radiation',
        '001010': 'Third body + Atmospheric drag',
        '001100': 'Third body + J2',
        '010000': 'Solar wind',
        '100000': 'Thermal radiation'
    }
    
    # Calculate force magnitude from probability
    def get_magnitude(prob):
        if prob > 0.5: return "Strong"
        if prob > 0.2: return "Medium"
        return "Weak"
    
    return force_interpretations.get(state, f"Complex interaction state {state}")

# =================== TIME EVOLUTION OF PERTURBATION =========================

# This code implements:

# Time-dependent quantum gates to model force evolution
# RZ and RX gates for phase and amplitude evolution
# Periodic behaviors using sine and cosine functions
# Exponential decay for atmospheric drag
# Time-dependent coupling between different forces
# Multiple time steps to track the evolution
# Measurement statistics at each time step
# The results show how the quantum states evolve over time, representing the changing magnitudes and interactions of perturbative forces during orbital motion.

# Create a time-dependent circuit with 4 qubits
def create_advanced_time_evolution_circuit(t):
    qc = QuantumCircuit(6)  # Increased to 6 qubits for more interactions
    
    # Initial state preparation with complex superposition
    qc.h(range(6))
    qc.rx(np.pi/4, [0, 2, 4])
    qc.ry(np.pi/3, [1, 3, 5])
    
    # Complex time-dependent evolution
    
    # Solar radiation with daily variation
    daily_cycle = np.sin(2 * np.pi * t / 24)
    seasonal_cycle = np.sin(2 * np.pi * t / (365 * 24))
    qc.rz(0.5 * t * daily_cycle, 0)
    qc.rx(0.25 * t * seasonal_cycle, 0)
    
    # Atmospheric drag with altitude-dependent decay
    altitude_factor = np.exp(-0.1 * t) * (1 + 0.2 * np.sin(0.1 * t))
    qc.rz(0.2 * t * altitude_factor, 1)
    qc.rx(0.15 * t * altitude_factor, 1)
    
    # J2 effect with latitude dependence
    latitude_variation = np.sin(0.3 * t) * np.cos(0.1 * t)
    qc.rz(0.3 * latitude_variation, 2)
    qc.rx(0.25 * latitude_variation, 2)
    
    # Third body effects (Moon + Sun)
    lunar_cycle = np.sin(2 * np.pi * t / (27.3 * 24))  # 27.3 days lunar period
    solar_cycle = np.sin(2 * np.pi * t / (365 * 24))   # Annual period
    qc.rz(0.15 * lunar_cycle, 3)
    qc.rx(0.1 * solar_cycle, 3)
    
    # Solar wind variations
    solar_wind = np.sin(0.4 * t) * (1 + 0.3 * np.cos(0.1 * t))
    qc.rz(0.2 * solar_wind, 4)
    qc.rx(0.15 * solar_wind, 4)
    
    # Thermal radiation effects
    thermal_cycle = np.sin(2 * np.pi * t / 24) * np.exp(-0.05 * t)
    qc.rz(0.1 * thermal_cycle, 5)
    qc.rx(0.08 * thermal_cycle, 5)
    
    # Enhanced coupling terms
    # Multi-qubit interactions
    qc.rzz(0.1 * t, 0, 2)  # Solar-J2
    qc.rzz(0.15 * t, 1, 3)  # Drag-Third body
    qc.rzz(0.12 * t, 4, 5)  # Solar wind-Thermal
    
    # Three-qubit interactions
    qc.ccx(0, 1, 2)  # Solar-Drag-J2
    qc.ccx(3, 4, 5)  # Third body-Solar wind-Thermal
        
        # Phase-dependent couplings
    qc.cp(0.1 * t * daily_cycle, 0, 4)
    qc.cp(0.15 * t * lunar_cycle, 3, 5)

        # Controlled rotations
    qc.crx(0.1 * altitude_factor, 1, 4)
    qc.cry(0.2 * solar_wind, 2, 5)
        
    qc.measure_all()
    return qc

# Simulate over multiple time steps with finer granularity
shots = 1000
time_steps = np.linspace(0, 48, 100)  # 10 time steps over 48 hours
circuits = [create_advanced_time_evolution_circuit(t) for t in time_steps]

# Transpile and run
sim = AerSimulator()
circuits_t = [transpile(qc, sim) for qc in circuits]
sampler = SamplerV2()
job = sampler.run(circuits_t, shots=2000)
results = job.result()

# Analyze results with enhanced interpretation
for t, result in zip(time_steps, results):
    print(f"\nTime t = {t:.2f} hours:")
    counts = result.data.meas.get_counts()
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:6])
    
    for state, count in sorted_counts.items():
        probability = count/2000
        # Calculate total perturbation strength
        active_qubits = state.count('1')
        print(f"State |{state}⟩: {probability:.3f} probability (Interaction level: {active_qubits})")

        physical_meaning = interpret_quantum_state(state)
        print(f"{physical_meaning}: {probability:.3f} probability")

# ================= TIME EVOLUTION OF PERTURBATION END =======================

# ========= GRAPHICS ==========================================================

def plot_force_evolution():
    # Store results for plotting
    time_steps = np.linspace(0, 48, 20)
    circuits = [create_advanced_time_evolution_circuit(t) for t in time_steps]
    circuits_t = [transpile(qc, sim) for qc in circuits]
    
    # Run with high shot count
    sampler = SamplerV2()
    results = sampler.run(circuits_t, shots=20000).result()
    
    # Create data structure for plotting
    force_evolution = defaultdict(list)
    
    # Collect probabilities over time
    for result in results:
        counts = result.data.meas.get_counts()
        total_shots = sum(counts.values())
        # Store actual measured probabilities
        for state in counts:
            force_evolution[state].append(counts[state] / total_shots)
    
    # Select top states based on actual measurements
    top_states = sorted(force_evolution.keys(), 
                       key=lambda x: sum(force_evolution[x]), 
                       reverse=True)[:5]
    
    # Create visualization plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Time evolution with actual data
    plt.subplot(2, 2, 1)
    for state in top_states:
        plt.plot(time_steps, force_evolution[state], 
                label=interpret_quantum_state(state), 
                marker='o')
    plt.title('Evolution of Most Probable States')
    plt.xlabel('Time (hours)')
    plt.ylabel('Probability')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Heatmap with measured data
    heatmap_data = np.array([force_evolution[state] for state in top_states])
    plt.subplot(2, 2, 2)
    sns.heatmap(heatmap_data, 
                xticklabels=[f'{t:.1f}' for t in time_steps],
                yticklabels=[interpret_quantum_state(state) for state in top_states],
                cmap='viridis')
    
    # Plot 3: Interaction strength from measurements
    interaction_strength = []
    for t_idx in range(len(time_steps)):
        strength = sum(state.count('1') * force_evolution[state][t_idx] 
                      for state in top_states)
        interaction_strength.append(strength)
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, interaction_strength, 'r-', linewidth=2)
    
    # Plot 4: Final distribution using actual measurements
    final_probs = {state: force_evolution[state][-1] for state in top_states}
    plt.subplot(2, 2, 4)
    plt.bar(range(len(final_probs)), list(final_probs.values()))
    plt.xticks(range(len(final_probs)), 
               [interpret_quantum_state(state) for state in final_probs.keys()], 
               rotation=45)
    
    plt.tight_layout()
    plt.savefig('quantum_perturbation_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()


# Run the visualization
plot_force_evolution()

# ============== END GRAPHICS ==================================================

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

    sim = AerSimulator()
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
        rho0 = 1.225                  # kg/m^3 at sea level
        H = 7.249                     # Scale height in km
        rho = rho0 * np.exp(-h / H)
    elif h < 30:
        rho0 = 3.899e-2
        H = 6.349
        rho = rho0 * np.exp(-(h - 25) / H)
    elif h < 40:
        rho0 = 1.774e-2
        H = 6.682
        rho = rho0 * np.exp(-(h - 30) / H)
    elif h < 50:
        rho0 = 3.972e-3
        H = 7.554
        rho = rho0 * np.exp(-(h - 40) / H)
    elif h < 60:
        rho0 = 1.057e-3
        H = 8.382
        rho = rho0 * np.exp(-(h - 50) / H)
    elif h < 70:
        rho0 = 3.206e-4
        H = 7.714
        rho = rho0 * np.exp(-(h - 60) / H)
    elif h < 80:
        rho0 = 8.770e-5
        H = 6.549
        rho = rho0 * np.exp(-(h - 70) / H)
    elif h < 90:
        rho0 = 1.905e-5
        H = 5.799
        rho = rho0 * np.exp(-(h - 80) / H)
    elif h < 100:
        rho0 = 3.396e-6
        H = 5.382
        rho = rho0 * np.exp(-(h - 90) / H)
    elif h < 110:
        rho0 = 5.297e-7
        H = 5.877
        rho = rho0 * np.exp(-(h - 100) / H)
    elif h < 120:
        rho0 = 9.661e-8
        H = 6.396
        rho = rho0 * np.exp(-(h - 110) / H)
    elif h < 130:
        rho0 = 2.438e-8
        H = 7.054
        rho = rho0 * np.exp(-(h - 120) / H)
    elif h < 140:
        rho0 = 8.484e-9
        H = 8.131
        rho = rho0 * np.exp(-(h - 130) / H)
    elif h < 150:
        rho0 = 3.845e-9
        H = 9.492
        rho = rho0 * np.exp(-(h - 140) / H)
    elif h < 160:
        rho0 = 2.07e-9
        H = 11.06
        rho = rho0 * np.exp(-(h - 150) / H)
    elif h < 180:
        rho0 = 5.464e-10
        H = 16.08
        rho = rho0 * np.exp(-(h - 160) / H)
    elif h < 200:
        rho0 = 2.789e-10
        H = 22.33
        rho = rho0 * np.exp(-(h - 180) / H)
    elif h < 250:
        rho0 = 7.248e-11
        H = 29.74
        rho = rho0 * np.exp(-(h - 200) / H)
    elif h < 300:
        rho0 = 2.418e-11
        H = 37.105
        rho = rho0 * np.exp(-(h - 250) / H)
    elif h < 350:
        rho0 = 9.518e-12
        H = 45.546
        rho = rho0 * np.exp(-(h - 300) / H)
    elif h < 400:
        rho0 = 3.725e-12
        H = 53.628
        rho = rho0 * np.exp(-(h - 350) / H)
    elif h < 450:
        rho0 = 1.585e-12
        H = 53.298
        rho = rho0 * np.exp(-(h - 400) / H)
    elif h < 500:
        rho0 = 6.967e-13
        H = 58.515
        rho = rho0 * np.exp(-(h - 450) / H)
    elif h < 600:
        rho0 = 1.454e-13
        H = 60.828
        rho = rho0 * np.exp(-(h - 500) / H)
    elif h < 700:
        rho0 = 3.614e-14
        H = 63.822
        rho = rho0 * np.exp(-(h - 600) / H)
    elif h < 800:
        rho0 = 1.170e-14
        H = 71.835
        rho = rho0 * np.exp(-(h - 700) / H)
    elif h < 900:
        rho0 = 5.245e-15
        H = 88.667
        rho = rho0 * np.exp(-(h - 800) / H)
    elif h <= 1000:
        rho0 = 3.019e-15
        H = 124.64
        rho = rho0 * np.exp(-(h - 900) / H)
    else:
        # For altitudes above 1000 km, density is negligible
        rho = 0.0

    return rho

##############################################################################

def rk4_step(state_vector, dt):
    """
    RK4 integration step for orbital propagation.
    """
    k1 = calculate_derivatives(state_vector)
    k2 = calculate_derivatives(state_vector + 0.5 * dt * k1)
    k3 = calculate_derivatives(state_vector + 0.5 * dt * k2)
    k4 = calculate_derivatives(state_vector + dt * k3)

    next_state = state_vector + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state

def calculate_derivatives(state):
    """
    Compute derivatives including both gravitational force and drag.
    """
    # Calculate effective drag coefficient
    effective_Cd = calculate_drag_perturbation(state)

    # Compute state derivatives
    derivatives = calculate_state_derivatives(state, effective_Cd)
    return derivatives

def propagate_orbit(initial_state, duration, dt):
    """
    Propagate orbit for visualization
    """
    steps = int(duration/dt)
    trajectory = np.zeros((steps, 6))
    trajectory[0] = initial_state
    
    for i in range(1, steps):
        trajectory[i] = rk4_step(trajectory[i-1], dt)
    
    return trajectory

def plot_3d_orbit(trajectory):
    """
    Create 3D visualization of the orbit with Earth
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Earth
    r_earth = 6371000  # Earth radius in meters
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r_earth * np.outer(np.cos(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.sin(v))
    z = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            'r-', label='Orbit', linewidth=2)
    
    # Plot starting point
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
              color='green', s=100, label='Start')
    
    # Set axis limits based on orbit size
    max_range = np.max(np.abs(trajectory))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Orbital Trajectory Around Earth')
    ax.legend()
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.savefig('earth_orbit_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate and visualize trajectory
# Position Vector: 3280.1, -2958.42, 5170.41 m
# Velocity Vector: 3.62125, 6.58049, 1.46809 m/s

initial_state = np.array([3280100, -2958420, 5170410, 3.62125*1000, 6.58049*1000, 1.46809*1000])
duration = 3600  # 1 hour simulation
dt = 10.0       # 10-second steps

trajectory = propagate_orbit(initial_state, duration, dt)
plot_3d_orbit(trajectory)


##############################################################################


def variational_ansatz(params):
    num_qubits = 6  # Adjust based on the complexity of your system
    qc = QuantumCircuit(num_qubits)
    
    # Apply parameterized rotations
    for i in range(num_qubits):
        qc.rx(params[i], i)
        qc.ry(params[i + num_qubits], i)
    
    # Include entangling gates to capture interactions
    for i in range(num_qubits - 1):
        qc.cz(i, i + 1)
    
    qc.barrier()
    return qc

def cost_function(params, target_state):
    qc = variational_ansatz(params)
    sv = Statevector.from_instruction(qc)
    
    fidelity = np.abs(np.dot(np.conj(target_state.data), sv.data)) ** 2
    cost = 1 - fidelity
    return cost

def variational_time_evolution(initial_params, target_state):
    # Define the objective function for optimization
    def objective(params):
        return cost_function(params, target_state)
    
    # Perform optimization to find the optimal parameters
    result = minimize(objective, initial_params, method='BFGS')
    
    optimized_params = result.x
    return optimized_params

def run_variational_simulation(initial_state_vector, time_steps):
    num_qubits = 6  # Adjust based on the complexity of your system
    parameter_size = 2 * num_qubits  # Number of parameters in the ansatz
    
    # Initialize parameters (e.g., zeros or small random values)
    params = np.zeros(parameter_size)
    
    # Store the trajectory for visualization
    trajectory = []
    
    # Loop over time steps
    for t in time_steps:
        # Define the target state at time t
        # For demonstration, we'll assume the target state evolves trivially
        # In practice, you'd compute or define the desired state at each time step
        target_state = initial_state_vector  # Placeholder
        
        # Optimize parameters to approximate the target state
        optimized_params = variational_time_evolution(params, target_state)
        
        # Update parameters for the next time step
        params = optimized_params
        
        # Generate the quantum state from the optimized parameters
        qc = variational_ansatz(params)
        sv = Statevector.from_instruction(qc)
        
        # Extract classical information
        classical_state = extract_classical_state(sv)
        trajectory.append(classical_state)
    
    return np.array(trajectory)

def extract_classical_state(statevector):
    # This function maps the quantum state to classical state variables
    # Here, we use expectation values as a simple example
    num_qubits = 6
    positions = []
    velocities = []
    
    for i in range(num_qubits):
        # Expectation value of Pauli-Z for each qubit
        z_expectation = statevector.expectation_value('Z', i).real
        
        if i < 3:
            # Map to position components (e.g., x, y, z)
            positions.append(z_expectation)
        else:
            # Map to velocity components (e.g., vx, vy, vz)
            velocities.append(z_expectation)
    
    # Scale the expectation values to match physical units
    # These scaling factors need to be determined based on your encoding
    position_scale = 1e7  # Example scaling factor for positions
    velocity_scale = 1e3  # Example scaling factor for velocities
    
    positions = np.array(positions) * position_scale
    velocities = np.array(velocities) * velocity_scale
    
    classical_state = np.concatenate((positions, velocities))
    return classical_state

# Define initial classical state vector
# Replace with actual initial state values
initial_positions = np.array([7000e3, 0, 0])  # Positions in meters
initial_velocities = np.array([0, 7.5e3, 0])  # Velocities in meters per second
initial_classical_state = np.concatenate((initial_positions, initial_velocities))

# Convert initial classical state to quantum state vector
# This requires encoding your classical state into a quantum state
# For demonstration, we'll create a simple statevector
initial_state_vector = Statevector.from_label('000000')

# Define time steps
duration = 3600  # Total simulation time in seconds (e.g., 1 hour)
num_steps = 20
time_steps = np.linspace(0, duration, num_steps)

# Run the variational simulation
trajectory = run_variational_simulation(initial_state_vector, time_steps)

# Plot the trajectory
plot_3d_orbit(trajectory)