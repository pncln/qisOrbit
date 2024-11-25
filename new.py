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

from mpl_toolkits.mplot3d import Axes3D

sim = AerSimulator()
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

    # Define drag components
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

    # Normalize the components
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
        prob
