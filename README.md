# MetaAI-Simulation
Enabling Over-the-Air AI for Edge Computing via Metasurface-Driven Physical Neural Networks


# Assumptions
Far-field Conditions: Assumes far-field propagation for simplified path loss calculations
Linear Neural Networks: Implements single-layer linear networks.
2-bit Metasurface: Uses 4 discrete phase states (0, π/2, π, 3π/2) for meta-atoms
Synthetic Data: Uses generated data instead of real datasets for demonstration
Simplified Channel: Uses a basic AWGN channel model without complex multipath
Fixed Geometry: Assumes fixed transmitter-metasurface-receiver geometry

# Parameters

# Network Parameters
U = 64: Input dimension (reduced from paper for faster simulation)
R = 10: Number of output classes
M = 256: Number of meta-atoms (as optimized in the paper)

# Training Parameters
learning_rate = 8e-3: Learning rate for complex backpropagation
num_epochs = 60: Training epochs
batch_size = 64: Mini-batch size

# Wireless Parameters
SNR_dB = 20: Signal-to-noise ratio
symbol_rate = 1e6: Transmission symbol rate (1 MHz)

# Metasurface Parameters
phase_states = [0, pi/2, pi, 3*pi/2]: 2-bit phase quantization
d_s = 0.05: Meta-atom spacing in wavelengths
theta = 30°: Incidence angle

# To Run script 

run('metaai_simulation.m');

