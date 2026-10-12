"""Check that the NumPy mirror and the Qiskit circuits give identical statevectors (run after `pip install qiskit`)."""
import numpy as np, qmodel as Q
rng = np.random.default_rng(0); X = rng.uniform(-1.5, 1.5, (20, Q.N_QUBITS)); th = rng.uniform(-3, 3, Q.N_THETA)
for name, t in [('feature map only', None), ('full VQC circuit', th)]:
    a, b = Q.numpy_states(X, t), Q.qiskit_states(X, t)
    print(f'{name}: max |amplitude difference| = {np.abs(a - b).max():.2e}')
