"""
Quantum Stock Trend Prediction — Core Model Module
====================================================
Implements a 4-qubit Variational Quantum Classifier (VQC) and a Quantum Kernel
SVM for next-day stock trend prediction (UP vs DOWN).

Circuit architecture
--------------------
Feature map (ZZ-style, fm_reps=2 repetitions):
    H -> P(2*x_i) -> CX . P(2*x_i*x_{i+1}) . CX    (for each qubit / pair)
Ansatz (N_LAYERS=2 layers):
    [RY(theta_k) . RZ(theta_{k+1})]_{q=0..n-1} -> CX chain
Final layer:
    RY(theta_k) for each qubit
Readout:
    <Z_0> = 1 - 2*P(qubit_0=|1>) -> sigmoid(a*<Z_0> + b) = P(UP)

Two backends are provided:
    - 'numpy'  : Fast statevector simulation (default, ~10 s for full run)
    - 'qiskit' : Exact statevector via qiskit.quantum_info.Statevector
      (identical results to ~1e-15; slow due to per-sample circuit binding)

Usage
-----
>>> from qmodel import VQC, QuantumKernelSVM, load, FEATS, N_QUBITS
>>> df = load('data/quantum_stock_prediction_dataset.csv')
>>> vqc = VQC(backend='numpy').fit(X_train, y_train, maxiter=300)
>>> p_up = vqc.predict_proba(X_test)   # float array in [0, 1]

References
----------
Perez-Salinas et al. (2020): "Data re-uploading for a universal quantum classifier"
Havlicek et al. (2019): "Supervised learning with quantum-enhanced feature spaces"
McClean et al. (2018): "Barren plateaus in quantum neural network training landscapes"
"""
import numpy as np, pandas as pd

N_QUBITS, N_LAYERS, FM_REPS = 4, 2, 2
N_THETA = 2 * N_QUBITS * N_LAYERS + N_QUBITS          # 20 trainable circuit angles
FEATS = ['daily_return', 'return_5d', 'return_20d', 'volatility_20d', 'vol_chg', 'sentiment_score',
         'analyst_rating', 'news_count', 'sma5_gap', 'sma20_gap', 'range_pct', 'volatile_regime']

# ---------------------------------------------------------------- data
def load(path):
    df = pd.read_csv(path).sort_values(['date', 'ticker']).reset_index(drop=True)
    df['sma5_gap'] = df.close / df.sma_5 - 1
    df['sma20_gap'] = df.close / df.sma_20 - 1
    df['range_pct'] = (df.high - df.low) / df.close
    df['vol_chg'] = np.log1p(df.volume_change.clip(-0.9, 5))
    df['volatile_regime'] = (df.market_regime == 'volatile').astype(int)
    df = df.dropna(subset=['target_trend'])          # last row of each ticker has no label
    df['y'] = df.target_trend.map({'UP': 1, 'DOWN': 0, 'SIDEWAYS': -1})
    return df

def to_angles(Xraw, mean, std):
    """Standardise (train statistics) and squash into [-pi/2, pi/2] for rotation encoding."""
    return np.clip((Xraw - mean) / std, -3, 3) * np.pi / 6

# ---------------------------------------------------------------- Qiskit circuits
def qiskit_circuit(x, theta, with_ansatz=True, n=N_QUBITS):
    """x, theta: sequences of floats or qiskit Parameters/ParameterVectors."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for _ in range(FM_REPS):                          # ZZ-style feature map
        qc.h(range(n))
        for q in range(n): qc.p(2 * x[q], q)
        for i in range(n - 1):
            qc.cx(i, i + 1); qc.p(2 * x[i] * x[i + 1], i + 1); qc.cx(i, i + 1)
    if with_ansatz:                                   # variational ansatz
        k = 0
        for _ in range(N_LAYERS):
            for q in range(n):
                qc.ry(theta[k], q); qc.rz(theta[k + 1], q); k += 2
            for i in range(n - 1): qc.cx(i, i + 1)
        for q in range(n): qc.ry(theta[k], q); k += 1
    return qc

def qiskit_states(X, theta=None, n=N_QUBITS):
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector
    xs = ParameterVector('x', n)
    ts = ParameterVector('t', N_THETA) if theta is not None else None
    tmpl = qiskit_circuit(xs, ts, theta is not None, n)
    out = []
    for x in X:
        b = dict(zip(xs, x))
        if theta is not None: b.update(zip(ts, theta))
        out.append(Statevector(tmpl.assign_parameters(b)).data)
    return np.array(out)

# ---------------------------------------------------------------- NumPy mirror (same gates, same little-endian qubit order)
def _g1(psi, q, U, n):
    B = psi.shape[0]; ax = n - q                      # qubit q <-> axis n-q (axis 0 is batch)
    p = np.moveaxis(psi.reshape((B,) + (2,) * n), ax, -1)
    out = p @ U.T if U.ndim == 2 else np.einsum('b...j,bij->b...i', p, U)
    return np.moveaxis(out, -1, ax).reshape(B, -1)

def _cx(psi, c, t, n):
    B = psi.shape[0]; p = psi.reshape((B,) + (2,) * n).copy()
    ac, at = n - c, n - t
    idx = [slice(None)] * (n + 1); idx[ac] = 1
    p[tuple(idx)] = np.flip(p[tuple(idx)], axis=at - 1 if at > ac else at).copy()
    return p.reshape(B, -1)

_H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
def _P(phi):
    U = np.zeros((len(phi), 2, 2), complex); U[:, 0, 0] = 1; U[:, 1, 1] = np.exp(1j * phi); return U
def _RY(t):
    c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -s], [s, c]], complex)
def _RZ(t): return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])

def numpy_states(X, theta=None, n=N_QUBITS):
    B = len(X); psi = np.zeros((B, 2 ** n), complex); psi[:, 0] = 1
    for _ in range(FM_REPS):
        for q in range(n): psi = _g1(psi, q, _H, n)
        for q in range(n): psi = _g1(psi, q, _P(2 * X[:, q]), n)
        for i in range(n - 1):
            psi = _cx(psi, i, i + 1, n); psi = _g1(psi, i + 1, _P(2 * X[:, i] * X[:, i + 1]), n); psi = _cx(psi, i, i + 1, n)
    if theta is not None:
        k = 0
        for _ in range(N_LAYERS):
            for q in range(n):
                psi = _g1(psi, q, _RY(theta[k]), n); psi = _g1(psi, q, _RZ(theta[k + 1]), n); k += 2
            for i in range(n - 1): psi = _cx(psi, i, i + 1, n)
        for q in range(n): psi = _g1(psi, q, _RY(theta[k]), n); k += 1
    return psi

def states(X, theta=None, backend='numpy'):
    return qiskit_states(X, theta) if backend == 'qiskit' else numpy_states(X, theta)

# ---------------------------------------------------------------- models
def p1(psi):  # probability that qubit 0 is measured as |1>
    return (np.abs(psi) ** 2).reshape(len(psi), -1, 2)[:, :, 1].sum(1)

def sigmoid(z): return 1 / (1 + np.exp(-z))

class VQC:
    """Variational quantum classifier: angle encoding -> ZZ feature map -> trainable ansatz -> <Z0> -> sigmoid(a*<Z0>+b)."""
    def __init__(self, backend='numpy', seed=0):
        self.backend = backend; rng = np.random.default_rng(seed)
        self.w = np.concatenate([rng.normal(0, 0.3, N_THETA), [2.0, 0.0]]); self.history = []
    def predict_proba(self, X, w=None):
        w = self.w if w is None else w
        z = 1 - 2 * p1(states(X, w[:N_THETA], self.backend))   # <Z0>
        return sigmoid(w[-2] * z + w[-1])
    def fit(self, X, y, maxiter=300):
        from scipy.optimize import minimize
        def loss(w):
            p = np.clip(self.predict_proba(X, w), 1e-6, 1 - 1e-6)
            l = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)); self.history.append(float(l)); return l
        self.w = minimize(loss, self.w, method='COBYLA', options={'maxiter': maxiter, 'rhobeg': 0.5}).x
        return self

class QuantumKernelSVM:
    """Fidelity quantum kernel k(x,x')=|<phi(x)|phi(x')>|^2 fed to a classical SVM."""
    def __init__(self, backend='numpy', C=1.0): self.backend, self.C = backend, C
    def _K(self, A, B): return np.abs(A @ B.conj().T) ** 2
    def fit(self, X, y):
        from sklearn.svm import SVC
        self.S = states(X, None, self.backend); self.svm = SVC(kernel='precomputed', C=self.C, probability=True, random_state=0)
        self.svm.fit(self._K(self.S, self.S), y); return self
    def predict_proba(self, X):
        return self.svm.predict_proba(self._K(states(X, None, self.backend), self.S))[:, 1]
