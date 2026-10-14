"""Belief state: a Kalman filter and an echo-state reservoir, and their exact memory.

`docs/MATH.md` Section 5. The Kalman filter is the minimum-variance belief for a
linear-Gaussian system; the reservoir is a fixed recurrent feature map whose only
trained part is a linear readout. Both are O(state) memory, flat in sequence
length, which is the property the 8 GB budget depends on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def kalman(Y: np.ndarray, A: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray,
           x0: np.ndarray | None = None, P0: np.ndarray | None = None) -> np.ndarray:
    """Standard Kalman filter. Y: (T, p). Returns filtered states (T, r).

    predict: x- = A x, P- = A P A^T + Q
    gain:    G  = P- C^T (C P- C^T + R)^-1
    update:  x  = x- + G (y - C x-),  P = (I - G C) P-
    """
    r = A.shape[0]
    x = np.zeros(r) if x0 is None else x0.astype(float).copy()
    P = np.eye(r) if P0 is None else P0.astype(float).copy()
    out = np.empty((len(Y), r))
    for t, y in enumerate(Y):
        x = A @ x
        P = A @ P @ A.T + Q
        S = C @ P @ C.T + R
        G = P @ C.T @ np.linalg.inv(S)
        x = x + G @ (y - C @ x)
        P = (np.eye(r) - G @ C) @ P
        out[t] = x
    return out


def steady_state_gain(A: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray,
                      iters: int = 5000, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Iterate the Riccati recursion to the steady-state gain G and covariance P.

    The fixed point of P <- A P A^T + Q - A P C^T (C P C^T + R)^-1 C P A^T
    solves the discrete algebraic Riccati equation.
    """
    r = A.shape[0]
    P = np.eye(r)
    for _ in range(iters):
        P_new = A @ P @ A.T + Q - A @ P @ C.T @ np.linalg.inv(C @ P @ C.T + R) @ C @ P @ A.T
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    G = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
    return G, P


@dataclass
class Reservoir:
    """A sparse echo-state reservoir (Jaeger 2001). Only the readout is trained.

    Spectral radius < 1 gives the echo state property: the state forgets its
    initial condition and becomes a function of the recent input only.
    """

    n_inputs: int
    n_units: int = 300
    spectral_radius: float = 0.9
    sparsity: float = 0.05
    input_scaling: float = 0.5
    leak: float = 0.3
    seed: int = 42

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        W = rng.uniform(-1, 1, size=(self.n_units, self.n_units))
        W[rng.random(W.shape) > self.sparsity] = 0.0
        # scale to the target spectral radius without a dense eigendecomposition
        v = rng.normal(size=self.n_units)
        for _ in range(200):
            v = W @ v
            nrm = np.linalg.norm(v)
            if nrm < 1e-12:
                break
            v = v / nrm
        radius = float(np.linalg.norm(W @ v)) if nrm > 1e-12 else 1.0
        if radius > 1e-9:
            W *= self.spectral_radius / radius
        self.W = W.astype(np.float32)
        self.W_in = (rng.uniform(-1, 1, size=(self.n_units, self.n_inputs)) * self.input_scaling).astype(np.float32)
        self.bias = (rng.uniform(-1, 1, size=self.n_units) * 0.2).astype(np.float32)

    def run(self, U: np.ndarray, washout: int = 0) -> np.ndarray:
        """Drive the reservoir with U: (T, n_inputs). Returns states (T-washout, n_units)."""
        state = np.zeros(self.n_units, dtype=np.float32)
        keep = max(len(U) - washout, 0)
        out = np.empty((keep, self.n_units), dtype=np.float32)
        for t in range(len(U)):
            pre = self.W_in @ U[t] + self.W @ state + self.bias
            state = (1 - self.leak) * state + self.leak * np.tanh(pre)
            if t >= washout:
                out[t - washout] = state
        return out

    def memory_bytes(self) -> int:
        """Bytes held at inference: the weight matrix plus one state vector."""
        return self.W.nbytes + self.W_in.nbytes + self.bias.nbytes + self.n_units * 4


def state_memory_bytes(n_layer: int, n_head: int, d_head: int, bits: int = 32) -> int:
    """Constant attention state for a linear-attention / recurrent stack."""
    return n_layer * n_head * d_head * d_head * bits // 8


def kv_memory_bytes(n_layer: int, n_kv_head: int, d_head: int, seq_len: int, bits: int = 16) -> int:
    """KV cache for a full-attention stack: grows linearly in seq_len."""
    return n_layer * 2 * n_kv_head * d_head * seq_len * bits // 8
