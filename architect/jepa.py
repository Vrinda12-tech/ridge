"""A JEPA-style latent predictor with the classical readout that tells you whether it collapsed.

The maths is `docs/MATH.md` Section 5. Two things are demonstrated, and both are
measured, not asserted:

  * a predictor trained only on next-step prediction collapses (the encoder maps
    everything to ~0), so the latent carries no information;
  * the *dimension* of the latent should be read from the data, not chosen: the
    Hankel singular-value gap gives the number of independent frequencies, and
    DMD on the Hankel matrix returns them with |lambda| = 1 for a non-collapsing,
    non-exploding latent.

The anti-collapse terms are the participation-ratio floor and VICReg's
variance/covariance hinges. Nothing here is copied; it is the same physics as the
referenced work, re-derived and re-tested.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- data reading

def hankel(X: np.ndarray, K: int) -> np.ndarray:
    """Takens delay matrix. X: (T, m) -> H: (m*K, T-K+1)."""
    T, m = X.shape
    return np.vstack([X[k:T - K + 1 + k].T for k in range(K)])


def hankel_rank_gap(X: np.ndarray, K: int, top: int = 12) -> tuple[int, np.ndarray]:
    """Rank from the largest multiplicative gap in the Hankel singular spectrum.

    Returns (r_star, normalized singular values).
    """
    H = hankel(X - X.mean(0), K)
    s = np.linalg.svd(H, compute_uv=False)
    s = s / s[0]
    n = min(top, len(s) - 1)
    ratios = s[:n] / np.maximum(s[1:n + 1], 1e-12)
    r_star = int(np.argmax(ratios)) + 1
    return r_star, s[:top]


def dmd(X: np.ndarray, K: int, r: int, dt: float = 1.0) -> dict:
    """Exact DMD on the Hankel matrix (docs/MATH.md 5.1).

    Returns the reduced Koopman matrix, its eigenvalues, the recovered angular
    frequencies and the growth rates. |lambda| = 1 means neutral stability.
    """
    H = hankel(X - X.mean(0), K)
    H1, H2 = H[:, :-1], H[:, 1:]
    U, S, Vt = np.linalg.svd(H1, full_matrices=False)
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T
    Atil = Ur.T @ H2 @ Vr @ np.diag(1.0 / Sr)
    lam = np.linalg.eigvals(Atil)
    omega = np.abs(np.angle(lam)) / dt
    growth = np.log(np.maximum(np.abs(lam), 1e-300)) / dt
    return {
        "Atil": Atil,
        "eigenvalues": lam,
        "abs_lambda": np.abs(lam),
        "omega": np.sort(omega),
        "growth": growth,
    }


def participation_ratio(C: np.ndarray) -> float:
    """PR = (sum mu)^2 / sum mu^2 of the covariance C (docs/MATH.md 5.2)."""
    mu = np.linalg.eigvalsh(0.5 * (C + C.T))
    mu = np.clip(mu, 0.0, None)
    s1, s2 = mu.sum(), (mu ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def vicreg_terms(Z: np.ndarray, gamma: float = 1.0, eps: float = 1e-4) -> tuple[float, float]:
    """Variance hinge and covariance penalty on Z: (N, k)."""
    Zc = Z - Z.mean(0)
    std = np.sqrt(Zc.var(0) + eps)
    v = float(np.mean(np.maximum(0.0, gamma - std)))
    C = Zc.T @ Zc / max(len(Z) - 1, 1)
    off = C - np.diag(np.diag(C))
    c = float((off ** 2).sum() / Z.shape[1])
    return v, c


# ------------------------------------------------------------------- the model

@dataclass
class TrainReport:
    mode: str
    seed: int
    latent_energy: float
    participation_ratio: float
    next_step_r2: float
    hankel_rank: int
    dmd_omega: list
    dmd_abs_lambda: list
    grad_norm: float


def _regularizer_grad(Z: np.ndarray, mode: str, r_star: int,
                      gamma: float = 1.0, eps: float = 1e-4) -> tuple[float, np.ndarray]:
    """Return (loss, dloss/dZ) for the anti-collapse terms. Z: (N,k).

    mode "none" is pure prediction with no regularizer, so it is free to collapse.
    Every other mode carries the energy floor; "vicreg" adds the VICReg hinges,
    "koopman_dim" adds the participation-ratio dimension floor.
    """
    N, k = Z.shape
    Zc = Z - Z.mean(0)
    g = np.zeros_like(Z)
    loss = 0.0

    if mode == "none":
        return 0.0, g

    # energy floor: stop the whole latent shrinking to zero
    var = Zc.var(0)
    energy_deficit = max(0.0, 1.0 - float(var.mean()))
    loss += energy_deficit ** 2
    if energy_deficit > 0:
        d_energy = -2.0 * energy_deficit / k
        g += d_energy * 2.0 * Zc / max(N - 1, 1)

    if mode == "vicreg":
        std = np.sqrt(var + eps)
        hinge = np.maximum(0.0, gamma - std)
        loss += float(hinge.mean())
        active = hinge > 0
        if active.any():
            dstd = np.zeros(k)
            dstd[active] = -1.0 / k
            g += (dstd / std)[None, :] * Zc / max(N - 1, 1)
        C = Zc.T @ Zc / max(N - 1, 1)
        off = C - np.diag(np.diag(C))
        loss += float((off ** 2).sum() / k)
        dC = 2.0 * off / k
        g += 2.0 * (Zc @ dC) / max(N - 1, 1)

    if mode == "koopman_dim":
        # Dimension floor on the participation ratio, with the gradient taken by
        # the exact chain rule: C = Zc^T Zc/(N-1) => dL/dZ = 2/(N-1) * Zc @ dL/dC.
        # dL/dC is a numeric gradient on the small (k x k) covariance.
        C = Zc.T @ Zc / max(N - 1, 1) + eps * np.eye(k)
        pr = participation_ratio(C)
        deficit = max(0.0, r_star - 0.5 - pr)
        loss += deficit ** 2
        if deficit > 0:
            dC = np.zeros_like(C)
            h = 1e-6
            for i in range(k):
                for j in range(i, k):
                    Cp = C.copy(); Cp[i, j] += h
                    Cm = C.copy(); Cm[i, j] -= h
                    if i != j:
                        Cp[j, i] += h; Cm[j, i] -= h
                    dpr = (participation_ratio(Cp) - participation_ratio(Cm)) / (2 * h)
                    val = -2.0 * deficit * dpr
                    dC[i, j] += val
                    if i != j:
                        dC[j, i] += val
            g += (2.0 / max(N - 1, 1)) * (Zc @ dC)

    return loss, g


def train_latent(X: np.ndarray, d_lat: int, mode: str = "none", r_star: int = 4,
                 steps: int = 1200, lr: float = 1e-2, seed: int = 0,
                 reg_weight: float = 1.0) -> tuple[np.ndarray, np.ndarray, list[float], TrainReport]:
    """Train encoder W (d_lat, d_in) and predictor P (d_lat, d_lat) on next-step prediction.

    Returns (W, P, loss_history, report). Modes: "none", "vicreg", "koopman_dim".
    """
    rng = np.random.default_rng(seed)
    T, d_in = X.shape
    Xc = X - X.mean(0)
    W = rng.normal(0, 1.0 / np.sqrt(d_in), size=(d_lat, d_in))
    P = np.eye(d_lat)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mP = np.zeros_like(P); vP = np.zeros_like(P)
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    hist = []
    grad_norm = 0.0

    for step in range(1, steps + 1):
        Z = Xc @ W.T                      # (T, d_lat)
        Z1, Z2 = Z[:-1], Z[1:]
        E = Z1 @ P.T - Z2
        pred = float((E ** 2).mean())

        dP = (2.0 / len(Z1)) * (E.T @ Z1)
        dZ = np.zeros_like(Z)
        dZ[:-1] += (2.0 / len(Z1)) * (E @ P)
        dZ[1:] += -(2.0 / len(Z1)) * E

        reg, dZ_reg = _regularizer_grad(Z, mode, r_star)
        dZ += reg_weight * dZ_reg
        loss = pred + reg_weight * reg
        dW = dZ.T @ Xc

        grad_norm = float(np.sqrt((dW ** 2).sum() + (dP ** 2).sum()))
        for (prm, gr, m, v) in ((W, dW, mW, vW), (P, dP, mP, vP)):
            m *= b1; m += (1 - b1) * gr
            v *= b2; v += (1 - b2) * gr * gr
            mh = m / (1 - b1 ** step); vh = v / (1 - b2 ** step)
            prm -= lr * mh / (np.sqrt(vh) + eps_a)
        hist.append(loss)

    Z = Xc @ W.T
    Z1, Z2 = Z[:-1], Z[1:]
    E = Z1 @ P.T - Z2
    ss_res = float((E ** 2).sum())
    ss_tot = float(((Z2 - Z2.mean(0)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rank, _ = hankel_rank_gap(Z, K=min(20, max(4, Z.shape[0] // 20)))
    dm = dmd(Z, K=min(20, max(4, Z.shape[0] // 20)), r=min(d_lat, 4))
    report = TrainReport(
        mode=mode, seed=seed,
        latent_energy=float((Z ** 2).mean()),
        participation_ratio=participation_ratio(np.cov(Z, rowvar=False)),
        next_step_r2=r2,
        hankel_rank=rank,
        dmd_omega=[round(float(w), 4) for w in dm["omega"]],
        dmd_abs_lambda=[round(float(a), 4) for a in dm["abs_lambda"]],
        grad_norm=grad_norm,
    )
    return W, P, hist, report


def make_rank_r_series(T: int = 2400, m: int = 8, freqs=(0.35, 0.72), noise: float = 0.05,
                       seed: int = 0) -> np.ndarray:
    """A linear system whose state is a sum of real sinusoids, so its Hankel rank is exactly 2*len(freqs).

    This is the ground truth the readout has to recover: the data decides the rank,
    not us.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    phases = rng.uniform(0, 2 * np.pi, size=m)
    A = rng.normal(size=(m, len(freqs)))
    B = rng.normal(size=(m, len(freqs)))
    X = np.zeros((T, m))
    for j, f in enumerate(freqs):
        X += np.outer(np.cos(2 * np.pi * f * t), A[:, j]) + np.outer(np.sin(2 * np.pi * f * t), B[:, j])
    X += noise * rng.standard_normal(X.shape)
    return X
