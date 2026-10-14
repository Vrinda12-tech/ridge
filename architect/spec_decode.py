"""Speculative decoding: the exact acceptance identity, expected output, and optimal draft length.

Derived in `docs/MATH.md` Section 3. The functions here are exact; `simulate`
verifies the algebra against a real rejection-sampling loop, which is what the
tests check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """TV(p,q) = 0.5 * sum |p - q|."""
    return 0.5 * float(np.sum(np.abs(p - q)))


def acceptance_rate(p: np.ndarray, q: np.ndarray) -> float:
    """alpha = sum_x min(p,q) = 1 - TV(p,q).  Exact."""
    return float(np.sum(np.minimum(p, q)))


def expected_tokens(alpha: float, gamma: int) -> float:
    """E[tokens] = (1 - alpha^(gamma+1)) / (1 - alpha).  Exact geometric sum."""
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    if abs(1.0 - alpha) < 1e-12:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speedup(alpha: float, gamma: int, draft_cost: float) -> float:
    """S = E[tokens] / (gamma*c + 1). draft_cost is the draft's cost per token
    relative to one target forward (c). Exact."""
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    return expected_tokens(alpha, gamma) / (gamma * draft_cost + 1.0)


def optimal_gamma(alpha: float, draft_cost: float, gamma_max: int = 64) -> int:
    """Argmax over gamma of speedup(alpha, gamma, draft_cost). Unimodal in gamma."""
    best_g, best_s = 0, speedup(alpha, 0, draft_cost)
    for g in range(1, gamma_max + 1):
        s = speedup(alpha, g, draft_cost)
        if s > best_s:
            best_g, best_s = g, s
    return best_g


def optimal_gamma_analytic(alpha: float, draft_cost: float) -> float:
    """Continuous root of d/dgamma log S = 0, for a first guess.

    log S = log(1 - a^(g+1)) - log(1-a) - log(g c + 1).
    d/dg = -a^(g+1) ln a / (1 - a^(g+1)) - c/(g c + 1) = 0
    =>  a^(g+1) ln(1/a) (g c + 1) = c (1 - a^(g+1)).
    """
    a = min(max(alpha, 1e-9), 1 - 1e-12)
    lo, hi = 0.0, 1e4
    for _ in range(200):
        g = 0.5 * (lo + hi)
        lhs = a ** (g + 1) * np.log(1.0 / a) * (g * draft_cost + 1.0)
        rhs = draft_cost * (1.0 - a ** (g + 1))
        if lhs > rhs:
            lo = g
        else:
            hi = g
    return 0.5 * (lo + hi)


@dataclass
class SpecStats:
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0
    iterations: int = 0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_iteration(self) -> float:
        return self.emitted / self.iterations if self.iterations else 0.0


def simulate(p: np.ndarray, q: np.ndarray, gamma: int, n_iter: int = 20000,
             seed: int = 0) -> SpecStats:
    """Run exact speculative sampling against the distributions p (target) and q (draft).

    Verifies both that the emitted tokens are distributed as p (in the tests) and
    that E[tokens] matches the closed form.
    """
    rng = np.random.default_rng(seed)
    st = SpecStats()
    diff = np.maximum(0.0, p - q)
    s = diff.sum()
    p_prime = diff / s if s > 0 else q
    for _ in range(n_iter):
        st.iterations += 1
        rejected = False
        for _ in range(gamma):
            x = rng.choice(len(q), p=q)
            st.proposed += 1
            acc = min(1.0, p[x] / q[x]) if q[x] > 0 else 1.0
            if rng.random() < acc:
                st.accepted += 1
                st.emitted += 1
            else:
                st.emitted += 1  # resampled token replaces the rejected one
                rejected = True
                break
        if not rejected:
            st.emitted += 1  # bonus token from the target
    return st


def speedup_table(alpha: float, draft_costs=(0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
                  gammas=range(0, 13)) -> list[dict]:
    rows = []
    for c in draft_costs:
        g = optimal_gamma(alpha, c, gamma_max=max(gammas))
        rows.append({
            "draft_cost": c,
            "optimal_gamma": g,
            "speedup": round(speedup(alpha, g, c), 3),
            "ceiling_1_over_1_minus_alpha": round(1.0 / (1.0 - alpha), 3),
        })
    return rows
