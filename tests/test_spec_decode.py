"""Exactness tests for the speculative-decoding identity (docs/MATH.md Section 3)."""
import numpy as np
import pytest

from architect.spec_decode import (
    acceptance_rate, expected_tokens, optimal_gamma, simulate, speedup, total_variation,
)


def _dist(rng, n=64):
    p = rng.random(n) + 1e-3
    return p / p.sum()


def test_acceptance_is_one_minus_tv():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p, q = _dist(rng), _dist(rng)
        assert acceptance_rate(p, q) == pytest.approx(1.0 - total_variation(p, q), abs=1e-12)


def test_acceptance_is_between_zero_and_one():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p, q = _dist(rng), _dist(rng)
        a = acceptance_rate(p, q)
        assert 0.0 <= a <= 1.0 + 1e-12


def test_expected_tokens_closed_form_matches_simulation():
    """The geometric sum must match a real rejection-sampling loop."""
    rng = np.random.default_rng(2)
    for _ in range(6):
        p, q = _dist(rng), _dist(rng)
        alpha = acceptance_rate(p, q)
        for gamma in (1, 2, 4, 8):
            st = simulate(p, q, gamma, n_iter=200000, seed=3)
            got = st.emitted / st.iterations
            assert got == pytest.approx(expected_tokens(alpha, gamma), rel=0.02, abs=0.02)


def test_emitted_distribution_is_the_target_distribution():
    """Losslessness: accepted+resampled tokens must be distributed as p, not q."""
    rng = np.random.default_rng(4)
    p, q = _dist(rng, 16), _dist(rng, 16)
    # force a real TV gap
    q = np.roll(q, 1)
    counts = np.zeros(len(p))
    # replicate simulate() but keep the emitted tokens
    n_iter = 400000
    diff = np.maximum(0.0, p - q)
    s = diff.sum()
    p_prime = diff / s if s > 0 else q
    gamma = 3
    for _ in range(n_iter):
        rejected = False
        for _ in range(gamma):
            x = rng.choice(len(q), p=q)
            acc = min(1.0, p[x] / q[x]) if q[x] > 0 else 1.0
            if rng.random() < acc:
                counts[x] += 1
            else:
                y = rng.choice(len(p), p=p_prime)
                counts[y] += 1
                rejected = True
                break
        if not rejected:
            y = rng.choice(len(p), p=p)
            counts[y] += 1
    emp = counts / counts.sum()
    assert np.abs(emp - p).max() < 0.01


def test_identical_distributions_give_alpha_one():
    rng = np.random.default_rng(5)
    p = _dist(rng)
    assert acceptance_rate(p, p) == pytest.approx(1.0, abs=1e-12)
    assert expected_tokens(1.0, 7) == pytest.approx(8.0)


def test_speedup_never_exceeds_the_ceiling():
    """S <= 1/(1-alpha) always, because E[tokens] <= 1/(1-alpha)."""
    for alpha in (0.2, 0.5, 0.8, 0.95):
        for c in (0.05, 0.1, 0.5):
            for g in range(0, 17):
                assert speedup(alpha, g, c) <= 1.0 / (1.0 - alpha) + 1e-9


def test_equal_cost_draft_is_a_loss():
    """c=1: no draft length beats a plain pass."""
    for alpha in (0.5, 0.8, 0.95):
        for g in range(1, 17):
            assert speedup(alpha, g, 1.0) <= 1.0 + 1e-9


def test_optimal_gamma_is_the_argmax():
    for alpha in (0.5, 0.7, 0.9):
        for c in (0.02, 0.1, 0.3):
            g = optimal_gamma(alpha, c, gamma_max=48)
            best = max(speedup(alpha, gg, c) for gg in range(0, 49))
            assert speedup(alpha, g, c) == pytest.approx(best, rel=1e-9)


def test_optimal_gamma_grows_as_the_draft_gets_cheaper():
    a = optimal_gamma(0.8, 0.3)
    b = optimal_gamma(0.8, 0.05)
    assert b >= a
