"""Tests for the belief state and the Hankel/Koopman readout (docs/MATH.md Section 5)."""
import numpy as np
import pytest

from architect.jepa import (
    dmd, hankel, hankel_rank_gap, make_rank_r_series, participation_ratio, train_latent, vicreg_terms,
)
from architect.state import Reservoir, kalman, steady_state_gain


def test_hankel_shape_and_rank_of_a_pure_exponential():
    """A signal that is a sum of p complex exponentials has Hankel rank p (MATH.md 5.1).

    Slower-decaying modes than the sampling can resolve keep the rank exact.
    """
    T, p = 400, 3
    t = np.arange(T)
    z = np.array([0.999 * np.exp(1j * 0.10), 0.999 * np.exp(-1j * 0.17), 0.999 * np.exp(1j * 0.26)])
    x = np.zeros((T, 1))
    for zj in z:
        x[:, 0] += (zj ** t).real
    s = np.linalg.svd(hankel(x, K=20), compute_uv=False)
    s = s / s[0]
    # 3 complex modes -> rank 6
    assert s[5] > 1e-6
    assert s[6] < 1e-8


def test_hankel_rank_recovers_two_frequencies_as_rank_four():
    """2 real sinusoids -> rank 4, the number the homeo derivation predicts."""
    X = make_rank_r_series(T=3000, m=8, freqs=(0.20, 0.35), noise=0.0)
    r, s = hankel_rank_gap(X, K=60)
    assert r == 4


def test_dmd_recovers_the_frequencies_with_unit_modulus():
    freqs = (0.20, 0.35)  # below Nyquist, so the angle is unambiguous
    X = make_rank_r_series(T=3000, m=8, freqs=freqs, noise=0.0)
    out = dmd(X, K=60, r=4, dt=1.0)
    want = sorted(2 * np.pi * f for f in freqs)
    got = np.sort(out["omega"])[:4]
    for w in want:
        assert np.min(np.abs(got - w)) < 0.02
    assert np.allclose(out["abs_lambda"], 1.0, atol=1e-3)


def test_participation_ratio_bounds():
    assert participation_ratio(np.eye(4)) == pytest.approx(4.0)
    assert participation_ratio(np.outer([1.0, 0, 0, 0], [1.0, 0, 0, 0])) == pytest.approx(1.0)


def test_vicreg_terms_cannot_see_topology():
    """A ring and isotropic noise with the same covariance give the same VICReg terms.

    This is the blindness claim (homeo Section 6.2): the terms depend only on the
    covariance, and the covariance does not know about the hole.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 2000)
    ring = np.c_[np.cos(t), np.sin(t)]
    noise = rng.standard_normal((2000, 2))
    noise *= ring.std(0) / noise.std(0)
    v1, c1 = vicreg_terms(ring)
    v2, c2 = vicreg_terms(noise)
    assert v1 == pytest.approx(v2, abs=0.05)
    assert c1 == pytest.approx(c2, abs=0.05)


def test_prediction_only_training_collapses():
    """Mode 'none' must drive the latent energy toward 0 (docs/MATH.md 5.2).

    Without a regularizer the encoder is free to map everything to zero: the
    prediction loss is then trivially zero and the latent carries no information.
    """
    X = make_rank_r_series(T=2000, m=8, freqs=(0.20, 0.35), noise=0.05, seed=1)
    _, _, _, rep = train_latent(X, d_lat=8, mode="none", steps=1500, seed=0)
    assert rep.latent_energy < 0.05
    assert rep.next_step_r2 > 0.9  # trivially predictable, because nothing moves


def test_dimension_floor_keeps_the_latent_alive():
    """The koopman_dim mode must keep a latent with real energy and several dimensions."""
    X = make_rank_r_series(T=2000, m=8, freqs=(0.20, 0.35), noise=0.05, seed=2)
    _, _, _, rep = train_latent(X, d_lat=8, mode="koopman_dim", r_star=4, steps=1500, seed=0)
    assert rep.latent_energy > 0.3
    assert rep.participation_ratio > 2.0
    assert rep.hankel_rank >= 3


def test_regularized_latent_keeps_more_energy_than_unregularized():
    """The control: the same data and seeds, regularizer on versus off."""
    X = make_rank_r_series(T=2000, m=8, freqs=(0.20, 0.35), noise=0.05, seed=3)
    _, _, _, none_rep = train_latent(X, d_lat=8, mode="none", steps=1200, seed=0)
    _, _, _, reg_rep = train_latent(X, d_lat=8, mode="koopman_dim", r_star=4, steps=1200, seed=0)
    assert reg_rep.latent_energy > 10 * max(none_rep.latent_energy, 1e-9)
    assert reg_rep.participation_ratio > none_rep.participation_ratio


def test_kalman_denoises_a_linear_gaussian_system():
    """The filter must beat the raw observation in RMSE."""
    rng = np.random.default_rng(0)
    A = np.array([[0.98, 0.1], [-0.1, 0.98]])
    C = np.eye(2)
    Q = 1e-4 * np.eye(2)
    R = 0.09 * np.eye(2)
    x = np.zeros(2)
    xs, ys = [], []
    for _ in range(1500):
        x = A @ x + rng.normal(0, np.sqrt(Q[0, 0]), 2)
        xs.append(x.copy())
        ys.append(x + rng.normal(0, np.sqrt(R[0, 0]), 2))
    xs, ys = np.array(xs), np.array(ys)
    filt = kalman(ys, A, C, Q, R)
    rmse_raw = np.sqrt(((ys - xs) ** 2).mean())
    rmse_filt = np.sqrt(((filt - xs) ** 2).mean())
    assert rmse_filt < rmse_raw


def test_steady_state_gain_matches_the_recursion():
    A = np.array([[0.95, 0.2], [-0.1, 0.9]])
    C = np.eye(2)
    Q = 1e-3 * np.eye(2)
    R = 0.05 * np.eye(2)
    G, P = steady_state_gain(A, C, Q, R)
    # P must satisfy the discrete algebraic Riccati equation
    lhs = P
    rhs = A @ P @ A.T + Q - A @ P @ C.T @ np.linalg.inv(C @ P @ C.T + R) @ C @ P @ A.T
    assert np.allclose(lhs, rhs, atol=1e-8)


def test_reservoir_state_memory_is_flat_in_sequence_length():
    r = Reservoir(n_inputs=4, n_units=128)
    m1 = r.memory_bytes()
    r.run(np.zeros((10, 4)))
    r.run(np.zeros((1000, 4)))
    assert r.memory_bytes() == m1


def test_reservoir_has_fading_memory():
    """Two different initial states converge, the echo state property."""
    r = Reservoir(n_inputs=2, n_units=64, spectral_radius=0.7, seed=0)
    U = np.zeros((400, 2))
    s1 = r.run(U)[-1]
    r2 = Reservoir(n_inputs=2, n_units=64, spectral_radius=0.7, seed=0)
    st = np.full(64, 5.0, dtype=np.float32)
    # drive with a different start by running a spike first
    r2.run(np.ones((5, 2)) * 3.0)
    s2 = r2.run(U)[-1]
    assert np.linalg.norm(s1 - s2) < 1.0
