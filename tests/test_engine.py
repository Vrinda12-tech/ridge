"""Engine tests: the C core against an independent NumPy reference.

The references here are written from the maths, not from the C. If both agree it
is because the maths is right, not because they share code.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from ridge import engine_core as ec


# --------------------------------------------------------------- references --


def ref_delta(q, k, v, beta, g, scale):
    """Gated delta rule, the recurrence as written in docs/ENGINE.md section 3.2."""
    T, H, dk = k.shape
    dv = v.shape[-1]
    h = np.zeros((H, dk, dv), dtype=np.float64)
    out = np.zeros((T, H, dv), dtype=np.float64)
    for t in range(T):
        h = h * np.exp(g[t])[:, None, None]
        pred = np.einsum("hi,hij->hj", k[t], h)
        delta = (v[t] - pred) * beta[t][:, None]
        h = h + k[t][:, :, None] * delta[:, None, :]
        out[t] = scale * np.einsum("hi,hij->hj", q[t], h)
    return out, h


def ref_attn_full(q, kv):
    H, d = q.shape
    T = kv.shape[0]
    K = kv[:, 0]           # [T,H,d]
    V = kv[:, 1]
    scale = 1.0 / np.sqrt(d)
    out = np.zeros((H, d), dtype=np.float64)
    for h in range(H):
        logits = np.array([np.dot(q[h], K[t, h]) for t in range(T)]) * scale
        logits -= logits.max()
        w = np.exp(logits)
        w /= w.sum()
        out[h] = sum(w[t] * V[t, h] for t in range(T))
    return out


def ref_layer(x, L, kv, kv_len, state, use_delta, pos, n_heads):
    """A NumPy transformer layer matching the C one."""
    d = L["d"]
    dh = d // n_heads
    dff = L["d_ff"]
    eps = 1e-5

    def rms(x, w):
        inv = 1.0 / np.sqrt((x * x).mean() + eps)
        return x * inv * w

    def rope(vec, pos):
        v = vec.copy()
        for i in range(0, dh, 2):
            freq = 1.0 / (10000.0 ** (i / dh))
            ang = pos * freq
            c, s = np.cos(ang), np.sin(ang)
            a, b = v[i], v[i + 1]
            v[i] = a * c - b * s
            v[i + 1] = a * s + b * c
        return v

    n = rms(x, L["attn_norm"])
    q = L["wq"] @ n
    k = L["wk"] @ n
    v = L["wv"] @ n
    for h in range(n_heads):
        q[h * dh:(h + 1) * dh] = rope(q[h * dh:(h + 1) * dh], pos)
        k[h * dh:(h + 1) * dh] = rope(k[h * dh:(h + 1) * dh], pos)

    scale = 1.0 / np.sqrt(dh)
    if use_delta:
        o = np.zeros(d)
        for h in range(n_heads):
            hh = state[h]
            hh *= 1.0
            pred = k[h * dh:(h + 1) * dh] @ hh
            delta = (v[h * dh:(h + 1) * dh] - pred) * 1.0
            hh += np.outer(k[h * dh:(h + 1) * dh], delta)
            o[h * dh:(h + 1) * dh] = scale * (q[h * dh:(h + 1) * dh] @ hh)
        attn = o
    else:
        kv[kv_len, 0] = k
        kv[kv_len, 1] = v
        T = kv_len + 1
        attn = np.zeros(d)
        for h in range(n_heads):
            qh = q[h * dh:(h + 1) * dh]
            logits = np.array([np.dot(qh, kv[t, 0, h * dh:(h + 1) * dh]) for t in range(T)]) * scale
            logits -= logits.max()
            w = np.exp(logits)
            w /= w.sum()
            acc = np.zeros(dh)
            for t in range(T):
                acc += w[t] * kv[t, 1, h * dh:(h + 1) * dh]
            attn[h * dh:(h + 1) * dh] = acc

    x = x + L["wo"] @ attn
    n2 = rms(x, L["ffn_norm"])
    gate = L["w_gate"] @ n2
    up = L["w_up"] @ n2
    silu = (gate / (1.0 + np.exp(-gate))) * up
    x = x + L["w_down"] @ silu
    return x


# ------------------------------------------------------------------- state ---


def test_delta_rule_matches_reference():
    rng = np.random.default_rng(0)
    T, H, dk, dv = 7, 3, 4, 4
    q = rng.normal(size=(T, H, dk)).astype(np.float32)
    k = rng.normal(size=(T, H, dk)).astype(np.float32)
    v = rng.normal(size=(T, H, dv)).astype(np.float32)
    beta = rng.random((T, H)).astype(np.float32)
    g = -rng.random((T, H)).astype(np.float32)
    scale = 1.0 / np.sqrt(dk)

    out_c, h_c = ec.delta_run(q, k, v, beta, g, scale=scale)
    out_r, h_r = ref_delta(q.astype(np.float64), k.astype(np.float64),
                           v.astype(np.float64), beta.astype(np.float64),
                           g.astype(np.float64), scale)
    assert np.allclose(out_c, out_r, atol=1e-4), np.abs(out_c - out_r).max()
    assert np.allclose(h_c, h_r, atol=1e-4)


def test_delta_rule_is_a_rank_one_associative_memory():
    """beta=1, g=0, orthonormal keys: the state stores v exactly."""
    T, H, dk, dv = 5, 1, 5, 5
    q = np.zeros((T, H, dk), np.float32)
    k = np.zeros((T, H, dk), np.float32)
    v = np.zeros((T, H, dv), np.float32)
    for t in range(T):
        k[t, 0, t] = 1.0
        q[t, 0, t] = 1.0
        v[t, 0, t] = float(t + 1)
    beta = np.ones((T, H), np.float32)
    g = np.zeros((T, H), np.float32)
    out, h = ec.delta_run(q, k, v, beta, g, scale=1.0)
    for t in range(T):
        assert out[t, 0, t] == pytest.approx(float(t + 1), abs=1e-4)


def test_delta_state_is_constant_in_T():
    """The whole reason for the delta rule: state bytes do not depend on T."""
    b1 = ec.lib().ridge_delta_state_bytes if False else None
    # state size is a pure function of (H, dk, dv); verify via the C helper
    import ctypes as ct
    L = ec.lib()
    L.ridge_delta_state_bytes.restype = ct.c_size_t
    L.ridge_delta_state_bytes.argtypes = [ct.c_int, ct.c_int, ct.c_int]
    s = L.ridge_delta_state_bytes(8, 64, 64)
    assert s == 8 * 64 * 64 * 4


def test_full_attention_matches_reference():
    rng = np.random.default_rng(1)
    T, H, d = 6, 2, 4
    q = rng.normal(size=(H, d)).astype(np.float32)
    kv = rng.normal(size=(T, 2, H, d)).astype(np.float32)
    got = ec.attn_full(q, kv)
    want = ref_attn_full(q.astype(np.float64), kv.astype(np.float64))
    assert np.allclose(got, want, atol=1e-4)


# ------------------------------------------------------------------ memory ---


def test_memory_pool_bound_and_spill_roundtrip(tmp_path):
    blk, nlog, pool = 4096, 128, 16 * 4096
    spill = str(tmp_path / "spill.bin")
    m = ec.Memory(pool, blk, nlog, spill)
    ids = [m.alloc() for _ in range(nlog)]
    assert all(i >= 0 for i in ids)

    # write a signature into each block's first word
    for i, bid in enumerate(ids):
        p = m.touch(bid)
        ctypes.memmove(p, ctypes.byref(ctypes.c_uint32(i * 2654435761 + 7)), 4)
        m.mark_dirty(bid)

    # read them all back, in a different order
    for i in reversed(range(nlog)):
        p = m.touch(ids[i])
        val = ctypes.c_uint32()
        ctypes.memmove(ctypes.byref(val), p, 4)
        assert val.value == (i * 2654435761 + 7) % (2 ** 32)

    del m


def test_memory_never_exceeds_pool(tmp_path):
    blk, pool = 4096, 8 * 4096
    m = ec.Memory(pool, blk, 64, str(tmp_path / "s.bin"))
    peak = 0
    for _ in range(64):
        bid = m.alloc()
        m.touch(bid)
        m.mark_dirty(bid)
        peak = max(peak, m._L.ridge_mem_resident_blocks(m.ptr))
    assert peak <= pool // blk


def test_memory_clean_blocks_are_evictable_without_spill(tmp_path):
    """A clean block can be dropped with no spill file at all: nothing to write."""
    blk, pool = 4096, 4 * 4096
    m = ec.Memory(pool, blk, 32, None)   # no spill file
    ids = []
    for _ in range(32):
        bid = m.alloc()
        assert bid >= 0
        m.touch(bid)
        ids.append(bid)                  # never marked dirty
    assert m._L.ridge_mem_resident_blocks(m.ptr) <= pool // blk


def test_memory_lru_keeps_hot_block(tmp_path):
    blk, pool = 4096, 4 * 4096
    m = ec.Memory(pool, blk, 64, str(tmp_path / "s.bin"))
    hot = m.alloc()
    m.touch(hot)
    m.mark_dirty(hot)
    # touch the hot block repeatedly while allocating others; it must stay resident
    for _ in range(30):
        m.touch(hot)
        b = m.alloc()
        m.touch(b)
        m.mark_dirty(b)
        assert m.resident(hot)


# ------------------------------------------------------------------- layer ---


def _make_layer(rng, d, n_heads, dff):
    def w(shape):
        return rng.normal(scale=0.05, size=shape).astype(np.float32)
    return {
        "d": d,
        "d_ff": dff,
        "wq": w((d, d)), "wk": w((d, d)), "wv": w((d, d)), "wo": w((d, d)),
        "w_gate": w((dff, d)), "w_up": w((dff, d)), "w_down": w((d, dff)),
        "attn_norm": np.ones(d, np.float32), "ffn_norm": np.ones(d, np.float32),
    }


def test_layer_full_attention_matches_reference():
    rng = np.random.default_rng(3)
    d, n_heads, dff, max_T = 8, 2, 16, 6
    L = _make_layer(rng, d, n_heads, dff)
    clayer = ec.Layer(L["wq"], L["wk"], L["wv"], L["wo"], L["w_gate"],
                      L["w_up"], L["w_down"], L["attn_norm"], L["ffn_norm"])
    clayer.set_heads(n_heads)

    kv_c = np.zeros((max_T, 2, d), np.float32)
    kv_r = np.zeros((max_T, 2, d), np.float64)
    x_c = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, 0.7], np.float32)
    x_r = x_c.astype(np.float64).copy()
    kv_len = 0
    for pos in range(4):
        x_c, _, kv_len = clayer.forward(x_c, kv=kv_c, kv_len=kv_len,
                                        max_T=max_T, pos=pos)
        x_r = ref_layer(x_r, L, kv_r, pos, None, False, pos, n_heads)
    assert np.allclose(x_c, x_r, atol=1e-4), np.abs(x_c - x_r).max()


def test_layer_delta_state_matches_reference():
    rng = np.random.default_rng(4)
    d, n_heads, dff = 8, 2, 16
    L = _make_layer(rng, d, n_heads, dff)
    clayer = ec.Layer(L["wq"], L["wk"], L["wv"], L["wo"], L["w_gate"],
                      L["w_up"], L["w_down"], L["attn_norm"], L["ffn_norm"])
    clayer.set_heads(n_heads)
    dh = d // n_heads

    st_c = np.zeros((n_heads, dh, dh), np.float32)
    st_r = np.zeros((n_heads, dh, dh), np.float64)
    x_c = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, 0.7], np.float32)
    x_r = x_c.astype(np.float64).copy()
    for pos in range(4):
        x_c, _, _ = clayer.forward(x_c, state=st_c, use_delta=1, pos=pos)
        x_r = ref_layer(x_r, L, None, 0, st_r, True, pos, n_heads)
    assert np.allclose(x_c, x_r, atol=1e-4), np.abs(x_c - x_r).max()
    assert np.allclose(st_c, st_r, atol=1e-4)


def test_c_selftests_all_pass():
    assert ec.c_selftests() == (0, 0)
    assert ec.layer_selftest() == 0
