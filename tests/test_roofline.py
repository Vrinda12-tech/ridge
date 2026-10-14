"""Tests for the roofline cost model and the RAM budget (docs/MATH.md Sections 1, 4, 6)."""
import pytest

from ridge.model import ModelSpec, bytes_per_weight, reference_specs
from ridge.roofline import (
    Machine, decode_time, fits, intensity, max_params_in_budget, plan_tiers, tokens_per_second,
)
from ridge.state import kv_memory_bytes, state_memory_bytes


def test_bytes_per_weight():
    assert bytes_per_weight("fp16") == 2.0
    assert bytes_per_weight("q4") == 0.5
    assert bytes_per_weight("q2") == 0.25
    with pytest.raises(ValueError):
        bytes_per_weight("q3")


def test_decode_intensity_of_a_weight_is_two_over_bytes():
    """I = 2 flops / b_w bytes for a mat-vec."""
    for b, dt in ((2.0, "fp16"), (0.5, "q4"), (0.25, "q2")):
        assert intensity(2.0, b) == pytest.approx(2.0 / b)


def test_memory_bound_below_ridge_compute_bound_above():
    m = Machine()
    # fp16 weights: intensity 1.0 < ridge 3.07 -> memory bound
    t_fp16 = decode_time(weight_bytes=2e9, flops=2e9, machine=m, dtype="fp16")
    assert t_fp16 == pytest.approx(2e9 / (m.read_gbs * 1e9), rel=1e-9)
    # q2 weights, small model: intensity 8.0 > ridge, but the compute roof is the
    # measured bit-sliced rate, not the fp32 FMA peak
    t_q2 = decode_time(weight_bytes=0.25e9, flops=2e9, machine=m, dtype="q2")
    assert t_q2 == pytest.approx(2e9 / (m.compute_gflops * 1e9), rel=1e-9)


def test_quantisation_crosses_the_ridge():
    m = Machine()
    assert intensity(2.0, bytes_per_weight("fp16")) < m.ridge_flop_per_byte
    assert intensity(2.0, bytes_per_weight("q2")) > m.ridge_flop_per_byte


def test_ssd_streaming_is_two_orders_slower_than_ram():
    """The claim that kills the SSD-in-the-loop design."""
    m = Machine()
    w = 3e9
    ram = decode_time(w, 2e9, m)
    ssd = w / (m.disk_direct_gbs * 1e9)
    assert ssd > 100 * ram


def test_kv_grows_linearly_and_state_is_flat():
    n_layer, n_head, d_head = 27, 32, 72
    s1 = state_memory_bytes(n_layer, n_head, d_head)
    s2 = state_memory_bytes(n_layer, n_head, d_head)
    assert s1 == s2  # constant in sequence length
    k1 = kv_memory_bytes(n_layer, n_head, d_head, 4096)
    k2 = kv_memory_bytes(n_layer, n_head, d_head, 8192)
    assert k2 == pytest.approx(2 * k1)
    assert k2 > s1  # at long context the KV is the bigger object


def test_linear_attention_spec_kv_is_constant():
    s = reference_specs()["kimi-linear-48b-a3b"]
    assert s.kv_bytes(1024) == s.kv_bytes(65536)
    assert s.attention == "linear"


def test_full_attention_kv_scales_with_length():
    s = reference_specs()["qwen2.5-coder-7b"]
    assert s.kv_bytes(8192) == pytest.approx(2 * s.kv_bytes(4096))


def test_budget_ceiling_matches_math_doc():
    """docs/MATH.md 6: the ceiling is ~14.3B params at 4-bit, ~28.7B at 2-bit, 8 GB box."""
    m = Machine(ram_bytes=8e9)
    kv = reference_specs()["qwen2.5-coder-7b"].kv_bytes(4096)
    assert kv == pytest.approx(0.235e9, rel=0.01)
    p4 = max_params_in_budget(m, bits_per_weight=4, kv_bytes=kv, runtime_bytes=0.6e9)
    p2 = max_params_in_budget(m, bits_per_weight=2, kv_bytes=kv, runtime_bytes=0.6e9)
    assert 14e9 < p4 < 15e9
    assert 28e9 < p2 < 30e9


def test_reference_plan_does_not_fit_and_a_small_pair_does():
    """33B on SSD + 7B + 1.1B resident does not fit; a 4-bit 7B + 2-bit 1B draft does."""
    m = Machine(ram_bytes=8e9)
    specs = reference_specs()
    big = plan_tiers(specs["deepseek-coder-33b"], specs["qwen2.5-coder-7b"], m, "q4", "q4", 4096)
    assert not big["fits"]
    small = plan_tiers(specs["qwen2.5-coder-7b"], specs["tinyllama-1.1b"], m, "q4", "q2", 4096)
    assert small["fits"]


def test_fits_is_exact():
    m = Machine(ram_bytes=8e9)
    assert fits(7.0e9, 0.3e9, m, runtime_bytes=0.6e9)
    assert not fits(7.5e9, 0.3e9, m, runtime_bytes=0.6e9)


def test_tokens_per_second_reports_the_binding_roof():
    m = Machine()
    r = tokens_per_second(reference_specs()["qwen2.5-coder-1.5b"], "q4", m, seq_len=1024)
    assert r["bound"] in ("memory", "compute")
    assert r["tokens_per_s"] > 0
    assert r["streamed_from_ssd_tokens_per_s"] < r["tokens_per_s"]
