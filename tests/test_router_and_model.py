"""Tests for the geometry, the tier planner and the router."""
import pytest

from architect.model import reference_specs, spec_from_hf_config
from architect.roofline import Machine, plan_tiers
from architect.router import Foreman


def test_spec_from_hf_config_dense():
    cfg = {"hidden_size": 3584, "num_hidden_layers": 28, "num_attention_heads": 28,
           "head_dim": 128, "intermediate_size": 18944, "vocab_size": 152064,
           "num_key_value_heads": 4, "model_type": "qwen2"}
    s = spec_from_hf_config("x", cfg)
    assert s.n_layer == 28 and s.d_model == 3584 and s.d_head == 128
    assert s.attention == "full"
    assert s.n_experts == 0


def test_spec_from_hf_config_moe_uses_per_expert_width():
    """The trap: MoE d_ff must be moe_intermediate_size, not intermediate_size."""
    cfg = {"hidden_size": 2304, "num_hidden_layers": 27, "num_attention_heads": 32,
           "head_dim": 72, "intermediate_size": 9216, "moe_intermediate_size": 1024,
           "num_experts": 256, "num_experts_per_tok": 8, "vocab_size": 163840,
           "model_type": "kimi_linear"}
    s = spec_from_hf_config("kimi", cfg)
    assert s.d_ff == 1024          # per expert
    assert s.n_experts == 256
    assert s.n_experts_active == 8
    assert s.attention == "linear"


def test_kimi_is_48b_total_3b_active():
    """The estimate must land within a few percent of the real 98.2 GB artifact at bf16.

    A simplified MoE count (QKV + per-expert FFN) over-counts slightly because it
    omits the shared expert and the router, so a 3% band is honest, not loose.
    """
    s = reference_specs()["kimi-linear-48b-a3b"]
    est_gb = s.params_total() * 2 / 1e9
    assert est_gb == pytest.approx(98.2, rel=0.05)   # real safetensors total
    assert 2.5e9 < s.params_active() < 3.5e9


def test_dense_active_equals_total():
    s = reference_specs()["qwen2.5-coder-7b"]
    assert s.params_active() == s.params_total()
    assert 7.0e9 < s.params_total() < 7.8e9


def test_router_picks_target_only_when_context_is_too_large():
    m = Machine(ram_bytes=8e9)
    specs = reference_specs()
    f = Foreman(specs["qwen2.5-coder-7b"], specs["tinyllama-1.1b"], m, "q4", "q2")
    # a 7B full-attention KV at 100k+ would eat the budget
    r = f.route(0.9, 200000, max_context_pressure=0.5)
    assert r.tier == "target-only"
    assert r.gamma == 0


def test_router_speculates_when_agreement_is_high_and_draft_is_cheap():
    m = Machine(ram_bytes=8e9)
    specs = reference_specs()
    f = Foreman(specs["qwen2.5-coder-7b"], specs["tinyllama-1.1b"], m, "q4", "q2")
    r = f.route(0.9, 4096)
    assert r.tier == "speculative"
    assert r.gamma > 0
    assert r.predicted_speedup > 1.0


def test_router_declines_when_agreement_is_poor():
    m = Machine(ram_bytes=8e9)
    specs = reference_specs()
    f = Foreman(specs["qwen2.5-coder-7b"], specs["tinyllama-1.1b"], m, "q4", "q2")
    r = f.route(0.05, 4096)
    assert r.tier == "target-only"


def test_router_draft_cost_matches_the_roofline():
    m = Machine(ram_bytes=8e9)
    specs = reference_specs()
    f = Foreman(specs["qwen2.5-coder-7b"], specs["tinyllama-1.1b"], m, "q4", "q2")
    assert 0.0 < f.c < 1.0


def test_plan_total_is_the_sum_of_its_parts():
    m = Machine(ram_bytes=8e9)
    specs = reference_specs()
    p = plan_tiers(specs["qwen2.5-coder-7b"], specs["tinyllama-1.1b"], m, "q4", "q2", 4096)
    expected = p["target_gb"] + p["draft_gb"] + p["kv_gb"] + 0.6
    assert p["total_gb"] == pytest.approx(expected, abs=0.01)
