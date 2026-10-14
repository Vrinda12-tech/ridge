"""Model geometry and the memory budget.

The geometry is read from a HuggingFace `config.json` (real numbers, see
`tools/hf_probe.py`) or given directly. The budget is exact arithmetic; see
`docs/MATH.md` Section 6.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

BYTES_PER_WEIGHT = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "q8": 1.0, "q4": 0.5, "q2": 0.25}


def bytes_per_weight(dtype: str) -> float:
    try:
        return BYTES_PER_WEIGHT[dtype]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(BYTES_PER_WEIGHT)}")


@dataclass
class ModelSpec:
    """Geometry of a decoder. `n_experts == 0` means a dense model."""

    name: str
    n_layer: int
    d_model: int
    n_head: int
    d_head: int
    d_ff: int
    vocab: int
    n_kv_head: int = 0
    n_experts: int = 0
    n_experts_active: int = 0
    attention: str = "full"  # "full" | "linear" | "recurrent"
    tie_embeddings: bool = False

    @property
    def effective_kv_head(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def n_active_experts(self) -> int:
        return self.n_experts_active or self.n_experts

    # ---- parameter counts (per-element, not per-byte) ----------------------

    def params_attention(self) -> int:
        d, h, dh = self.d_model, self.n_head, self.d_head
        if self.attention == "full":
            q = d * h * dh
            kv = 2 * d * self.effective_kv_head * dh
            o = h * dh * d
            return q + kv + o
        if self.attention == "linear":
            # QKV + per-head state projections (a,b) + output, the Kimi-Linear shape
            qkv = 3 * d * d
            ab = 2 * d * h * dh
            o = h * dh * d
            return qkv + ab + o
        if self.attention == "recurrent":
            # RWKV-7 style: one projection for the rank-1 recurrence plus output
            return 2 * d * d
        raise ValueError(f"unknown attention {self.attention!r}")

    def params_ffn(self) -> int:
        base = 3 * self.d_model * self.d_ff
        if self.n_experts:
            return base * self.n_experts
        return base

    def params_ffn_active(self) -> int:
        base = 3 * self.d_model * self.d_ff
        return base * (self.n_active_experts if self.n_experts else 1)

    def params_embedding(self) -> int:
        return self.vocab * self.d_model * (1 if self.tie_embeddings else 2)

    def params_norm(self) -> int:
        return self.n_layer * 2 * self.d_model

    def params_total(self) -> int:
        return self.n_layer * (self.params_attention() + self.params_ffn()) + self.params_embedding() + self.params_norm()

    def params_active(self) -> int:
        """Parameters touched per token: with MoE only the routed experts count."""
        return self.n_layer * (self.params_attention() + self.params_ffn_active()) + self.params_embedding() + self.params_norm()

    # ---- bytes -------------------------------------------------------------

    def weight_bytes(self, dtype: str) -> float:
        return self.params_total() * bytes_per_weight(dtype)

    def active_bytes(self, dtype: str) -> float:
        return self.params_active() * bytes_per_weight(dtype)

    def kv_bytes(self, seq_len: int, dtype: str = "fp16") -> float:
        """Attention state per sequence. Constant for linear/recurrent."""
        if self.attention != "full":
            return self.n_layer * self.n_head * self.d_head * self.d_head * bytes_per_weight("fp32")
        return self.n_layer * 2 * self.effective_kv_head * self.d_head * seq_len * bytes_per_weight(dtype)

    def flops_per_token(self) -> float:
        """Active forward flops for one token, batch 1 (matmuls: 2 flops per MAC)."""
        return 2.0 * self.params_active()


@dataclass
class Budget:
    """RAM the machine actually has, and what is reserved."""

    total_bytes: float = 8e9
    runtime_bytes: float = 0.6e9
    weights_bytes: float = 0.0  # set by the planner

    def available_for_weights(self, state_bytes: float, kv_bytes: float) -> float:
        return self.total_bytes - self.runtime_bytes - state_bytes - kv_bytes


def spec_from_hf_config(name: str, cfg: dict) -> ModelSpec:
    """Build a ModelSpec from a HuggingFace config dict (as returned by the hub API)."""
    d = cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model")
    n_layer = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    n_head = cfg.get("num_attention_heads") or cfg.get("n_head")
    if d is None or n_layer is None or n_head is None:
        raise ValueError(f"config for {name!r} is missing hidden_size/layers/heads")
    d_head = cfg.get("head_dim") or d // n_head
    n_experts = int(cfg.get("num_experts") or cfg.get("n_routed_experts") or 0)
    # For a MoE the per-expert width is the FFN width; intermediate_size is the
    # (larger) shared width and would over-count every routed expert.
    if n_experts and cfg.get("moe_intermediate_size"):
        d_ff = cfg["moe_intermediate_size"]
    else:
        d_ff = cfg.get("intermediate_size") or 4 * d
    model_type = str(cfg.get("model_type", ""))
    attn = "linear" if "linear" in model_type else "recurrent" if "rwkv" in model_type else "full"
    return ModelSpec(
        name=name,
        n_layer=int(n_layer),
        d_model=int(d),
        n_head=int(n_head),
        d_head=int(d_head),
        d_ff=int(d_ff),
        vocab=int(cfg.get("vocab_size", 32000)),
        n_kv_head=int(cfg.get("num_key_value_heads") or cfg.get("n_head_kv") or 0),
        n_experts=n_experts,
        n_experts_active=int(cfg.get("num_experts_per_tok") or cfg.get("n_expert_used") or 0),
        attention=attn,
    )


# Geometry for the models referenced in the brief, taken from the hub (results/hf_models.json).
def reference_specs() -> dict[str, ModelSpec]:
    return {
        "qwen2.5-coder-7b": ModelSpec(
            name="Qwen/Qwen2.5-Coder-7B-Instruct", n_layer=28, d_model=3584, n_head=28, d_head=128,
            d_ff=18944, vocab=152064, n_kv_head=4, attention="full"),
        "qwen2.5-coder-1.5b": ModelSpec(
            name="Qwen/Qwen2.5-Coder-1.5B-Instruct", n_layer=28, d_model=1536, n_head=12, d_head=128,
            d_ff=8960, vocab=151936, n_kv_head=2, attention="full"),
        "tinyllama-1.1b": ModelSpec(
            name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", n_layer=22, d_model=2048, n_head=32, d_head=64,
            d_ff=5632, vocab=32000, n_kv_head=4, attention="full"),
        "deepseek-coder-33b": ModelSpec(
            name="deepseek-ai/deepseek-coder-33b-instruct", n_layer=62, d_model=7168, n_head=56,
            d_head=128, d_ff=19200, vocab=32256, n_kv_head=8, attention="full"),
        "kimi-linear-48b-a3b": ModelSpec(
            name="moonshotai/Kimi-Linear-48B-A3B-Instruct", n_layer=27, d_model=2304, n_head=32,
            d_head=72, d_ff=1024, vocab=163840, n_experts=256, n_experts_active=8,
            attention="linear"),
        "qwen3-32b": ModelSpec(
            name="Qwen/Qwen3-32B", n_layer=64, d_model=5120, n_head=64, d_head=128,
            d_ff=25600, vocab=151936, n_kv_head=8, attention="full"),
        "bitnet-b1.58-2b": ModelSpec(
            name="microsoft/bitnet-b1.58-2B-4T", n_layer=30, d_model=2560, n_head=20, d_head=128,
            d_ff=6912, vocab=128256, n_kv_head=5, attention="full"),
        "ternary-bonsai-27b": ModelSpec(
            name="prism-ml/Ternary-Bonsai-2-27B", n_layer=64, d_model=5120, n_head=40, d_head=128,
            d_ff=13824, vocab=152064, n_kv_head=8, attention="full"),
        "rwkv7-1.5b": ModelSpec(
            name="BlinkDL/rwkv7-g1k-1.5b", n_layer=24, d_model=2048, n_head=32, d_head=64,
            d_ff=8192, vocab=65536, attention="recurrent"),
    }


def spec_as_dict(s: ModelSpec) -> dict:
    d = asdict(s)
    d["params_total_b"] = round(s.params_total() / 1e9, 3)
    d["params_active_b"] = round(s.params_active() / 1e9, 3)
    d["kv_bytes_8k"] = s.kv_bytes(8192)
    return d
