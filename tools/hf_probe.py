"""Probe real model size/bytes/config from the HuggingFace Hub API (no weights downloaded).

The point: every number that goes into the cost model must come from the hub, not from memory.
"""
import json
import sys
import urllib.error
import urllib.request

API = "https://huggingface.co/api/models/{}"

KEEP = (
    "model_type", "ridgeures", "num_hidden_layers", "n_layer", "hidden_size", "n_embd",
    "num_attention_heads", "n_head", "intermediate_size", "vocab_size", "max_position_embeddings",
    "context_length", "num_key_value_heads", "n_head_kv", "moe_intermediate_size", "num_experts",
    "n_experts", "n_routed_experts", "num_experts_per_tok", "n_expert_used", "head_dim",
    "hidden_act", "torch_dtype", "image_size", "patch_size", "hidden_dim", "layer_norm_eps",
)


def _get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def probe(repo, revision="main"):
    try:
        meta = _get_json(API.format(repo))
    except urllib.error.HTTPError as e:
        return {"repo": repo, "error": f"HTTP {e.code}"}

    tags = meta.get("tags", [])
    out = {
        "repo": repo,
        "downloads": meta.get("downloads"),
        "likes": meta.get("likes"),
        "gated": meta.get("gated"),
        "license": next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None),
    }

    try:
        cfg = _get_json(f"https://huggingface.co/{repo}/raw/{revision}/config.json")
        out["config"] = {k: cfg[k] for k in KEEP if k in cfg}
        qc = cfg.get("quantization_config")
        if qc:
            out["config"]["quantization_config"] = {
                k: qc.get(k) for k in ("quant_method", "bits", "group_size", "modules_to_not_convert")
            }
    except Exception as e:  # noqa: BLE001
        out["config_error"] = str(e)

    try:
        files = _get_json(f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=0")
        weights = [
            {"path": f["path"], "size_gb": round(f.get("size", 0) / 1e9, 3)}
            for f in files
            if f["path"].endswith((".safetensors", ".bin", ".gguf", ".ggml", ".pth", ".onnx"))
        ]
        out["weight_files"] = weights
        out["total_weight_gb"] = round(sum(f["size_gb"] for f in weights), 3)
    except Exception as e:  # noqa: BLE001
        out["files_error"] = str(e)

    return out


if __name__ == "__main__":
    print(json.dumps([probe(r) for r in sys.argv[1:]], indent=2))
