"""ctypes binding for the C engine core.

This is a thin layer on purpose: it loads the compiled shared object, declares
the signatures, and hands NumPy arrays straight in as pointers. No copies, no
wrapper types. If the library is not built it raises with the build command.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine"
LIB_PATH = ENGINE / "librridge_engine.so"

_SOURCES = [
    ENGINE / "ridge_mem.c",
    ENGINE / "ridge_state.c",
    ENGINE / "ridge_quant.c",
    ENGINE / "ridge_layer.c",
]

_NONE = -1


def build(force: bool = False) -> Path:
    """Compile the engine into a shared object. Idempotent."""
    newest_src = max((s.stat().st_mtime for s in _SOURCES if s.exists()), default=0)
    if LIB_PATH.exists() and not force and LIB_PATH.stat().st_mtime >= newest_src:
        return LIB_PATH
    cmd = [
        "gcc", "-O3", "-march=native", "-fPIC", "-shared",
        "-I", str(ENGINE),
        "-o", str(LIB_PATH),
        *[str(s) for s in _SOURCES],
        "-lm",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return LIB_PATH


class _MemStats(ctypes.Structure):
    _fields_ = [
        ("allocs", ctypes.c_size_t),
        ("frees", ctypes.c_size_t),
        ("evictions", ctypes.c_size_t),
        ("spills", ctypes.c_size_t),
        ("recalls", ctypes.c_size_t),
        ("touches", ctypes.c_size_t),
        ("spill_writes", ctypes.c_size_t),
        ("spill_reads", ctypes.c_size_t),
        ("resident_bytes", ctypes.c_size_t),
        ("peak_resident_bytes", ctypes.c_size_t),
        ("logical_in_use", ctypes.c_size_t),
    ]


_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(build()))
        _declare(_lib)
    return _lib


def _declare(L: ctypes.CDLL) -> None:
    f = ctypes.c_float
    i32 = ctypes.c_int32
    sz = ctypes.c_size_t
    P = ctypes.POINTER

    L.ridge_mem_new.restype = ctypes.c_void_p
    L.ridge_mem_new.argtypes = [sz, sz, i32, ctypes.c_char_p]
    L.ridge_mem_free.argtypes = [ctypes.c_void_p]
    L.ridge_mem_alloc.restype = i32
    L.ridge_mem_alloc.argtypes = [ctypes.c_void_p]
    L.ridge_mem_release.argtypes = [ctypes.c_void_p, i32]
    L.ridge_mem_touch.restype = ctypes.c_void_p
    L.ridge_mem_touch.argtypes = [ctypes.c_void_p, i32]
    L.ridge_mem_mark_dirty.argtypes = [ctypes.c_void_p, i32]
    L.ridge_mem_is_resident.restype = ctypes.c_int
    L.ridge_mem_is_resident.argtypes = [ctypes.c_void_p, i32]
    L.ridge_mem_spill.argtypes = [ctypes.c_void_p, i32]
    L.ridge_mem_advise.argtypes = [ctypes.c_void_p, ctypes.c_void_p, sz, ctypes.c_int]
    L.ridge_mem_selftest.restype = ctypes.c_int

    L.ridge_delta_step.argtypes = [
        P(f), P(f), P(f), P(f), P(f), P(f), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, f, P(f),
    ]
    L.ridge_delta_run.argtypes = [
        P(f), P(f), P(f), P(f), P(f), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, f, P(f), P(f),
    ]
    L.ridge_attn_full.argtypes = [P(f), P(f), ctypes.c_int, ctypes.c_int, ctypes.c_int, P(f)]
    L.ridge_state_selftest.restype = ctypes.c_int

    L.ridge_layer_fwd.restype = ctypes.c_int
    L.ridge_layer_fwd.argtypes = [
        P(_Layer), P(f), P(f), P(ctypes.c_int),
        ctypes.c_int, P(f), ctypes.c_int, ctypes.c_int,
    ]
    L.ridge_layer_scratch_bytes.restype = ctypes.c_size_t
    L.ridge_layer_scratch_bytes.argtypes = [P(_Layer)]
    L.ridge_layer_selftest.restype = ctypes.c_int
    L.ridge_layer_quantize.restype = ctypes.c_int
    L.ridge_layer_quantize.argtypes = [P(_Layer)]
    L.ridge_layer_dequantize.argtypes = [P(_Layer)]

    L.ridge_quant_selftest.restype = ctypes.c_int
    L.ridge_quant_has_vnni.restype = ctypes.c_int


class _QMat(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.POINTER(ctypes.c_int8)),
        ("rowsum", ctypes.POINTER(ctypes.c_int)),
        ("scale", ctypes.POINTER(ctypes.c_float)),
        ("n", ctypes.c_int),
        ("k", ctypes.c_int),
        ("k_pad", ctypes.c_int),
    ]


class _Layer(ctypes.Structure):
    _fields_ = [
        ("d", ctypes.c_int),
        ("n_heads", ctypes.c_int),
        ("d_head", ctypes.c_int),
        ("d_ff", ctypes.c_int),
        ("wq", ctypes.POINTER(ctypes.c_float)),
        ("wk", ctypes.POINTER(ctypes.c_float)),
        ("wv", ctypes.POINTER(ctypes.c_float)),
        ("wo", ctypes.POINTER(ctypes.c_float)),
        ("w_gate", ctypes.POINTER(ctypes.c_float)),
        ("w_up", ctypes.POINTER(ctypes.c_float)),
        ("w_down", ctypes.POINTER(ctypes.c_float)),
        ("attn_norm", ctypes.POINTER(ctypes.c_float)),
        ("ffn_norm", ctypes.POINTER(ctypes.c_float)),
        ("rope_theta", ctypes.c_float),
        ("q_wq", _QMat), ("q_wk", _QMat), ("q_wv", _QMat), ("q_wo", _QMat),
        ("q_gate", _QMat), ("q_up", _QMat), ("q_down", _QMat),
        ("quantized", ctypes.c_int),
    ]


_FP = ctypes.POINTER(ctypes.c_float)


def _fp(a: np.ndarray):
    return a.ctypes.data_as(_FP)


class Layer:
    """A C transformer layer. Weights are kept alive by this object."""

    def __init__(self, wq, wk, wv, wo, w_gate, w_up, w_down, attn_norm,
                 ffn_norm, rope_theta=10000.0):
        self.wq = np.ascontiguousarray(wq, dtype=np.float32)
        self.wk = np.ascontiguousarray(wk, dtype=np.float32)
        self.wv = np.ascontiguousarray(wv, dtype=np.float32)
        self.wo = np.ascontiguousarray(wo, dtype=np.float32)
        self.w_gate = np.ascontiguousarray(w_gate, dtype=np.float32)
        self.w_up = np.ascontiguousarray(w_up, dtype=np.float32)
        self.w_down = np.ascontiguousarray(w_down, dtype=np.float32)
        self.attn_norm = np.ascontiguousarray(attn_norm, dtype=np.float32)
        self.ffn_norm = np.ascontiguousarray(ffn_norm, dtype=np.float32)
        self.d = int(self.attn_norm.shape[0])
        self.d_ff = int(self.w_gate.shape[0])
        self.rope_theta = float(rope_theta)
        self.n_heads = 1
        self.d_head = self.d
        self._s = None

    def set_heads(self, n_heads: int) -> None:
        if self.d % n_heads:
            raise ValueError("d must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = self.d // n_heads
        self._s = None   # geometry changed; rebuild the struct on next use

    def _struct(self):
        # Keep one struct alive for the object's lifetime: ridge_layer_quantize
        # writes the quantised matrices into it, so a fresh struct per call would
        # throw the quantisation away and leak.
        if self._s is None:
            self._s = _Layer(
                self.d, self.n_heads, self.d_head, self.d_ff,
                _fp(self.wq), _fp(self.wk), _fp(self.wv), _fp(self.wo),
                _fp(self.w_gate), _fp(self.w_up), _fp(self.w_down),
                _fp(self.attn_norm), _fp(self.ffn_norm),
                ctypes.c_float(self.rope_theta),
            )
        return self._s

    def quantize(self) -> None:
        """Quantise the weights to int8; the forward pass then uses the VNNI GEMV."""
        rc = lib().ridge_layer_quantize(ctypes.byref(self._struct()))
        if rc != 0:
            raise RuntimeError(f"ridge_layer_quantize rc={rc}")

    def dequantize(self) -> None:
        if self._s is not None:
            lib().ridge_layer_dequantize(ctypes.byref(self._s))

    def scratch_bytes(self) -> int:
        s = self._struct()
        return int(lib().ridge_layer_scratch_bytes(ctypes.byref(s)))

    def forward(self, x, kv=None, kv_len=0, max_T=0, state=None, use_delta=0, pos=0):
        """Run one token. Returns (x_updated, kv, kv_len)."""
        L = lib()
        s = self._struct()
        x = np.ascontiguousarray(x, dtype=np.float32).copy()
        kv_p = _fp(kv) if kv is not None else None
        st_p = _fp(state) if state is not None else None
        kvlen = ctypes.c_int(int(kv_len))
        rc = L.ridge_layer_fwd(
            ctypes.byref(s), _fp(x), kv_p, ctypes.byref(kvlen),
            int(max_T), st_p, int(use_delta), int(pos),
        )
        if rc != 0:
            raise RuntimeError(f"ridge_layer_fwd rc={rc}")
        return x, kv, kvlen.value


def layer_selftest() -> int:
    return lib().ridge_layer_selftest()


class Memory:
    """A ridge_mem* with Python lifetime management."""

    def __init__(self, pool_bytes: int, block_bytes: int, max_logical: int,
                 spill_path: str | None = None):
        L = lib()
        p = L.ridge_mem_new(pool_bytes, block_bytes, max_logical,
                            spill_path.encode() if spill_path else None)
        if not p:
            raise RuntimeError("ridge_mem_new failed")
        self._L = L
        self._p = ctypes.c_void_p(p)

    @property
    def ptr(self) -> ctypes.c_void_p:
        return self._p

    def alloc(self) -> int:
        return self._L.ridge_mem_alloc(self._p)

    def release(self, bid: int) -> None:
        self._L.ridge_mem_release(self._p, bid)

    def touch(self, bid: int) -> ctypes.c_void_p:
        return self._L.ridge_mem_touch(self._p, bid)

    def mark_dirty(self, bid: int) -> None:
        self._L.ridge_mem_mark_dirty(self._p, bid)

    def resident(self, bid: int) -> bool:
        return bool(self._L.ridge_mem_is_resident(self._p, bid))

    def spill(self, bid: int) -> int:
        return self._L.ridge_mem_spill(self._p, bid)

    def __del__(self):
        if getattr(self, "_p", None):
            try:
                self._L.ridge_mem_free(self._p)
            except Exception:
                pass
            self._p = None


def delta_run(q, k, v, beta, g, scale=None, h=None):
    """Run the gated delta rule. Arrays are [T,H,dk]/[T,H,dv]/[T,H]/[T,H].

    Returns (out[T,H,dv], final_state[H,dk,dv])."""
    L = lib()
    q = np.ascontiguousarray(q, dtype=np.float32)
    k = np.ascontiguousarray(k, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    beta = np.ascontiguousarray(beta, dtype=np.float32)
    g = np.ascontiguousarray(g, dtype=np.float32)
    T, H, dk = k.shape
    dv = v.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(dk)
    out = np.zeros((T, H, dv), dtype=np.float32)
    if h is None:
        h = np.zeros((H, dk, dv), dtype=np.float32)
    h = np.ascontiguousarray(h, dtype=np.float32)
    rc = L.ridge_delta_run(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        g.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        T, H, dk, dv, ctypes.c_float(scale),
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"ridge_delta_run rc={rc}")
    return out, h


def attn_full(q, kv):
    """Full attention. q is [H,d]; kv is [T,2,H,d]. Returns [H,d]."""
    L = lib()
    q = np.ascontiguousarray(q, dtype=np.float32)
    kv = np.ascontiguousarray(kv, dtype=np.float32)
    H, d = q.shape
    T = kv.shape[0]
    out = np.zeros((H, d), dtype=np.float32)
    rc = L.ridge_attn_full(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        kv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        T, H, d,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"ridge_attn_full rc={rc}")
    return out


def c_selftests() -> tuple[int, int]:
    L = lib()
    return L.ridge_mem_selftest(), L.ridge_state_selftest()
