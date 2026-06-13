#!/usr/bin/env python3
"""Compare safetensors bf16 weights against GGUF Q*-quantized weights, per tensor.

Used to verify GGUF conversion fidelity for DFlash drafter models. Surfaces
per-tensor relative RMS error after dequantization. Uniform small errors
(~quantization noise) indicate a clean conversion; outlier tensors suggest
a converter bug (wrong shape, missing tensor, miscalibrated scale).

Usage:
    scripts/compare-dflash-weights.py <safetensors_path> <gguf_path>

Defaults to the Anbeeld/z-lab Gemma-4-31B-it-DFlash files under $MODELS
when no args given.
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np
from safetensors import safe_open

try:
    import gguf
    import gguf.quants
except ImportError:
    sys.stderr.write("gguf-py not found. Install via the convert venv:\n")
    sys.stderr.write("  source .venv-convert/bin/activate && pip install gguf\n")
    sys.exit(1)


SAFETENSORS_DEFAULT = os.path.expanduser(
    "~/ht/forks/ht-llama.cpp/models/dflash-gemma4-31b/model.safetensors"
)
GGUF_DEFAULT = os.path.expanduser(
    "~/ht/forks/ht-llama.cpp/models/dflash-gemma4-31b-gguf/gemma4-31b-it-dflash-Q6_K.gguf"
)


def bf16_bytes_to_fp32(buf: bytes) -> np.ndarray:
    """bf16 → fp32: shift mantissa left 16 bits, reinterpret. Avoids torch dep."""
    u16 = np.frombuffer(buf, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def build_name_map(num_layers: int = 5) -> dict[str, str]:
    """safetensors → GGUF tensor name mapping for DFlash drafters."""
    name_map = {
        "fc.weight": "dflash_fc.weight",
        "hidden_norm.weight": "dflash_hidden_norm.weight",
        "norm.weight": "output_norm.weight",
    }
    for i in range(num_layers):
        name_map[f"layers.{i}.input_layernorm.weight"] = f"blk.{i}.attn_norm.weight"
        name_map[f"layers.{i}.post_attention_layernorm.weight"] = f"blk.{i}.post_attention_norm.weight"
        name_map[f"layers.{i}.self_attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
        name_map[f"layers.{i}.self_attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
        name_map[f"layers.{i}.self_attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
        name_map[f"layers.{i}.self_attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"
        name_map[f"layers.{i}.self_attn.q_norm.weight"] = f"blk.{i}.attn_q_norm.weight"
        name_map[f"layers.{i}.self_attn.k_norm.weight"] = f"blk.{i}.attn_k_norm.weight"
        name_map[f"layers.{i}.mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"
        name_map[f"layers.{i}.mlp.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
        name_map[f"layers.{i}.mlp.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
    return name_map


def main() -> int:
    st_path = sys.argv[1] if len(sys.argv) > 1 else SAFETENSORS_DEFAULT
    gguf_path = sys.argv[2] if len(sys.argv) > 2 else GGUF_DEFAULT

    print(f"Loading GGUF:        {gguf_path}")
    reader = gguf.GGUFReader(gguf_path)
    gguf_tensors = {t.name: t for t in reader.tensors}
    print(f"  {len(gguf_tensors)} tensors")

    print(f"Loading safetensors: {st_path}")
    with safe_open(st_path, framework="numpy") as f:
        st_keys = sorted(f.keys())
    print(f"  {len(st_keys)} tensors")

    num_layers = sum(1 for k in st_keys if k.startswith("layers.") and k.endswith(".input_layernorm.weight"))
    name_map = build_name_map(num_layers)

    results: list[tuple[str, str, tuple[int, ...], str, float, float, float, float]] = []
    with open(st_path, "rb") as fh:
        header_size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_size).decode("utf-8"))
        data_start = 8 + header_size

        for st_name in sorted(header.keys()):
            if st_name == "__metadata__":
                continue
            meta = header[st_name]
            if meta["dtype"] != "BF16":
                continue
            gguf_name = name_map.get(st_name)
            if gguf_name is None or gguf_name not in gguf_tensors:
                continue

            offset_start, offset_end = meta["data_offsets"]
            fh.seek(data_start + offset_start)
            bf16_buf = fh.read(offset_end - offset_start)
            st_fp32 = bf16_bytes_to_fp32(bf16_buf).reshape(meta["shape"])

            gt = gguf_tensors[gguf_name]
            gt_data = gguf.quants.dequantize(gt.data, gt.tensor_type)
            gt_data = gt_data.reshape(list(gt.shape[::-1]))

            if gt_data.shape != st_fp32.shape:
                if gt_data.shape == st_fp32.shape[::-1]:
                    gt_data = gt_data.T
                else:
                    print(f"  SHAPE MISMATCH {st_name}: st={st_fp32.shape} gguf={gt_data.shape}")
                    continue

            diff = (gt_data - st_fp32).astype(np.float32)
            rms_err = float(np.sqrt((diff ** 2).mean()))
            rms_val = float(np.sqrt((st_fp32.astype(np.float32) ** 2).mean()))
            rel_err = rms_err / (rms_val + 1e-12)
            max_abs_err = float(np.abs(diff).max())
            results.append((st_name, gguf_name, st_fp32.shape, gt.tensor_type.name, rms_val, rms_err, rel_err, max_abs_err))

    results.sort(key=lambda r: -r[6])

    print()
    print("Per-tensor comparison (sorted by relative RMS error, worst first):")
    hdr = f"{'safetensors name':<55}{'GGUF type':<10}{'shape':<22}{'rms_val':>10}{'rms_err':>12}{'rel_err':>10}{'max_abs_err':>14}"
    print(hdr)
    for st_name, _, shape, qt, rms_val, rms_err, rel_err, max_abs_err in results:
        print(f"{st_name:<55}{qt:<10}{str(shape):<22}{rms_val:>10.4f}{rms_err:>12.6f}{rel_err:>10.5f}{max_abs_err:>14.6f}")

    if results:
        rel_errs = [r[6] for r in results]
        print()
        print(f"Total tensors compared: {len(results)}")
        print(f"Median relative RMS error: {np.median(rel_errs):.5f}")
        print(f"Max    relative RMS error: {max(rel_errs):.5f}  ({results[0][0]})")
        print(f"Mean   relative RMS error: {np.mean(rel_errs):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
