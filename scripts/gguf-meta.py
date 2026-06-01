#!/usr/bin/env python3
# Minimal, dependency-free GGUF metadata reader.
#
# Reads only the header key/value block (strings + scalars + short arrays).
# Deliberately avoids numpy / gguf-py so it runs anywhere, including hosts
# where the project .venv has a broken numpy (missing MKL).
#
# Usage:
#   gguf-meta.py FILE [FILE ...]            # human-readable dump of key fields
#   gguf-meta.py --check-instruct FILE      # exit 0 iff FILE is a non-corrupt
#                                           #   gemma4 *instruct* target; else exit 1
#
# The --check-instruct mode is what the DFlash target-sweep bench uses to
# refuse base-fine-tune or truncated GGUFs, so the base-vs-instruct confound
# (a DFlash-it drafter benched against a base target) cannot be run by accident.

import sys
import struct

GGUF_MAGIC = 0x46554747

(T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL,
 T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64) = range(13)

SCALAR_FMT = {
    T_UINT8: ('<B', 1), T_INT8: ('<b', 1), T_UINT16: ('<H', 2), T_INT16: ('<h', 2),
    T_UINT32: ('<I', 4), T_INT32: ('<i', 4), T_FLOAT32: ('<f', 4), T_BOOL: ('<?', 1),
    T_UINT64: ('<Q', 8), T_INT64: ('<q', 8), T_FLOAT64: ('<d', 8),
}

WANT = [
    "general.architecture", "general.name", "general.basename",
    "general.finetune", "general.size_label", "general.file_type",
    "general.quantization_version", "gemma4.block_count",
]


def _rd(f, fmt, n):
    return struct.unpack(fmt, f.read(n))[0]


def _rstr(f):
    ln = _rd(f, '<Q', 8)
    return f.read(ln).decode('utf-8', 'replace')


def _rval(f, t):
    if t in SCALAR_FMT:
        fmt, n = SCALAR_FMT[t]
        return _rd(f, fmt, n)
    if t == T_STRING:
        return _rstr(f)
    if t == T_ARRAY:
        at = _rd(f, '<I', 4)
        ln = _rd(f, '<Q', 8)
        out = []
        for i in range(ln):
            out.append(_rval(f, at))
            if len(out) >= 8:  # only need a peek; skip the tail
                rem = ln - len(out)
                if at == T_STRING:
                    for _ in range(rem):
                        _rstr(f)
                elif at in SCALAR_FMT:
                    _, nb = SCALAR_FMT[at]
                    f.read(nb * rem)
                break
        return out
    raise ValueError(f"unknown gguf value type {t}")


def read_meta(path):
    """Return dict of header KV fields, plus _has_chat_template / _ok flags.

    Raises ValueError on a non-GGUF / truncated file. Also walks the tensor-info
    section to confirm the tensor DATA is actually present (catches a file whose
    header parses cleanly but whose weights were truncated — same failure class
    as the HF-xet silent shard drop). Sets _data_complete / _min_size_bytes."""
    import os
    meta = {}
    file_size = os.path.getsize(path)
    with open(path, 'rb') as f:
        magic = _rd(f, '<I', 4)
        if magic != GGUF_MAGIC:
            raise ValueError("not a GGUF file (bad magic) — truncated or wrong format")
        ver = _rd(f, '<I', 4)
        n_tensors = _rd(f, '<Q', 8)
        n_kv = _rd(f, '<Q', 8)
        if ver == 0 or n_tensors == 0 or n_kv == 0:
            raise ValueError(
                f"truncated/stub GGUF (version={ver}, n_tensors={n_tensors}, n_kv={n_kv}) "
                f"— valid magic but no payload; partial/corrupt write")
        meta["_version"] = ver
        meta["_n_tensors"] = n_tensors
        has_ct = False
        alignment = 32  # GGUF default; overridden by general.alignment if present
        for _ in range(n_kv):
            key = _rstr(f)
            t = _rd(f, '<I', 4)
            v = _rval(f, t)
            if key == "tokenizer.chat_template":
                has_ct = True
            if key == "general.alignment":
                alignment = int(v) or 32
            if key in WANT:
                meta[key] = v
        meta["_has_chat_template"] = has_ct

        # Walk the tensor-info section: name, n_dims, dims[], type, offset.
        # The max offset is a lower bound on how far the data section must extend,
        # so the file must be at least data_start + max_offset long.
        max_offset = 0
        try:
            for _ in range(n_tensors):
                _rstr(f)                       # tensor name
                ndim = _rd(f, '<I', 4)
                for _d in range(ndim):
                    _rd(f, '<Q', 8)            # dim
                _rd(f, '<I', 4)               # ggml type
                off = _rd(f, '<Q', 8)         # data offset (relative to data section)
                if off > max_offset:
                    max_offset = off
            data_start = f.tell()
            if alignment > 1 and (data_start % alignment) != 0:
                data_start += alignment - (data_start % alignment)
            min_size = data_start + max_offset + 1  # last tensor needs >=1 byte
            meta["_min_size_bytes"] = min_size
            meta["_actual_size_bytes"] = file_size
            meta["_data_complete"] = file_size >= min_size
        except Exception:
            # Could not read the full tensor table → it's truncated within the header.
            meta["_data_complete"] = False
            meta["_min_size_bytes"] = None
            meta["_actual_size_bytes"] = file_size
    return meta


def is_instruct(meta):
    name = str(meta.get("general.name", "")).lower()
    ft = str(meta.get("general.finetune", "")).lower()
    base = str(meta.get("general.basename", "")).lower()
    return ("it" in ft) or ("-it" in name) or name.endswith(" it") or \
           ("it" in base.split("-")) or meta.get("_has_chat_template", False)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2

    if args[0] == "--check-instruct":
        if len(args) != 2:
            print("usage: gguf-meta.py --check-instruct FILE", file=sys.stderr)
            return 2
        path = args[1]
        try:
            meta = read_meta(path)
        except Exception as e:
            print(f"REJECT {path}: {e}", file=sys.stderr)
            return 1
        arch = str(meta.get("general.architecture", ""))
        if arch != "gemma4":
            print(f"REJECT {path}: architecture={arch!r} (expected gemma4)", file=sys.stderr)
            return 1
        if not is_instruct(meta):
            print(f"REJECT {path}: BASE fine-tune (name={meta.get('general.name')!r}, "
                  f"finetune={meta.get('general.finetune')!r}, chat_template="
                  f"{meta.get('_has_chat_template')}). DFlash-it drafter needs an INSTRUCT target.",
                  file=sys.stderr)
            return 1
        if meta.get("_data_complete") is False:
            mn = meta.get("_min_size_bytes")
            act = meta.get("_actual_size_bytes")
            print(f"REJECT {path}: TRUNCATED — header valid but tensor data incomplete "
                  f"(file {act} bytes < min {mn} bytes implied by tensor offsets). "
                  f"Partial/corrupt write; would load garbage or crash mid-bench.",
                  file=sys.stderr)
            return 1
        print(f"OK {path}: instruct gemma4 (name={meta.get('general.name')!r}, "
              f"file_type={meta.get('general.file_type')})")
        return 0

    rc = 0
    for path in args:
        print(f"===== {path.split('/')[-1]} =====")
        try:
            meta = read_meta(path)
        except Exception as e:
            print(f"  ERROR: {e}")
            rc = 1
            continue
        for k in WANT:
            if k in meta:
                print(f"  {k:30s} = {str(meta[k])[:90]}")
        print(f"  {'has_chat_template':30s} = {meta.get('_has_chat_template')}")
        print(f"  {'is_instruct':30s} = {is_instruct(meta)}")
        print(f"  {'n_tensors':30s} = {meta.get('_n_tensors')}")
        complete = meta.get('_data_complete')
        if complete is False:
            print(f"  {'data_complete':30s} = False  (TRUNCATED: "
                  f"{meta.get('_actual_size_bytes')} < {meta.get('_min_size_bytes')} bytes)")
        else:
            print(f"  {'data_complete':30s} = {complete}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
