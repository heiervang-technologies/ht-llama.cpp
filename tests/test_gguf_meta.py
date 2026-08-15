import importlib.util
import struct
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "gguf-meta.py"
SPEC = importlib.util.spec_from_file_location("gguf_meta", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gguf_meta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gguf_meta)


def pack_string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def minimal_gemma4(*, include_tensor_data: bool = True) -> bytes:
    metadata = b"".join(
        (
            pack_string("general.architecture"),
            struct.pack("<I", gguf_meta.T_STRING),
            pack_string("gemma4"),
            pack_string("general.name"),
            struct.pack("<I", gguf_meta.T_STRING),
            pack_string("Gemma-4-it"),
        )
    )
    tensor_info = b"".join(
        (
            pack_string("token_embd.weight"),
            struct.pack("<I", 1),
            struct.pack("<Q", 1),
            struct.pack("<I", 0),
            struct.pack("<Q", 0),
        )
    )
    header = struct.pack("<IIQQ", gguf_meta.GGUF_MAGIC, 3, 1, 2)
    payload = header + metadata + tensor_info
    payload += b"\0" * (-len(payload) % 32)
    if include_tensor_data:
        payload += b"\0"
    return payload


def test_read_meta_accepts_structurally_complete_instruct_gguf(tmp_path: Path):
    path = tmp_path / "model.gguf"
    path.write_bytes(minimal_gemma4())

    metadata = gguf_meta.read_meta(path)

    assert metadata["general.architecture"] == "gemma4"
    assert metadata["_data_complete"] is True
    assert gguf_meta.is_instruct(metadata) is True


def test_read_meta_flags_missing_tensor_data(tmp_path: Path):
    path = tmp_path / "truncated.gguf"
    path.write_bytes(minimal_gemma4(include_tensor_data=False))

    assert gguf_meta.read_meta(path)["_data_complete"] is False


def test_read_meta_rejects_short_header(tmp_path: Path):
    path = tmp_path / "short.gguf"
    path.write_bytes(struct.pack("<I", gguf_meta.GGUF_MAGIC))

    with pytest.raises(EOFError, match="truncated GGUF metadata"):
        gguf_meta.read_meta(path)


def test_is_instruct_does_not_match_arbitrary_it_substring():
    assert gguf_meta.is_instruct({"general.finetune": "creative-writing"}) is False
