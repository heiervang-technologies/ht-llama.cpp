#!/usr/bin/env python3
"""Fetch the pinned validation artifacts; publish files only after SHA256 passes."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(model, root, workers):
    path = root / Path(model["file"]).name
    if path.exists():
        if path.stat().st_size != model["size"] or digest(path) != model["sha256"]:
            raise RuntimeError(f"Existing file does not match manifest: {path}")
        print(f"verified {model['id']}: {path}", flush=True)
        return
    # A separate partial file avoids treating interrupted downloads as models.
    partial = path.with_suffix(".download")
    url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['file']}"
    chunk_size = 32 * 1024 * 1024
    with partial.open("w+b") as stream:
        stream.truncate(model["size"])
        fd = stream.fileno()

        def chunk(start):
            end = min(model["size"], start + chunk_size) - 1
            for attempt in range(3):
                try:
                    request = urllib.request.Request(
                        f"{url}?download=true&offset={start}",
                        headers={"Range": f"bytes={start}-{end}"},
                    )
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{model['size']}":
                            raise RuntimeError("Server did not honor byte range")
                        position = start
                        while data := response.read(min(1024 * 1024, end + 1 - position)):
                            view = memoryview(data)
                            while view:
                                written = os.pwrite(fd, view, position)
                                if written <= 0:
                                    raise OSError("Short disk write")
                                position += written
                                view = view[written:]
                        if position != end + 1:
                            raise RuntimeError("Truncated download")
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for i, _ in enumerate(pool.map(chunk, range(0, model["size"], chunk_size)), 1):
                if i % 16 == 0:
                    print(f"{model['id']}: {min(i * chunk_size, model['size'])}/{model['size']} bytes", flush=True)
        os.fsync(fd)
    if digest(partial) != model["sha256"]:
        raise RuntimeError(f"SHA256 mismatch: {partial}")
    partial.rename(path)
    print(f"verified {model['id']}: {model['sha256']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("GGUFS"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.models_dir is None or args.workers < 1:
        parser.error("Set GGUFS or --models-dir; --workers must be positive")
    args.models_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(__file__).with_name("models.json").read_text())
    for model in manifest["models"]:
        fetch(model, args.models_dir, args.workers)


if __name__ == "__main__":
    main()
