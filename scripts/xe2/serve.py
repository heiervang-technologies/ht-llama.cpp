#!/usr/bin/env python3
"""Launch a bounded Lunar Lake serving preset with the pinned model artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["12b", "26b"])
    parser.add_argument("--profile", choices=["baseline", "hybrid"], default="baseline")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("GGUFS"))
    parser.add_argument("--bin", type=Path, default=here.parent.parent / "build-vulkan/bin")
    parser.add_argument("--ctx-size", type=int)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    context = args.ctx_size if args.ctx_size is not None else (2048 if args.profile == "hybrid" else 8192)
    if args.models_dir is None or not 1 <= context <= 32768:
        parser.error("Set GGUFS or --models-dir; context must be between 1 and 32768")
    if args.profile == "hybrid" and context > 2048:
        parser.error("The experimental hybrid preset is limited to 2048 tokens")
    if args.mtp and args.model != "12b":
        parser.error("This preset validates MTP only for the matching 12B target")
    manifest = {m["id"]: m for m in json.loads((here / "models.json").read_text())["models"]}
    models = {key: args.models_dir / Path(m["file"]).name for key, m in manifest.items()}
    for key in [args.model] + (["12b-mtp"] if args.mtp else []):
        if not models[key].is_file():
            parser.error(f"Missing {models[key]}; run scripts/xe2/fetch-models.py first")
        with models[key].open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest[key]["sha256"]:
                parser.error(f"Model checksum mismatch: {models[key]}")
    binary = (args.bin / "llama-server").resolve()
    command = [str(binary), "-m", str(models[args.model]), "-ngl", "999", "-np", "1", "-c", str(context),
               "-b", "2048", "-ub", "512", "-t", "4", "-tb", "4", "-ctk", "f16", "-ctv", "f16",
               "-fa", "on" if args.profile == "hybrid" else "off", "--jinja",
               "--host", args.host, "--port", str(args.port)]
    if args.mtp:
        command += ["--spec-draft-model", str(models["12b-mtp"]), "--spec-type", "draft-mtp",
                    "--spec-draft-n-max", "16", "--spec-draft-p-min", "0.9", "--n-gpu-layers-draft", "999"]
    env = os.environ.copy()
    env["LLAMA_VK_GEMMA4_HYBRID_FA"] = "1" if args.profile == "hybrid" else "0"
    os.execve(binary, command, env)


if __name__ == "__main__":
    main()
