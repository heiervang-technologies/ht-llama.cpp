#!/usr/bin/env python3
"""Reproducible local Xe2 validation. Results and server logs stay out of Git."""
import argparse
import concurrent.futures
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MANIFEST = json.loads((HERE / "models.json").read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def environment(mode):
    env = os.environ.copy()
    for name in ("LLAMA_VK_GEMMA4_HYBRID_FA", "GGML_VK_DISABLE_MMVQ", "GGML_VK_FORCE_MMVQ"):
        env.pop(name, None)
    if mode == "hybrid":
        env["LLAMA_VK_GEMMA4_HYBRID_FA"] = "1"
    elif mode == "zero":
        env["LLAMA_VK_GEMMA4_HYBRID_FA"] = "0"
    return env


def snapshot(pid=None):
    data = {"time": time.time()}
    for path in (Path("/sys/class/power_supply/AC/online"), Path("/sys/firmware/acpi/platform_profile")):
        if path.exists():
            data[str(path)] = path.read_text().strip()
    data["temperatures"] = {str(p): p.read_text().strip() for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")}
    memory_info = Path("/proc/meminfo")
    if memory_info.exists():
        data["system_memory"] = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                                 for line in memory_info.read_text().splitlines()
                                 if line.startswith(("MemAvailable:", "SwapFree:"))}
    if pid:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, value = line.split(":", 1)
                    data[key] = value.strip()
            # Xe allocates GPU buffers outside the process RSS on this UMA
            # device. Keep per-client DRM accounting and deduplicate dup fds.
            clients = {}
            for path in Path(f"/proc/{pid}/fdinfo").iterdir():
                try:
                    fields = dict(line.split(":", 1) for line in path.read_text().splitlines()
                                  if line.startswith("drm-"))
                except FileNotFoundError:
                    continue
                if "drm-client-id" in fields:
                    clients[fields["drm-client-id"].strip()] = {k: v.strip() for k, v in fields.items()}
            data["drm_clients"] = clients
        except FileNotFoundError:
            pass
    return data


def run(args, command, name, env=None):
    print(name, flush=True)
    start = snapshot()
    with (args.output / f"{name}.stdout").open("w") as out, (args.output / f"{name}.stderr").open("w") as err:
        process = subprocess.Popen([str(x) for x in command], stdout=out, stderr=err, env=env)
        samples = []
        try:
            while process.poll() is None:
                samples.append(snapshot(process.pid))
                time.sleep(1)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    write_json(args.output / f"{name}.run.json", {
        "command": [str(x) for x in command], "start": start, "end": snapshot(),
        "returncode": process.returncode, "samples": samples,
        "tuning_environment": {k: v for k, v in (env or os.environ).items() if k.startswith(("LLAMA_VK_", "GGML_VK_"))},
    })
    if process.returncode:
        raise RuntimeError(f"{name} failed; inspect {args.output / (name + '.stderr')}")


def request(port, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        result = json.load(response)

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("Non-finite server output")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        if isinstance(value, list):
            for item in value:
                finite(item)
    finite(result)
    return result


@contextmanager
def server(args, model, mode, name, mtp=False, slots=1, cache="f16", context=2048):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(args.bin / "llama-server"), "-m", str(model), "-ngl", "999", "-c", str(context * slots),
               "-np", str(slots), "-b", "2048", "-ub", "512", "-t", "4", "-tb", "4",
               "-fa", "off" if mode == "off" else "on", "-ctk", cache, "-ctv", cache,
               "--cache-ram", "1024", "--host", "127.0.0.1", "--port", str(port), "--jinja"]
    if mtp:
        command += ["--spec-draft-model", str(args.models["12b-mtp"]), "--spec-type", "draft-mtp",
                    "--spec-draft-n-max", "16", "--spec-draft-p-min", "0.9", "--n-gpu-layers-draft", "999"]
    write_json(args.output / f"{name}.command.json", command)
    with (args.output / f"{name}.server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=environment(mode))
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: {name}")
                try:
                    request(port, "/health")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"Server startup timed out: {name}")
                    time.sleep(1)
            yield port, process
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def formatted_prompt(port, text):
    result = request(port, "/apply-template", {"messages": [{"role": "user", "content": text}],
                     "chat_template_kwargs": {"enable_thinking": False}})
    prompt = result.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("Missing formatted prompt")
    return prompt


def completion(port, prompt, cached=True):
    result = request(port, "/completion", {"prompt": prompt, "n_predict": 64, "temperature": 0,
                     "seed": 1234, "cache_prompt": cached, "return_tokens": True})
    if not result.get("tokens") or result.get("truncated"):
        raise RuntimeError("Empty or truncated completion")
    return result


def cancel_completion(port):
    payload = {"prompt": formatted_prompt(port, "Write a long story about an observatory."), "n_predict": 512,
               "temperature": 0, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        while True:
            line = response.readline()
            if not line:
                raise RuntimeError("Stream closed before cancellation test")
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            if event.get("content"):
                break
            if event.get("stop"):
                raise RuntimeError("Generation ended before a token could be cancelled")
    # Closing the response cancels this request. The next completion proves the
    # slot can be reused; no process restart masks lifecycle failures.
    return event


def chat_lifecycle(port):
    results = []
    for thinking in (False, True):
        result = request(port, "/v1/chat/completions", {"messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "What is two plus three?"}], "temperature": 0, "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": thinking}})
        if not result.get("choices"):
            raise RuntimeError("Missing chat result")
        content = result["choices"][0]["message"].get("content", "")
        if not re.search(r"\b(?:five|5)\b", content, re.IGNORECASE):
            raise RuntimeError("Chat failed the arithmetic answer check")
        results.append(result)
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    messages = [{"role": "user", "content": "Use get_weather to check Oslo's weather."}]
    response = request(port, "/v1/chat/completions", {"messages": messages, "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, "temperature": 0,
        "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}})
    assistant = response["choices"][0]["message"]
    calls = assistant.get("tool_calls", [])
    if not calls or calls[0]["function"]["name"] != "get_weather":
        raise RuntimeError("Expected structured weather tool call")
    arguments = json.loads(calls[0]["function"]["arguments"])
    if arguments.get("city", "").casefold() != "oslo":
        raise RuntimeError("Weather tool call used the wrong city")
    messages.append(assistant)
    for call in calls:
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "Oslo: sunny, 18 C."})
    final = request(port, "/v1/chat/completions", {"messages": messages, "tools": tools, "tool_choice": "none",
        "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}})
    if not final["choices"][0]["message"].get("content"):
        raise RuntimeError("Missing response after tool result")
    return {"thinking": results, "tool_call": response, "tool_result": final}


def parity(args):
    for key in args.model_ids:
        reference = args.output / f"{key}.chat.cpu-logits.bin"
        for backend in ("cpu", "gpu"):
            run(args, [args.bin / "test-gemma4-device", args.models[key], reference, backend],
                f"{key}-parity-{backend}", environment("off"))


def bench(args):
    for key in args.model_ids:
        for depth in args.depths:
            for repeat in range(args.repetitions):
                modes = ["off", "on", "hybrid"]
                modes = modes[repeat % 3:] + modes[:repeat % 3]
                for mode in modes:
                    name = f"{key}-c{depth}-{mode}-r{repeat}"
                    run(args, [args.bin / "llama-bench", "-m", args.models[key], "-ngl", "999",
                        "-p", "512", "-n", "64", "-pg", "512,64", "-d", str(depth - 576),
                        "-fa", "off" if mode == "off" else "on", "-ctk", "f16", "-ctv", "f16",
                        "-ub", "512", "-b", "2048", "-t", "4", "-r", "1", "-o", "json"], name, environment(mode))
                    rows = json.loads((args.output / f"{name}.stdout").read_text())
                    if len(rows) != 3 or any(not math.isfinite(row["avg_ts"]) or row["avg_ts"] <= 0 for row in rows):
                        raise RuntimeError(f"Invalid benchmark metrics: {name}")


def smoke(args):
    query = "What is the capital of Norway? Answer briefly."
    for key in args.model_ids:
        baseline = None
        hybrid_target = None
        hybrid_verification = None
        configs = [("on", "on", False, "f16"), ("zero", "zero", False, "f16"),
                   ("hybrid", "hybrid", False, "f16"), ("q8", "hybrid", False, "q8_0"),
                   ("q4", "hybrid", False, "q4_0")]
        if key == "12b":
            configs.append(("mtp", "hybrid", True, "f16"))
        for config, mode, mtp, cache in configs:
            if config not in args.smoke_configs:
                continue
            name = f"{key}-smoke-{mode}-{cache}-mtp{int(mtp)}"
            with server(args, args.models[key], mode, name, mtp=mtp, cache=cache) as (port, process):
                prompt = formatted_prompt(port, query)
                first = completion(port, prompt, False)
                second = completion(port, prompt)
                evidence = {"first": first, "cached": second, "passed": False}
                write_json(args.output / f"{name}.json", evidence)
                if "oslo" not in first.get("content", "").casefold():
                    raise RuntimeError(f"Completion failed the capital answer check: {name}")
                if first["tokens"] != second["tokens"]:
                    raise RuntimeError(f"Prefix reuse changed greedy tokens: {name}")
                if mode == "on":
                    baseline = first["tokens"]
                if mode == "zero" and first["tokens"] != baseline:
                    raise RuntimeError("Hybrid=0 differs from unset")
                if mode == "hybrid" and cache == "f16" and not mtp:
                    hybrid_target = first["tokens"]
                if mode == "hybrid" and cache == "f16":
                    verification = completion(port, formatted_prompt(port,
                        "Write a Python function called add_numbers that returns a + b. Return only the function."), False)
                    evidence["verification"] = verification
                    write_json(args.output / f"{name}.json", evidence)
                    if not mtp:
                        hybrid_verification = verification["tokens"]
                if mtp:
                    if first["tokens"] != hybrid_target or verification["tokens"] != hybrid_verification:
                        raise RuntimeError("MTP differs from target-only greedy tokens")
                    if sum(r.get("timings", {}).get("draft_n_accepted", 0)
                           for r in (first, second, verification)) <= 0:
                        raise RuntimeError("MTP smoke test accepted no draft tokens")
                chat = chat_lifecycle(port)
                cancelled_after = cancel_completion(port)
                after_cancel = completion(port, prompt)
                evidence.update(chat=chat, cancelled_after=cancelled_after, after_cancel=after_cancel, memory=snapshot(process.pid))
                write_json(args.output / f"{name}.json", evidence)
                if after_cancel["tokens"] != first["tokens"]:
                    raise RuntimeError(f"Cancellation/reuse changed greedy tokens: {name}")
                evidence["passed"] = True
                write_json(args.output / f"{name}.json", evidence)


def soak(args):
    for key in args.model_ids:
        for profile in ("baseline", "hybrid"):
            name = f"{key}-soak-{profile}"
            slots = 4 if key == "26b" and profile == "hybrid" else 1
            context = 8192 if profile == "baseline" else 2048
            mode = "off" if profile == "baseline" else "hybrid"
            with server(args, args.models[key], mode, name, mtp=key == "12b", slots=slots, context=context) as (port, process):
                started = time.monotonic()
                count = 0
                drafted = accepted = 0
                prompts = ["What is the capital of Norway? Answer briefly.", "Write a Python function that adds two integers.\n",
                           "An observatory studies stars. " * (900 if profile == "baseline" else 180) + "\nSummarize in one sentence:"]
                prompts = [formatted_prompt(port, p) for p in prompts]
                with (args.output / f"{name}.jsonl").open("w") as output:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
                        while time.monotonic() - started < args.soak_seconds / len(args.model_ids) / 2:
                            results = list(pool.map(lambda p: completion(port, p),
                                           [prompts[(count + i) % len(prompts)] for i in range(slots)]))
                            for result in results:
                                drafted += result.get("timings", {}).get("draft_n", 0)
                                accepted += result.get("timings", {}).get("draft_n_accepted", 0)
                            output.write(json.dumps({"elapsed": time.monotonic() - started, "memory": snapshot(process.pid),
                                        "results": results}, allow_nan=False) + "\n")
                            output.flush()
                            count += slots
                            if process.poll() is not None:
                                raise RuntimeError("Server died during soak")
                write_json(args.output / f"{name}.summary.json", {"requests": count, "elapsed": time.monotonic() - started,
                           "drafted": drafted, "accepted": accepted})
                if key == "12b" and (drafted == 0 or accepted == 0):
                    raise RuntimeError(f"MTP did not draft and accept tokens: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["parity", "bench", "smoke", "soak"])
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("GGUFS"))
    parser.add_argument("--bin", type=Path, default=REPO / "build-vulkan/bin")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-ids", nargs="+", choices=["12b", "26b"], default=["12b", "26b"])
    parser.add_argument("--depths", nargs="+", type=int, default=[2048, 8192, 16384, 32768])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--smoke-configs", nargs="+", choices=["on", "zero", "hybrid", "q8", "q4", "mtp"],
                        default=["on", "zero", "hybrid", "q8", "q4", "mtp"])
    parser.add_argument("--soak-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.models_dir is None or min(args.depths) < 576 or args.repetitions < 1 or args.soak_seconds < 1:
        parser.error("Set GGUFS or --models-dir; depths >=576 and counts positive")
    if args.stage == "smoke" and (("zero" in args.smoke_configs and "on" not in args.smoke_configs) or
                                  ("mtp" in args.smoke_configs and "hybrid" not in args.smoke_configs)):
        parser.error("The zero comparison requires on; the MTP comparison requires hybrid")
    args.bin = args.bin.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.models = {m["id"]: args.models_dir / Path(m["file"]).name for m in MANIFEST["models"]}
    for m in MANIFEST["models"]:
        if m["id"] not in args.model_ids and not (m["id"] == "12b-mtp" and "12b" in args.model_ids and args.stage in ("smoke", "soak")):
            continue
        with args.models[m["id"]].open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != m["sha256"]:
                raise RuntimeError(f"Model checksum mismatch: {m['id']}")
    packages = subprocess.run(["pacman", "-Q", "mesa", "vulkan-intel", "gcc"], capture_output=True, text=True).stdout if shutil.which("pacman") else ""
    write_json(args.output / f"{args.stage}-{'-'.join(args.model_ids)}.metadata.json", {
        "platform": platform.platform(), "packages": packages, "models": MANIFEST,
        "settings": {"model_ids": args.model_ids, "depths": args.depths, "repetitions": args.repetitions,
                     "soak_seconds": args.soak_seconds, "smoke_configs": args.smoke_configs, "models_dir": str(args.models_dir)},
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "diff": subprocess.check_output(["git", "diff"], cwd=REPO, text=True), "environment": snapshot(),
        "server_version": subprocess.check_output([str(args.bin / "llama-server"), "--version"], text=True, stderr=subprocess.STDOUT),
        "devices": subprocess.check_output([str(args.bin / "llama-bench"), "--list-devices"], text=True, stderr=subprocess.STDOUT)})
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
