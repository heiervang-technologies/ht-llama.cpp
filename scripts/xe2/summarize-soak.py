#!/usr/bin/env python3
"""Report soak completion, MTP engagement, and warm CPU/GPU memory trends."""
import argparse
import json
from pathlib import Path
import re
import statistics


def mib(value):
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)\s*(B|kB|KiB|MiB|GiB)?", value)
    if not match:
        raise ValueError(f"Unknown memory quantity: {value}")
    number, unit = match.groups()
    scale = {None: 1, "B": 1, "kB": 1024, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[unit]
    return int(number) * scale / 1024**2


def trend(points):
    if len(points) < 2:
        return "insufficient samples"
    times, values = zip(*points)
    mean_time, mean_value = statistics.mean(times), statistics.mean(values)
    denominator = sum((t - mean_time)**2 for t in times)
    slope = sum((t - mean_time)*(v - mean_value) for t, v in points) / denominator if denominator else 0
    return f"{min(values):.1f}–{max(values):.1f}; {slope * 60:+.3f}/min"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--warmup-seconds", type=float, default=300)
    args = parser.parse_args()
    print("| Model | Profile | State | Minutes | Requests | Warm RSS MiB; cycle-peak trend | Warm GPU resident GTT MiB; slope | MTP accepted/drafted | Telemetry errors |")
    print("|---|---|---|---:|---:|---|---|---:|---:|")
    complete = True
    for model in ("12b", "26b"):
        for profile in ("baseline", "hybrid"):
            path = args.results / f"{model}-soak-{profile}.jsonl"
            summary_path = path.with_suffix(".summary.json")
            samples = []
            if path.exists():
                # A live writer may have an incomplete final line.
                samples = [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
            elapsed = samples[-1]["elapsed"] if samples else 0
            requests = sum(len(sample["results"]) for sample in samples)
            drafted = sum(result.get("timings", {}).get("draft_n", 0) for sample in samples for result in sample["results"])
            accepted = sum(result.get("timings", {}).get("draft_n_accepted", 0) for sample in samples for result in sample["results"])
            done = bool(summary and summary["elapsed"] >= 900 and summary["requests"] == requests and requests > 0
                        and (model != "12b" or accepted > 0))
            complete &= done
            rss, gpu = [], []
            cycles = {}
            for index, sample in enumerate(samples):
                if sample["elapsed"] < args.warmup_seconds:
                    continue
                memory = sample["memory"]
                resident = mib(memory.get("VmRSS"))
                if resident is not None:
                    rss.append((sample["elapsed"], resident))
                    cycles.setdefault(index // 3, []).append((sample["elapsed"], resident))
                clients = memory.get("drm_clients", {})
                amounts = [mib(client["drm-resident-gtt"]) for client in clients.values() if "drm-resident-gtt" in client]
                if amounts:
                    gpu.append((sample["elapsed"], sum(amounts)))
            peaks = [(statistics.mean(t for t, _ in points), max(v for _, v in points))
                     for points in cycles.values() if len(points) == 3]
            rss_report = (f"{min(v for _, v in rss):.1f}–{max(v for _, v in rss):.1f}; peaks {trend(peaks)}"
                          if rss else "insufficient samples")
            errors = sum(bool(sample["memory"].get("read_errors")) for sample in samples)
            print(f"| {model} | {profile} | {'complete' if done else 'incomplete'} | {elapsed / 60:.2f} | {requests} | "
                  f"{rss_report} | {trend(gpu)} | {accepted}/{drafted} | {errors} |")
    print(f"\nFull four-phase hour completed: {'yes' if complete else 'no'}.")
    print(f"Memory trends exclude the first {args.warmup_seconds:g} seconds of each phase. "
          "RSS peak trends use complete three-round workload cycles to distinguish cache churn from growth. "
          "CPU RSS and DRM GTT are separate measurements and must not be added as disjoint pools. "
          "A finite soak can reveal growth; it cannot prove the absence of every leak.")


if __name__ == "__main__":
    main()
