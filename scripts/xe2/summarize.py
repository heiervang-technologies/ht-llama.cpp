#!/usr/bin/env python3
"""Summarize benchmark spread and paired hybrid differences without hiding losses."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    groups = defaultdict(dict)
    pattern = re.compile(r"(12b|26b)-c(\d+)-(off|on|hybrid)-r(\d+)\.stdout")
    for path in args.results.glob("*.stdout"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        model, context, mode, repeat = match.groups()
        for row in json.loads(path.read_text()):
            test = f"pp{row['n_prompt']}+tg{row['n_gen']}"
            groups[(model, int(context), test, mode)][int(repeat)] = float(row["avg_ts"])
    print("| Model | Context envelope | Test | Mode | Runs | Mean token/s | SD |")
    print("|---|---:|---|---|---:|---:|---:|")
    for (model, context, test, mode), runs in sorted(groups.items()):
        values = list(runs.values())
        deviation = statistics.stdev(values) if len(values) > 1 else 0
        print(f"| {model} | {context} | {test} | {mode} | {len(values)} | {statistics.mean(values):.2f} | {deviation:.2f} |")
    print("\nPaired hybrid comparisons (95% t interval for exactly five matching runs):\n")
    for (model, context, test, mode), runs in sorted(groups.items()):
        if mode != "hybrid":
            continue
        for baseline in ("off", "on"):
            other = groups.get((model, context, test, baseline), {})
            repeats = sorted(runs.keys() & other.keys())
            if len(repeats) != 5:
                print(f"- {model} {context} {test} vs {baseline}: incomplete ({len(repeats)}/5 paired runs)")
                continue
            differences = [100 * (runs[r] / other[r] - 1) for r in repeats]
            mean = statistics.mean(differences)
            interval = 2.776 * statistics.stdev(differences) / math.sqrt(5)
            print(f"- {model} {context} {test} vs {baseline}: {mean:+.2f}% [{mean-interval:+.2f}%, {mean+interval:+.2f}%]")


if __name__ == "__main__":
    main()
