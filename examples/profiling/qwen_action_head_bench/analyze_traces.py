#!/usr/bin/env python
"""Extract per-module timing from torch.profiler chrome trace JSONs.

For each head (OFT/PI/FAST), picks one rank's trace and stream-parses the
`record_function` events we tagged from bench.py:

    step_total   vlm_forward   action_head_forward   backward   optimizer_step

Streaming required: each trace is ~2 GB on 8-GPU runs; `json.load()` would need
8+ GB RAM. We use `ijson` to yield events one-at-a-time.

Output: a markdown table comparing OFT / PI / FAST and a per-head per-label
breakdown (mean / count / total time over the traced window).

Usage:
    pip install ijson      # one-time
    python examples/profiling/qwen_action_head_bench/analyze_traces.py \
        --root /Users/tc_ali/Lab/starVLA

Each head's data dir is auto-discovered under <root>:
    out_profile_8gpu/tb_trace_OFT/
    out_profile_8gpu/tb_trace_PI/
    out_profile_8gpu_FAST/tb_trace_FAST/
"""
import argparse
import os
import sys
from pathlib import Path

LABELS = ["step_total", "vlm_forward", "action_head_forward", "backward", "optimizer_step"]

# Map head -> (dir under root, tb_trace subdir name)
DEFAULT_LAYOUT = {
    "OFT":  ("out_profile_8gpu",      "tb_trace_OFT"),
    "PI":   ("out_profile_8gpu",      "tb_trace_PI"),
    "FAST": ("out_profile_8gpu_FAST", "tb_trace_FAST"),
}


def pick_trace_file(trace_dir: Path) -> Path:
    """Pick one trace file from a tb_trace_X/ dir.

    Strategy: among the most-recent mtime cluster (within 1 hour of the newest
    file), pick the smallest — same record_function pattern across ranks, smaller
    is faster to parse.
    """
    files = sorted(trace_dir.glob("*.pt.trace.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"no .pt.trace.json in {trace_dir}")
    newest_mtime = files[0].stat().st_mtime
    cluster = [f for f in files if newest_mtime - f.stat().st_mtime < 3600]
    return min(cluster, key=lambda p: p.stat().st_size)


def stream_events(path: Path):
    """Yield events from a chrome trace JSON, memory-efficient."""
    try:
        import ijson
    except ImportError:
        sys.stderr.write(
            "ERROR: ijson not installed. Run:  pip install ijson\n"
            "(needed because torch.profiler traces are ~2 GB each; json.load OOMs)\n"
        )
        raise SystemExit(2)

    with open(path, "rb") as f:
        yield from ijson.items(f, "traceEvents.item")


def aggregate_one(path: Path) -> dict:
    """Scan trace, collect durations per label.

    Returns {label: [dur_us, ...]} for events whose name matches our labels.
    record_function events appear as ph='X' (complete event with dur) OR as
    ph='B'/'E' pairs (begin/end). torch.profiler writes ph='X' for these.
    """
    size_mb = path.stat().st_size / 1e6
    print(f"  scanning {path.name} ({size_mb:.0f} MB) ...", flush=True)

    by_label = {lbl: [] for lbl in LABELS}
    count_total = 0
    label_set = set(LABELS)

    for ev in stream_events(path):
        count_total += 1
        name = ev.get("name")
        if name in label_set and ev.get("ph") == "X" and ev.get("dur") is not None:
            by_label[name].append(float(ev["dur"]))

    print(f"  scanned {count_total:,} events total")
    return by_label


def fmt_ms(us_list):
    """us list -> mean ms string."""
    if not us_list:
        return "n/a"
    mean_us = sum(us_list) / len(us_list)
    return f"{mean_us / 1000:.1f}"


def print_per_head_table(head: str, by_label: dict):
    """Print a small table for one head."""
    step_us = by_label["step_total"]
    if step_us:
        mean_step_us = sum(step_us) / len(step_us)
    else:
        mean_step_us = None

    print(f"\n### {head}")
    print(f"\n| label                | mean (ms) | count | % of step_total |")
    print(f"|----------------------|----------:|------:|----------------:|")
    for lbl in LABELS:
        vals = by_label[lbl]
        mean_ms = fmt_ms(vals)
        cnt = len(vals)
        if mean_step_us and vals:
            mean_us = sum(vals) / len(vals)
            pct = f"{mean_us / mean_step_us * 100:.1f}%"
        else:
            pct = "—"
        print(f"| {lbl:20s} | {mean_ms:>9s} | {cnt:>5d} | {pct:>15s} |")


def print_comparison(all_data: dict):
    """Cross-head markdown table."""
    heads = list(all_data.keys())
    print("\n## Cross-head comparison (mean ms over traced window)\n")
    header = "| label                | " + " | ".join(f"{h:>6s}" for h in heads) + " |"
    sep    = "|----------------------|" + "|".join(["-------:" for _ in heads]) + "|"
    print(header)
    print(sep)
    for lbl in LABELS:
        row = "| " + f"{lbl:20s}" + " | "
        cells = []
        for h in heads:
            cells.append(f"{fmt_ms(all_data[h][lbl]):>6s}")
        row += " | ".join(cells) + " |"
        print(row)

    # Also a percentage table relative to step_total
    print("\n## As % of step_total\n")
    print(header)
    print(sep)
    for lbl in LABELS:
        if lbl == "step_total":
            continue
        row = "| " + f"{lbl:20s}" + " | "
        cells = []
        for h in heads:
            step_vals = all_data[h]["step_total"]
            lbl_vals = all_data[h][lbl]
            if step_vals and lbl_vals:
                pct = sum(lbl_vals) / len(lbl_vals) / (sum(step_vals) / len(step_vals)) * 100
                cells.append(f"{pct:>5.1f}%")
            else:
                cells.append(f"{'—':>6s}")
        row += " | ".join(cells) + " |"
        print(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[3]),
                   help="repo root (default: auto-detected from script location)")
    p.add_argument("--head", action="append",
                   help="only analyze this head (can pass multiple times). Default: all.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    heads = args.head or list(DEFAULT_LAYOUT.keys())

    print(f"# Trace analysis ({root.name})\n")
    all_data = {}
    for head in heads:
        if head not in DEFAULT_LAYOUT:
            print(f"# {head}: unknown head, skipping", file=sys.stderr)
            continue
        subdir, trace_dirname = DEFAULT_LAYOUT[head]
        trace_dir = root / subdir / trace_dirname
        if not trace_dir.is_dir():
            print(f"# {head}: {trace_dir} not found, skipping", file=sys.stderr)
            continue
        try:
            trace_file = pick_trace_file(trace_dir)
        except Exception as e:
            print(f"# {head}: pick failed: {e}", file=sys.stderr)
            continue
        print(f"\n## {head} from {trace_file.relative_to(root)}")
        by_label = aggregate_one(trace_file)
        all_data[head] = by_label
        print_per_head_table(head, by_label)

    if len(all_data) >= 2:
        print_comparison(all_data)

    print("\n---")
    print("Caveats: record_function('action_head_forward') only fires if the framework")
    print("calls action_model.__call__() / forward(). PI's flow-matching head does;")
    print("OFT's L1RegressionActionHead is invoked via .predict_action() and bypasses")
    print("the hook; FAST does next-token prediction on the LLM head with no separate")
    print("action_model.forward(). So 'action_head_forward' is meaningful only for PI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
