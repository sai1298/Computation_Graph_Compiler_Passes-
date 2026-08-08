"""Correctness verification + latency benchmark for the FX passes.

Order of operations matters: correctness is checked BEFORE any timing is
reported. A pass that changes the model's output is a bug, and a latency number
from a broken pass is worse than no number at all.

All timing is CPU wall-clock via time.perf_counter(), warmup discarded, every
raw sample written to disk.
"""

import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
import torch
import torch.fx as fx

from model_def import build_model
from passes.constant_fold import constant_fold
from passes.fuse_conv_bn import fuse_conv_bn

TORCH_THREADS = 4
TOL = 1e-5


def hardware() -> dict:
    cpu = "unknown"
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        else:
            out = subprocess.check_output(
                ["bash", "-lc", "grep -m1 'model name' /proc/cpuinfo"], text=True)
            cpu = out.split(":", 1)[1].strip()
    except Exception:
        pass
    return {
        "cpu": cpu, "arch": platform.machine(), "logical_cores": os.cpu_count(),
        "torch_threads": TORCH_THREADS, "torch": torch.__version__,
        "python": platform.python_version(),
    }


@torch.no_grad()
def latency(model, x, runs=300, warmup=50):
    for _ in range(warmup):
        model(x)
    s = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        s.append((time.perf_counter() - t0) * 1000)
    return s


def summary(s):
    a = np.array(s)
    return {
        "n": int(a.size), "mean_ms": float(a.mean()), "std_ms": float(a.std(ddof=1)),
        "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)), "min_ms": float(a.min()),
    }


def node_counts(gm: fx.GraphModule) -> dict:
    c = {}
    for n in gm.graph.nodes:
        c[n.op] = c.get(n.op, 0) + 1
    c["total"] = sum(v for k, v in c.items() if k != "total")
    return c


def tabular(gm: fx.GraphModule) -> str:
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        gm.graph.print_tabular()
    return buf.getvalue()


@torch.no_grad()
def max_abs_diff(a_model, b_model, inputs) -> float:
    worst = 0.0
    for x in inputs:
        d = (a_model(x) - b_model(x)).abs().max().item()
        worst = max(worst, d)
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default="results/fx_results.json")
    args = p.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(TORCH_THREADS)
    os.makedirs("results", exist_ok=True)

    model = build_model()
    gm = fx.symbolic_trace(model)

    variants = {"baseline (traced)": gm}
    fused, n_fused = fuse_conv_bn(gm)
    variants["conv-bn fused"] = fused
    folded, fold_stats = constant_fold(gm)
    variants["const folded"] = folded
    both, _ = constant_fold(fused)
    variants["fused + folded"] = both

    print(f"[fx] conv-bn pairs fused: {n_fused}")
    print(f"[fx] constant subgraphs folded: {fold_stats['folded_subgraphs']} "
          f"-> values {fold_stats['values']}")

    # ---- correctness FIRST -------------------------------------------------
    check_inputs = [torch.randn(args.batch_size, 3, 32, 32) for _ in range(10)]
    print("\n[check] verifying every optimized graph against the original")
    correctness = {}
    all_ok = True
    for name, m in variants.items():
        m.eval()
        d = max_abs_diff(model, m, check_inputs)
        ok = bool(d < TOL)
        all_ok &= ok
        correctness[name] = {"max_abs_diff": d, "allclose_tol": TOL, "passed": ok}
        print(f"[check] {name:20s} max|diff| = {d:.3e}   {'PASS' if ok else 'FAIL'}")

    if not all_ok:
        raise SystemExit("[check] a pass changed the model output -- refusing to report latency")

    # ---- structure ---------------------------------------------------------
    counts = {name: node_counts(m) for name, m in variants.items()}
    print("\n[graph] node counts")
    for name, c in counts.items():
        print(f"[graph] {name:20s} total={c['total']:3d}  "
              + "  ".join(f"{k}={v}" for k, v in sorted(c.items()) if k != "total"))

    graphs = {name: tabular(m) for name, m in variants.items()}
    with open("results/graphs_before_after.txt", "w") as f:
        for name, t in graphs.items():
            f.write(f"{'=' * 78}\n{name}\n{'=' * 78}\n{t}\n\n")
    print("[graph] wrote results/graphs_before_after.txt")

    # ---- latency -----------------------------------------------------------
    x = torch.randn(args.batch_size, 3, 32, 32)
    print(f"\n[bench] batch={args.batch_size}, {args.runs} runs, {args.warmup} warmup")
    raw, lat = {}, {}
    for name, m in variants.items():
        s = latency(m, x, args.runs, args.warmup)
        raw[name] = s
        lat[name] = summary(s)
        print(f"[bench] {name:20s} p50={lat[name]['p50_ms']:7.3f}ms  "
              f"p95={lat[name]['p95_ms']:7.3f}ms  mean={lat[name]['mean_ms']:7.3f}ms")

    base = lat["baseline (traced)"]["p50_ms"]
    for name, s in lat.items():
        s["speedup_vs_baseline"] = base / s["p50_ms"]
        s["latency_reduction_pct"] = 100 * (1 - s["p50_ms"] / base)

    print("\n[bench] p50 latency reduction vs traced baseline")
    for name, s in lat.items():
        print(f"[bench] {name:20s} {s['latency_reduction_pct']:+6.2f}%  "
              f"({s['speedup_vs_baseline']:.3f}x)")

    payload = {
        "hardware": hardware(),
        "protocol": {
            "runs": args.runs, "warmup": args.warmup, "batch_size": args.batch_size,
            "timed_region": "model(x) forward only",
            "correctness_tolerance": TOL,
            "correctness_inputs": "10 random batches, max abs diff vs eager model",
        },
        "passes": {"conv_bn_pairs_fused": n_fused, "constant_fold": fold_stats},
        "correctness": correctness,
        "node_counts": counts,
        "latency": lat,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    with open("results/latency_raw.json", "w") as f:
        json.dump(raw, f)
    print(f"\n[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
