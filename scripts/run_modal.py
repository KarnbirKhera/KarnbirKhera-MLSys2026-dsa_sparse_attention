"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/mlsys26-contest/

Usage:
    # Benchmark your solution only (original behaviour)
    modal run scripts/run_modal.py

    # Benchmark your solution AND the FlashInfer baseline side-by-side
    modal run scripts/run_modal.py --compare-baseline
    (requires flashinfer_baseline.json to exist at the project root)
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

# Volume mounts at /data. Dataset lives at /data/mlsys26-contest inside it.
MOUNT_PATH      = "/data"
TRACE_SET_PATH  = "/data/mlsys26-contest"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install("build-essential", "ninja-build", "git")
    .run_commands(
        "git clone https://github.com/flashinfer-ai/flashinfer-bench.git /flashinfer-bench && cd /flashinfer-bench && pip install -v -e ."
    )
    .pip_install("torch", "triton", "numpy", "ninja")
    # Tier 3 requires sm_100a (tcgen05.* instructions are architecture-specific
    # and not available on generic sm_100). TorchBuilder invokes
    # torch.utils.cpp_extension.load() which picks arch from TORCH_CUDA_ARCH_LIST;
    # without this env, it defaults to sm_100 and ptxas rejects every tcgen05 op.
    .env({"TORCH_CUDA_ARCH_LIST": "10.0a"})
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={MOUNT_PATH: trace_volume})
def run_benchmark(solutions: list, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results for all solutions."""
    import os
    # Defensive: ensure sm_100a arch is selected at JIT build time (GT-27).
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    # All solutions must target the same definition.
    definition_name = solutions[0].definition
    for s in solutions:
        if s.definition not in trace_set.definitions:
            raise ValueError(f"Definition '{s.definition}' not found in trace set at {TRACE_SET_PATH}")
        if s.definition != definition_name:
            raise ValueError(f"All solutions must target the same definition. "
                             f"Got '{s.definition}' and '{definition_name}'.")

    definition = trace_set.definitions[definition_name]
    workloads  = trace_set.workloads.get(definition_name, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{definition_name}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: solutions},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])

    # Key results by (solution_name, workload_uuid) so the caller can compare.
    results = {}
    compile_error_logged = False
    for trace in traces:
        if trace.evaluation:
            sol_name = trace.solution if isinstance(trace.solution, str) else trace.solution.name
            wl_uuid  = trace.workload.uuid
            wl_uuid  = trace.workload.uuid
            status   = trace.evaluation.status.value

            entry = {
                "status":   status,
                "solution": trace.solution,
            }
            if status == "COMPILE_ERROR" and not compile_error_logged:
                entry["compile_log"]       = getattr(trace.evaluation, "log", None)
                entry["compile_eval_repr"] = repr(trace.evaluation)
                compile_error_logged = True
            if trace.evaluation.performance:
                entry["latency_ms"]           = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"]       = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error

            results.setdefault(sol_name, {})[wl_uuid] = entry

    raw = result_trace_set.traces.get(definition.name, [])
    for tr in raw:
        ev = tr.evaluation
        if ev and "COMPILE" in str(ev.status):
            raise RuntimeError(str(ev))

    return results


def print_results(results: dict) -> dict:
    """
    Print benchmark results for every solution and return a summary dict.

    If more than one solution is present, also print a head-to-head speedup
    comparison showing how much faster the first solution (your kernel) is
    relative to each other solution (e.g. the FlashInfer baseline).
    """
    import statistics

    summary = {}
    solution_names = list(results.keys())

    for sol_name, traces in results.items():
        total  = len(traces)
        passed = sum(1 for r in traces.values() if r.get("status") == "PASSED")
        failed = total - passed
        summary[sol_name] = {"passed": passed, "total": total, "failed": failed}

        print(f"\n{'='*60}")
        print(f"{sol_name}")
        print(f"  PASSED: {passed}/{total}   FAILED: {failed}/{total}")
        print(f"{'='*60}")

        latencies = []
        compile_log_printed = False
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                lat = result["latency_ms"]
                latencies.append(lat)
                print(f" | {lat:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup vs ref", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

            if not compile_log_printed and result.get("compile_log"):
                print(f"\n{'='*60}")
                print("COMPILE ERROR LOG:")
                print(result["compile_log"])
                print(f"{'='*60}\n")
                compile_log_printed = True
            elif not compile_log_printed and result.get("compile_eval_repr"):
                print(f"\n{'='*60}")
                print("COMPILE EVAL REPR:")
                print(result["compile_eval_repr"])
                print(f"{'='*60}\n")
                compile_log_printed = True

        if latencies:
            latencies.sort()
            print(f"\n  Latency summary across {len(latencies)} workloads:")
            print(f"    mean  = {statistics.mean(latencies):.3f} ms")
            print(f"    p50   = {latencies[len(latencies)//2]:.3f} ms")
            print(f"    p95   = {latencies[int(len(latencies)*0.95)]:.3f} ms")
            print(f"    min   = {latencies[0]:.3f} ms")
            print(f"    max   = {latencies[-1]:.3f} ms")

    # ── Head-to-head comparison ──────────────────────────────────────────────
    if len(solution_names) >= 2:
        your_name    = solution_names[0]   # first solution = your kernel
        your_traces  = results[your_name]

        for other_name in solution_names[1:]:
            other_traces = results[other_name]

            print(f"\n{'='*60}")
            print(f"HEAD-TO-HEAD: {your_name}  vs  {other_name}")
            print(f"{'='*60}")

            shared_workloads = set(your_traces.keys()) & set(other_traces.keys())
            ratios = []
            wins   = 0

            for wl_uuid in sorted(shared_workloads):
                your_r  = your_traces[wl_uuid]
                other_r = other_traces[wl_uuid]

                your_lat  = your_r.get("latency_ms")
                other_lat = other_r.get("latency_ms")

                if your_lat is None or other_lat is None:
                    print(f"  {wl_uuid[:8]}...: SKIP (missing latency)")
                    continue

                ratio = other_lat / your_lat   # > 1.0 means YOUR kernel is faster
                ratios.append(ratio)
                won = ratio > 1.0
                if won:
                    wins += 1

                print(
                    f"  {wl_uuid[:8]}...: "
                    f"yours={your_lat:.3f}ms  baseline={other_lat:.3f}ms  "
                    f"ratio={ratio:.3f}x  {'WIN' if won else 'LOSS'}"
                )

            if ratios:
                mean_ratio = sum(ratios) / len(ratios)
                win_rate   = wins / len(ratios)
                print(f"\n  Workloads compared : {len(ratios)}")
                print(f"  Wins (your kernel) : {wins}/{len(ratios)}  ({win_rate*100:.1f}%)")
                print(f"  Mean speedup ratio : {mean_ratio:.3f}x  "
                      f"({'faster' if mean_ratio > 1 else 'slower'} than baseline)")

    return summary


@app.local_entrypoint()
def main(compare_baseline: bool = False):
    """
    Pack solution and run benchmark on Modal.

    Pass --compare-baseline to also benchmark flashinfer_baseline.json
    and print a head-to-head comparison.
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    solutions_to_run = []

    print("\nLoading your solution...")
    your_solution = Solution.model_validate_json(solution_path.read_text())
    print(f"  {your_solution.name} ({your_solution.definition})")
    solutions_to_run.append(your_solution)

    if compare_baseline:
        baseline_file = PROJECT_ROOT / "flashinfer_baseline.json"
        if not baseline_file.exists():
            print(f"\nERROR: --compare-baseline requested but {baseline_file} not found.")
            print("Save the FlashInfer baseline solution JSON to flashinfer_baseline.json "
                  "at the project root and re-run.")
            raise SystemExit(1)
        print("\nLoading FlashInfer baseline solution...")
        fi_solution = Solution.model_validate_json(baseline_file.read_text())
        print(f"  {fi_solution.name} ({fi_solution.definition})")
        solutions_to_run.append(fi_solution)

    print(f"\nRunning benchmark on Modal B200 "
          f"({'with baseline comparison' if compare_baseline else 'your kernel only'})...")
    results = run_benchmark.remote(solutions_to_run)

    if not results:
        print("No results returned!")
        return

    summary = print_results(results)

    # Exit with non-zero if any solution had failures.
    any_failed = False
    for sol_name, s in summary.items():
        if s["failed"] > 0:
            print(f"\nFAIL: {s['failed']} workloads failed for {sol_name}")
            any_failed = True

    if any_failed:
        raise SystemExit(1)

    print(f"\nPASS: all workloads passed.")