"""
Verify submission using the OFFICIAL EVALUATION.md Docker image:
  flashinfer/flashinfer-ci-cu132:20260401-2c675fb

Mirrors CUDA 13.2 / PyTorch 2.12 / flashinfer-bench main — the exact
environment contest judges use. A PASS here means the kernel survives
the version upgrade from the CUDA 12.8 image used by run_modal.py.

Usage (after `python scripts/pack_solution.py`):
    modal run scripts/run_official_verify.py
"""

import math
import sys
from pathlib import Path
import modal

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Solution

app = modal.App("flashinfer-bench-official-verify")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

MOUNT_PATH     = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"

# Exact image from EVALUATION.md + the pipeline's runtime setup step.
#
# EVALUATION.md claims the image has flashinfer_bench baked in, but per the
# contest admin on Discord (2026-04-24), it does NOT. The judges install
# flashinfer-bench and cupti-python at pipeline setup time. This replicates
# that setup step so we match their actual runtime.
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
    .run_commands(
        "pip install -q git+https://github.com/flashinfer-ai/flashinfer-bench.git cupti-python"
    )
    .env({"TORCH_CUDA_ARCH_LIST": "10.0a"})
)


BASELINE_SOLUTION_NAME = "flashinfer_wrapper_5af199"


def _trace_solution_name(trace) -> str:
    sol = getattr(trace, "solution", None)
    if sol is None:
        return ""
    return getattr(sol, "name", None) or (sol if isinstance(sol, str) else "")


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={MOUNT_PATH: trace_volume},
)
def run_official_benchmark(solution_json: str) -> dict:
    import os
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

    solution = Solution.model_validate_json(solution_json)
    config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition_name = solution.definition
    if definition_name not in trace_set.definitions:
        raise RuntimeError(
            f"Definition '{definition_name}' not in trace set at {TRACE_SET_PATH}"
        )

    definition = trace_set.definitions[definition_name]
    workloads  = trace_set.workloads.get(definition_name, [])
    if not workloads:
        raise RuntimeError(f"No workloads for definition '{definition_name}'")

    baseline_solution = next(
        (s for s in trace_set.solutions.get(definition_name, [])
         if s.name == BASELINE_SOLUTION_NAME),
        None,
    )
    if baseline_solution is None:
        raise RuntimeError(
            f"Baseline '{BASELINE_SOLUTION_NAME}' not found under {TRACE_SET_PATH} "
            f"for definition '{definition_name}'"
        )

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution, baseline_solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    results = {}
    error_logged = False
    traces = result_trace_set.traces.get(definition.name, [])
    for trace in traces:
        if not trace.evaluation:
            continue
        wuuid     = trace.workload.uuid
        sol_name  = _trace_solution_name(trace)
        side      = "baseline" if sol_name == BASELINE_SOLUTION_NAME else "ours"
        status    = trace.evaluation.status.value
        entry     = {"status": status, "solution_name": sol_name}
        if trace.evaluation.performance:
            entry["latency_ms"]           = trace.evaluation.performance.latency_ms
            entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
            entry["speedup_factor"]       = trace.evaluation.performance.speedup_factor
        if trace.evaluation.correctness:
            entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
        if status in ("COMPILE_ERROR", "RUNTIME_ERROR") and side == "ours" and not error_logged:
            entry["error_log"]  = getattr(trace.evaluation, "log", None)
            entry["error_repr"] = repr(trace.evaluation)
            error_logged = True
        results.setdefault(wuuid, {})[side] = entry

    return results


@app.local_entrypoint()
def main():
    import statistics

    solution_path = PROJECT_ROOT / "solution.json"
    if not solution_path.exists():
        print(f"ERROR: {solution_path} does not exist.")
        print("Run `python scripts/pack_solution.py` first.")
        raise SystemExit(1)

    solution_json = solution_path.read_text()
    solution = Solution.model_validate_json(solution_json)
    print(f"Loaded solution: {solution.name} ({solution.definition})")
    print(f"  entry_point : {solution.spec.entry_point}")
    print(f"  binding     : {solution.spec.binding}")

    print("\nRunning in OFFICIAL image: flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
    print("(first run pulls ~3-4 GB — subsequent runs reuse Modal's cache)\n")

    results = run_official_benchmark.remote(solution_json)

    total  = len(results)
    passed = sum(1 for r in results.values() if r.get("ours", {}).get("status") == "PASSED")
    failed = total - passed

    print(f"\n{'='*78}")
    print(f"OFFICIAL IMAGE RESULT — PASSED: {passed}/{total}   FAILED: {failed}/{total}")
    print(f"  baseline: {BASELINE_SOLUTION_NAME} (FlashInfer trtllm wrapper)")
    print(f"{'='*78}")

    error_printed = False
    speedups_vs_baseline = []
    for uuid, r in results.items():
        ours      = r.get("ours", {})
        baseline  = r.get("baseline", {})
        status    = ours.get("status", "MISSING")
        our_lat   = ours.get("latency_ms")
        base_lat  = baseline.get("latency_ms")
        speedup   = (base_lat / our_lat) if (our_lat and base_lat) else None

        line = f"  {uuid[:8]}...: {status}"
        if our_lat is not None:
            line += f" | ours={our_lat:.3f} ms"
        if base_lat is not None:
            line += f" | flashinfer={base_lat:.3f} ms"
        if speedup is not None:
            line += f" | {speedup:.2f}x vs flashinfer"
            speedups_vs_baseline.append(speedup)
        if ours.get("max_abs_error") is not None:
            line += f" | abs_err={ours['max_abs_error']:.2e}"
        print(line)

        if status != "PASSED" and not error_printed and ours.get("error_log"):
            print(f"\n    FIRST ERROR LOG ({status}):")
            print(f"    {ours['error_log']}\n")
            error_printed = True

    lat = sorted(r["ours"]["latency_ms"] for r in results.values()
                 if r.get("ours", {}).get("latency_ms") is not None)
    if lat:
        print(f"\nOur latency across {len(lat)} workloads:")
        print(f"  mean = {statistics.mean(lat):.3f} ms")
        print(f"  p50  = {lat[len(lat)//2]:.3f} ms")
        print(f"  p95  = {lat[int(len(lat)*0.95)]:.3f} ms")
        print(f"  min  = {lat[0]:.3f} ms")
        print(f"  max  = {lat[-1]:.3f} ms")

    if speedups_vs_baseline:
        sp = sorted(speedups_vs_baseline)
        geomean = math.exp(statistics.fmean(math.log(s) for s in sp))
        print(f"\nSpeedup vs FlashInfer baseline across {len(sp)} workloads:")
        print(f"  mean    = {statistics.mean(sp):.3f}x")
        print(f"  geomean = {geomean:.3f}x")
        print(f"  p50     = {sp[len(sp)//2]:.3f}x")
        print(f"  min     = {sp[0]:.3f}x")
        print(f"  max     = {sp[-1]:.3f}x")

    if failed > 0:
        print(f"\n*** FAILED: {failed} workloads did not pass in the official image. ***")
        raise SystemExit(1)

    print("\nAll workloads PASSED in the official image.")
