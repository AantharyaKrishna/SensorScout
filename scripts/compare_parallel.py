
import json
import sys
import time
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluator import evaluate_configuration


def evaluate_job(job):
    """Evaluate one configuration in a worker process."""
    config, scenario_set, seed = job
    metrics = evaluate_configuration(
        config=config,
        scenario_set=scenario_set,
        seed=seed,
    )

    # Return only summary metrics to keep benchmark output small.
    summary = {
        key: value
        for key, value in metrics.items()
        if key != "scenario_results"
    }

    return config, summary


def generate_configurations(count=12, seed=2026):
    """Generate reproducible, unique three-sensor configurations."""
    rng = random.Random(seed)
    configurations = []
    seen = set()

    while len(configurations) < count:
        positions = sorted(rng.sample(range(8), 3))
        rates = [rng.choice([5, 10, 20]) for _ in range(3)]

        config = {
            "sensor_positions": positions,
            "sampling_rates_hz": rates,
        }

        key = (
            tuple(positions),
            tuple(rates),
        )

        if key not in seen:
            seen.add(key)
            configurations.append(config)

    return configurations


def main():
    count = 12
    scenario_seed = 98765
    scenario_set = "test"

    configurations = generate_configurations(count=count)

    jobs = [
        (config, scenario_set, scenario_seed)
        for config in configurations
    ]

    print("=== SensorScout Parallel Evaluation Benchmark ===")
    print(f"Configurations: {len(jobs)}")
    print(f"Scenario set: {scenario_set}")
    print(f"Scenario seed: {scenario_seed}")

    # Sequential benchmark.
    start = time.perf_counter()

    sequential_results = [
        evaluate_job(job)
        for job in jobs
    ]

    sequential_seconds = time.perf_counter() - start

    # Concurrent benchmark using separate worker processes.
    worker_count = min(4, len(jobs))

    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        parallel_results = list(executor.map(evaluate_job, jobs))

    parallel_seconds = time.perf_counter() - start

    # Compare scores in the same configuration order.
    equivalent = True

    for (seq_config, seq_metrics), (par_config, par_metrics) in zip(
        sequential_results,
        parallel_results,
    ):
        if seq_config != par_config:
            equivalent = False
            break

        if seq_metrics.keys() != par_metrics.keys():
            equivalent = False
            break

        for key in seq_metrics:
            seq_value = seq_metrics[key]
            par_value = par_metrics[key]

            if isinstance(seq_value, (int, float)) and isinstance(
                par_value, (int, float)
            ):
                if abs(seq_value - par_value) > 1e-9:
                    equivalent = False
                    break
            elif seq_value != par_value:
                equivalent = False
                break

        if not equivalent:
            break

    speedup = (
        sequential_seconds / parallel_seconds
        if parallel_seconds > 0
        else None
    )

    result = {
        "scenario_set": scenario_set,
        "scenario_seed": scenario_seed,
        "configuration_count": count,
        "workers": worker_count,
        "sequential_seconds": sequential_seconds,
        "parallel_seconds": parallel_seconds,
        "speedup": speedup,
        "results_equivalent": equivalent,
        "sequential_results": [
            {"config": config, "metrics": metrics}
            for config, metrics in sequential_results
        ],
        "parallel_results": [
            {"config": config, "metrics": metrics}
            for config, metrics in parallel_results
        ],
    }

    output_path = ROOT / "data" / "parallel_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(f"\nSequential time: {sequential_seconds:.3f} seconds")
    print(f"Parallel time:   {parallel_seconds:.3f} seconds")

    if speedup is not None:
        print(f"Speedup:         {speedup:.2f}x")

    print(f"Results equivalent: {equivalent}")
    print(f"Saved results to: {output_path}")

    if not equivalent:
        raise RuntimeError(
            "Sequential and parallel evaluations produced different results."
        )


if __name__ == "__main__":
    main()