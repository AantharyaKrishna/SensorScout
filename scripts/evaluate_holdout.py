
import argparse
import json
import sys
from pathlib import Path

# Find the SensorScout project root and make src importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluator import evaluate_configuration
from src.optimizer import BASELINE


def load_optimization_run(run_path):
    """Load a previously saved optimization result."""
    if not run_path.exists():
        raise FileNotFoundError(
            f"Optimization result not found: {run_path}\n"
            "Run: python scripts/run_optimization.py"
        )

    with run_path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    if "best_config" not in result:
        raise ValueError(
            "The optimization result does not contain 'best_config'."
        )

    return result


def summarize_metrics(metrics):
    """Keep summary metrics while excluding per-scenario details."""
    return {
        key: value
        for key, value in metrics.items()
        if key != "scenario_results"
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline and optimized SensorScout configurations "
            "on held-out test scenarios."
        )
    )

    parser.add_argument(
        "--run",
        default="data/mock_run.json",
        help="Saved optimization JSON path, relative to project root.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=98765,
        help="Seed for the held-out test scenarios.",
    )

    parser.add_argument(
        "--output",
        default="data/holdout_results.json",
        help="Output JSON path, relative to project root.",
    )

    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be a non-negative integer.")

    run_path = ROOT / args.run
    optimization_result = load_optimization_run(run_path)

    # Use the exact configurations from the saved optimization run.
    baseline_config = optimization_result.get("baseline", BASELINE)
    optimized_config = optimization_result["best_config"]

    configurations = {
        "baseline": baseline_config,
        "optimized": optimized_config,
    }

    results = {}

    print("\n=== SensorScout Held-Out Evaluation ===")
    print(f"Scenario set: test")
    print(f"Test seed: {args.seed}")
    print(f"Optimization run: {args.run}")

    for name, config in configurations.items():
        print(f"\nEvaluating {name} configuration...")

        metrics = evaluate_configuration(
            config=config,
            scenario_set="test",
            seed=args.seed,
        )

        results[name] = {
            "config": config,
            "metrics": summarize_metrics(metrics),
        }

        print(f"Configuration: {config}")
        print(f"Score: {metrics['score']:.4f}")
        print(f"Fault recall: {metrics['fault_recall']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(
            "False alarm rate: "
            f"{metrics['false_alarm_rate']:.4f}"
        )
        print(
            "Mean detection delay: "
            f"{metrics['mean_detection_delay_s']}"
        )
        print(f"Sensor cost: {metrics['sensor_cost_units']}")

    baseline_metrics = results["baseline"]["metrics"]
    optimized_metrics = results["optimized"]["metrics"]

    score_difference = (
        optimized_metrics["score"] - baseline_metrics["score"]
    )

    delay_difference = (
        optimized_metrics["mean_detection_delay_s"]
        - baseline_metrics["mean_detection_delay_s"]
    )

    output = {
        "scenario_set": "test",
        "seed": args.seed,
        "source_run": args.run,
        "score_difference_optimized_minus_baseline": score_difference,
        "detection_delay_difference_s_optimized_minus_baseline": (
            delay_difference
        ),
        "results": results,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\n=== Held-Out Comparison ===")
    print(f"Baseline score: {baseline_metrics['score']:.4f}")
    print(f"Optimized score: {optimized_metrics['score']:.4f}")
    print(f"Score difference: {score_difference:+.4f}")
    print(
        "Detection delay difference: "
        f"{delay_difference:+.4f} seconds"
    )

    if score_difference > 0:
        print("Result: Optimized configuration scored higher.")
    elif score_difference < 0:
        print("Result: Baseline configuration scored higher.")
    else:
        print("Result: Both configurations have the same score.")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()