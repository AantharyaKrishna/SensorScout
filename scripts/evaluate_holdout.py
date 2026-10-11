import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluator import evaluate_configuration
from src.optimizer import BASELINE


def load_optimization_run(run_path):
    if not run_path.exists():
        raise FileNotFoundError(
            f"Optimization result not found: {run_path}\n"
            "Run: python scripts/run_optimization.py"
        )

    with run_path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    if "best_config" not in result:
        raise ValueError("Saved run is missing 'best_config'.")

    return result


def summarize_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key != "scenario_results"
    }


def optional_mean(values):
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def main():
    parser = argparse.ArgumentParser(
        description="Compare fixed layouts across held-out test seeds."
    )
    parser.add_argument("--run", default="data/mock_run.json")
    parser.add_argument("--seed", type=int, default=98765)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--output",
        default="data/holdout_results.json",
    )
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.count < 1:
        parser.error("--count must be at least 1.")

    saved_run = load_optimization_run(ROOT / args.run)

    # Freeze layouts before evaluating any test scenarios.
    configurations = {
        "baseline": saved_run.get("baseline", BASELINE),
        "optimized": saved_run["best_config"],
    }

    # Each evaluation uses 32 consecutive scenario seeds.
    # A spacing of 1,000 prevents overlap between these runs.
    seeds = [args.seed + 1000 * index for index in range(args.count)]
    comparisons = []

    print("\n=== SensorScout Held-Out Evaluation ===")
    print("Scenario set: test")
    print(f"Test seeds: {seeds}")
    print(f"Baseline: {configurations['baseline']}")
    print(f"Optimized: {configurations['optimized']}")

    for seed in seeds:
        results = {}

        for name, config in configurations.items():
            metrics = evaluate_configuration(
                config=config,
                scenario_set="test",
                seed=seed,
            )
            results[name] = summarize_metrics(metrics)

        baseline = results["baseline"]
        optimized = results["optimized"]

        if baseline["score_version"] != optimized["score_version"]:
            raise ValueError("Scoring versions do not match.")

        score_difference = optimized["score"] - baseline["score"]

        baseline_delay = baseline["mean_detection_delay_s"]
        optimized_delay = optimized["mean_detection_delay_s"]
        delay_difference = (
            optimized_delay - baseline_delay
            if baseline_delay is not None and optimized_delay is not None
            else None
        )

        comparisons.append({
            "seed": seed,
            "score_difference": score_difference,
            "detection_delay_difference_s": delay_difference,
            "results": results,
        })

        print(
            f"Seed {seed}: "
            f"baseline={baseline['score']:.4f}, "
            f"optimized={optimized['score']:.4f}, "
            f"difference={score_difference:+.4f}"
        )

    metric_names = (
        "score",
        "fault_recall",
        "precision",
        "false_alarm_rate",
        "mean_detection_delay_s",
        "sensor_cost_units",
    )

    averages = {
        name: {
            metric: optional_mean([
                comparison["results"][name][metric]
                for comparison in comparisons
            ])
            for metric in metric_names
        }
        for name in configurations
    }

    differences = [item["score_difference"] for item in comparisons]
    tolerance = 1e-9

    summary = {
        "mean_metrics": averages,
        "mean_score_difference": mean(differences),
        "score_difference_standard_deviation": (
            stdev(differences) if len(differences) > 1 else None
        ),
        "minimum_score_difference": min(differences),
        "maximum_score_difference": max(differences),
        "optimized_wins": sum(value > tolerance for value in differences),
        "baseline_wins": sum(value < -tolerance for value in differences),
        "ties": sum(abs(value) <= tolerance for value in differences),
        "mean_paired_detection_delay_difference_s": optional_mean([
            item["detection_delay_difference_s"]
            for item in comparisons
        ]),
    }

    output = {
        "scenario_set": "test",
        "seeds": seeds,
        "source_run": args.run,
        "configurations": configurations,
        "score_version": comparisons[0]["results"]["baseline"]["score_version"],
        "summary": summary,
        "comparisons": comparisons,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== Summary Across Test Seeds ===")
    for name, metrics in averages.items():
        print(f"\n{name.capitalize()}:")
        for metric, value in metrics.items():
            formatted = f"{value:.4f}" if value is not None else "N/A"
            print(f"  {metric}: {formatted}")

    print(
        "\nMean score difference: "
        f"{summary['mean_score_difference']:+.4f}"
    )
    print(
        f"Optimized wins: {summary['optimized_wins']}/{len(seeds)}"
    )
    print(f"Baseline wins: {summary['baseline_wins']}/{len(seeds)}")
    print(f"Ties: {summary['ties']}")

    delay = summary["mean_paired_detection_delay_difference_s"]
    if delay is not None:
        print(f"Mean delay difference: {delay:+.4f} seconds")
        print("Negative delay difference means faster detection.")

    print(
        "\nThese tests vary simulator noise; they do not establish "
        "performance on real machines or new fault types."
    )
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()