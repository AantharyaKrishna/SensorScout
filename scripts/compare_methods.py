
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimizer import BASELINE, mock_evaluator, run_optimization


def random_search(rounds=5, seed=42):
    rng = random.Random(seed)

    best_config = BASELINE.copy()
    best_metrics = mock_evaluator(best_config, seed=seed)

    for _ in range(rounds):
        config = {
            "sensor_positions": sorted(rng.sample(range(8), 3)),
            "sampling_rates_hz": [
                rng.choice([5, 10, 20]) for _ in range(3)
            ],
        }

        metrics = mock_evaluator(config, seed=seed)

        if metrics["score"] > best_metrics["score"]:
            best_config = config
            best_metrics = metrics

    return best_config, best_metrics


def main():
    baseline_metrics = mock_evaluator(BASELINE, seed=42)

    random_config, random_metrics = random_search()

    ai_result = run_optimization(rounds=5, seed=42)

    print("\n=== SensorScout Method Comparison ===\n")
    print(f"Fixed baseline: {baseline_metrics['score']:.4f}")
    print(f"Random search:  {random_metrics['score']:.4f}")
    print(f"DeepSeek AI:    {ai_result['best_metrics']['score']:.4f}")

    print("\nBest random configuration:", random_config)
    print("Best AI configuration:", ai_result["best_config"])
    print("\nEvaluator: MOCK (synthetic scores only)")


if __name__ == "__main__":
    main()
