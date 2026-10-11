
import json
import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimizer import run_optimization


def main():
    print("\n=== SensorScout AI Optimization ===\n")

    result = run_optimization(rounds=5, seed=42)

    for item in result["history"]:
        print(
            f"Round {item['round']}: "
            f"score={item['metrics']['score']:.4f}, "
            f"best={item['best_score']:.4f}, "
            f"accepted={item['accepted']}"
        )

    output = Path("data/mock_run.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(json.dumps(result, indent=2))

    print("\nBest configuration:", result["best_config"])
    print("Best metrics:", result["best_metrics"])
    print("Evaluator mode:", result["evaluator_mode"])
    print("Saved results to:", output)


if __name__ == "__main__":
    main()
