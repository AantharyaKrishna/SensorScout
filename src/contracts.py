
from typing import Any

POSITIONS = tuple(range(8))
SAMPLING_RATES = (5, 10, 20)


def validate_configuration(config: dict) -> dict:
    """Validate a proposed three-sensor configuration."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")

    positions = config.get("sensor_positions")
    rates = config.get("sampling_rates_hz")

    if not isinstance(positions, list) or not isinstance(rates, list):
        raise ValueError("Positions and rates must be lists")

    if len(positions) != 3 or len(set(positions)) != 3:
        raise ValueError("Exactly 3 unique sensor positions required")

    if any(type(p) is not int or p not in POSITIONS for p in positions):
        raise ValueError("Sensor positions must be integers from 0 to 7")

    if len(rates) != 3 or any(
        type(r) is not int or r not in SAMPLING_RATES for r in rates
    ):
        raise ValueError("Sampling rates must be 5, 10, or 20 Hz")

    return {
        "sensor_positions": positions,
        "sampling_rates_hz": rates,
    }


def validate_metrics(metrics: dict[str, Any]) -> dict:
    """Check that the evaluator returned the required fields."""
    required = {
        "fault_recall",
        "precision",
        "false_alarms_per_hour",
        "mean_warning_lead_minutes",
        "sensor_cost",
        "score",
        "n_scenarios",
    }

    if not isinstance(metrics, dict):
        raise ValueError("Evaluator output must be a dictionary")

    missing = required - metrics.keys()
    if missing:
        raise ValueError(f"Missing evaluator metrics: {sorted(missing)}")

    return metrics
