import pandas as pd

from src.evaluator import detect_fault
from src.simulator import simulate_machine


def test_healthy_machine_has_no_alarm():
    readings = simulate_machine(fault_type="healthy", seed=42)

    result = detect_fault(readings, [0, 3, 6])

    assert result["fault_detected"] is False
    assert result["first_alarm_s"] is None


def test_nearby_sensor_detects_bearing_wear():
    readings = simulate_machine(
        fault_type="bearing_wear",
        fault_position=3,
        seed=42,
    )

    result = detect_fault(readings, [3])

    assert result["fault_detected"] is True
    assert result["first_alarm_s"] >= 40


def test_distant_sensor_misses_bearing_wear():
    readings = simulate_machine(
        fault_type="bearing_wear",
        fault_position=3,
        seed=42,
    )

    result = detect_fault(readings, [7])

    assert result["fault_detected"] is False


def test_simulation_is_repeatable():
    first = simulate_machine(fault_type="overheating", seed=42)
    second = simulate_machine(fault_type="overheating", seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_detector_ignores_fault_labels():
    readings = simulate_machine(
        fault_type="bearing_wear",
        fault_position=3,
        seed=42,
    )

    measurements_only = readings.drop(
        columns=["fault_type", "fault_active", "severity"]
    )

    assert detect_fault(readings, [3]) == detect_fault(
        measurements_only, [3]
    )

import pytest

from src.evaluator import evaluate_configuration


def test_evaluation_is_repeatable():
    config = {
        "sensor_positions": [0, 3, 6],
        "sampling_rates_hz": [5, 10, 20],
    }

    first = evaluate_configuration(config, seed=42)
    second = evaluate_configuration(config, seed=42)

    assert first == second


def test_evaluation_counts_all_fault_scenarios():
    config = {
        "sensor_positions": [0],
        "sampling_rates_hz": [10],
    }

    result = evaluate_configuration(config, seed=42)

    assert len(result["scenario_results"]) == 32
    assert result["faults_detected"] + result["faults_missed"] == 24

    assert result["fault_recall"] == pytest.approx(
        result["faults_detected"] / 24
    )

    assert result["false_alarm_rate"] == pytest.approx(
        result["false_alarm_windows"] / 32
    )


def test_distributed_layout_improves_recall():
    single = evaluate_configuration(
        {
            "sensor_positions": [0],
            "sampling_rates_hz": [10],
        },
        seed=42,
    )

    distributed = evaluate_configuration(
        {
            "sensor_positions": [0, 3, 6],
            "sampling_rates_hz": [10, 10, 10],
        },
        seed=42,
    )

    assert distributed["fault_recall"] > single["fault_recall"]


@pytest.mark.parametrize(
    "config",
    [
        {"sensor_positions": [], "sampling_rates_hz": []},
        {"sensor_positions": [0, 3], "sampling_rates_hz": [10]},
        {"sensor_positions": [0, 0], "sampling_rates_hz": [10, 10]},
        {"sensor_positions": [8], "sampling_rates_hz": [10]},
        {"sensor_positions": [0], "sampling_rates_hz": [7]},
    ],
)
def test_invalid_configurations_are_rejected(config):
    with pytest.raises(ValueError):
        evaluate_configuration(config)