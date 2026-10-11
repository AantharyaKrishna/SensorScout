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