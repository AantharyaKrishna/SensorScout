import numpy as np
import pandas as pd

from src.simulator import simulate_machine

FAULT_TYPES = ("healthy", "bearing_wear", "imbalance", "overheating")
SAMPLING_RATES = (5, 10, 20)

DURATION_S = 120
FAULT_START_S = 40
BASE_SAMPLING_RATE_HZ = 20
SCORE_VERSION = "v2_delay"


def detect_fault(readings, sensor_positions):
    """Detect faults using only selected sensors' measurements."""
    positions = list(sensor_positions)

    if not positions:
        raise ValueError("Select at least one sensor")

    if any(type(p) is not int or p not in range(8) for p in positions):
        raise ValueError("Sensor positions must be integers from 0 to 7")

    if len(set(positions)) != len(positions):
        raise ValueError("Sensor positions must be unique")

    selected = readings[
        readings["position"].isin(positions)
    ].copy()

    # Demo thresholds, not actual equipment safety limits.
    selected["alarm"] = (
        (selected["vibration_rms_mm_s"] > 1.5)
        | (selected["temperature_c"] > 45)
    )

    alarms = selected[selected["alarm"]]

    return {
        "fault_detected": not alarms.empty,
        "first_alarm_s": (
            float(alarms["time_s"].min())
            if not alarms.empty
            else None
        ),
        "sensor_positions": positions,
    }


def evaluate_configuration(config, scenario_set="validation", seed=42):
    """Evaluate a layout across 24 faulty and 8 healthy scenarios."""
    positions = list(config.get("sensor_positions", []))
    rates = list(config.get("sampling_rates_hz", []))

    if not positions:
        raise ValueError("Select at least one sensor")

    if len(positions) != len(rates):
        raise ValueError("Each sensor needs one sampling rate")

    if any(type(p) is not int or p not in range(8) for p in positions):
        raise ValueError("Sensor positions must be integers from 0 to 7")

    if len(set(positions)) != len(positions):
        raise ValueError("Sensor positions must be unique")

    if any(rate not in SAMPLING_RATES for rate in rates):
        raise ValueError("Supported sampling rates are 5, 10, or 20 Hz")

    rates = [int(rate) for rate in rates]

    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a non-negative integer")

    offsets = {
        "train": 0,
        "validation": 100_000,
        "test": 200_000,
    }

    if scenario_set not in offsets:
        raise ValueError("Choose train, validation, or test")

    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0

    delays = []
    scenario_results = []

    for fault_index, fault in enumerate(FAULT_TYPES):
        for fault_position in range(8):
            scenario_seed = (
                seed
                + offsets[scenario_set]
                + fault_index * 8
                + fault_position
            )

            # Every layout receives the same underlying scenario.
            readings = simulate_machine(
                fault_type=fault,
                fault_position=fault_position,
                duration_s=DURATION_S,
                sampling_rate_hz=BASE_SAMPLING_RATE_HZ,
                fault_start_s=FAULT_START_S,
                seed=scenario_seed,
            )

            # Sample each selected sensor at its configured rate.
            sampled_frames = []

            for position, rate in zip(positions, rates):
                sensor_readings = readings[
                    readings["position"] == position
                ]

                step = BASE_SAMPLING_RATE_HZ // rate
                sampled_frames.append(sensor_readings.iloc[::step])

            sampled = pd.concat(sampled_frames, ignore_index=True)

            if fault == "healthy":
                outcome = detect_fault(sampled, positions)
                false_alarm = outcome["fault_detected"]

                false_positives += int(false_alarm)
                true_negatives += int(not false_alarm)

                scenario_results.append({
                    "fault_type": fault,
                    "fault_position": fault_position,
                    "scenario_seed": scenario_seed,
                    "detected": False,
                    "false_alarm": false_alarm,
                    "detection_delay_s": None,
                })
                continue

            # Evaluate healthy and faulty observation windows separately.
            before_fault = sampled[
                sampled["time_s"] < FAULT_START_S
            ]
            after_fault = sampled[
                sampled["time_s"] >= FAULT_START_S
            ]

            early_alarm = detect_fault(
                before_fault, positions
            )["fault_detected"]

            outcome = detect_fault(after_fault, positions)
            detected = outcome["fault_detected"]

            false_positives += int(early_alarm)
            true_negatives += int(not early_alarm)
            true_positives += int(detected)
            false_negatives += int(not detected)

            delay = (
                outcome["first_alarm_s"] - FAULT_START_S
                if detected
                else None
            )

            if delay is not None:
                delays.append(delay)

            scenario_results.append({
                "fault_type": fault,
                "fault_position": fault_position,
                "scenario_seed": scenario_seed,
                "detected": detected,
                "false_alarm": early_alarm,
                "detection_delay_s": delay,
            })

    fault_count = true_positives + false_negatives

    recall = true_positives / fault_count

    # Each faulty window and healthy window contributes at most one alarm.
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )

    false_alarm_rate = (
        false_positives / (false_positives + true_negatives)
    )

    # Illustrative cost units, not hardware prices.
    sensor_cost = sum(100 + 2 * rate for rate in rates)
    cost_fraction = sensor_cost / (8 * 140)

    # Missed faults receive the entire post-onset observation duration.
    observation_duration_s = DURATION_S - FAULT_START_S

    delay_penalty = (
        sum(delays)
        + false_negatives * observation_duration_s
    ) / (fault_count * observation_duration_s)

    score = (
        100 * recall
        - 30 * false_alarm_rate
        - 10 * cost_fraction
        - 20 * delay_penalty
    )

    return {
        "config": {
            "sensor_positions": positions,
            "sampling_rates_hz": rates,
        },
        "scenario_set": scenario_set,
        "seed": seed,
        "fault_recall": recall,
        "precision": precision,
        "false_alarm_rate": false_alarm_rate,
        "mean_detection_delay_s": (
            float(np.mean(delays)) if delays else None
        ),
        "sensor_cost_units": sensor_cost,
        "score": score,
        "score_version": SCORE_VERSION,
        "delay_penalty": delay_penalty,
        "faults_detected": true_positives,
        "faults_missed": false_negatives,
        "false_alarm_windows": false_positives,
        "scenario_results": scenario_results,
    }


if __name__ == "__main__":
    config = {
        "sensor_positions": [0, 3, 6],
        "sampling_rates_hz": [5, 10, 20],
    }

    result = evaluate_configuration(
        config,
        scenario_set="validation",
        seed=42,
    )

    summary = {
        key: value
        for key, value in result.items()
        if key != "scenario_results"
    }

    print(summary)