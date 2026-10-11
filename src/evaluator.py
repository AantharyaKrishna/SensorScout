from src.simulator import simulate_machine


def detect_fault(readings, sensor_positions):
    """Detect a fault using only the selected sensors' measurements."""
    if not sensor_positions:
        raise ValueError("Select at least one sensor")

    if any(position not in range(8) for position in sensor_positions):
        raise ValueError("Sensor positions must be between 0 and 7")

    selected = readings[
        readings["position"].isin(sensor_positions)
    ].copy()

    # Initial demo thresholds, not real equipment safety limits.
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
        "sensor_positions": list(sensor_positions),
    }


if __name__ == "__main__":
    for fault in (
        "healthy",
        "bearing_wear",
        "imbalance",
        "overheating",
    ):
        readings = simulate_machine(
            fault_type=fault,
            fault_position=3,
            seed=42,
        )

        result = detect_fault(
            readings,
            sensor_positions=[0, 3, 6],
        )

        print(f"{fault}: {result}")
def evaluate_configuration(config, scenario_set="validation", seed=42):
    """Evaluate a layout across 24 faulty and 8 healthy scenarios."""
    import numpy as np
    import pandas as pd

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

    if any(rate not in (5, 10, 20) for rate in rates):
        raise ValueError("Supported sampling rates are 5, 10, or 20 Hz")

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

    faults = ("healthy", "bearing_wear", "imbalance", "overheating")

    for fault_index, fault in enumerate(faults):
        for fault_position in range(8):
            scenario_seed = (
                int(seed)
                + offsets[scenario_set]
                + fault_index * 8
                + fault_position
            )

            # Generate the same underlying signals for every layout.
            readings = simulate_machine(
                fault_type=fault,
                fault_position=fault_position,
                sampling_rate_hz=20,
                seed=scenario_seed,
            )

            # Select each sensor's readings at its configured rate.
            sampled_frames = []

            for position, rate in zip(positions, rates):
                sensor_readings = readings[
                    readings["position"] == position
                ]

                step = 20 // int(rate)
                sampled_frames.append(sensor_readings.iloc[::step])

            sampled = pd.concat(sampled_frames, ignore_index=True)

            # Measure false alarms during healthy operation.
            if fault == "healthy":
                outcome = detect_fault(sampled, positions)
                false_alarm = outcome["fault_detected"]

                false_positives += int(false_alarm)
                true_negatives += int(not false_alarm)

                scenario_results.append({
                    "fault_type": fault,
                    "fault_position": fault_position,
                    "detected": False,
                    "false_alarm": false_alarm,
                    "detection_delay_s": None,
                })
                continue

            # A pre-fault alarm must not count as fault detection.
            before_fault = sampled[sampled["time_s"] < 40]
            after_fault = sampled[sampled["time_s"] >= 40]

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
                outcome["first_alarm_s"] - 40
                if detected
                else None
            )

            if delay is not None:
                delays.append(delay)

            scenario_results.append({
                "fault_type": fault,
                "fault_position": fault_position,
                "detected": detected,
                "false_alarm": early_alarm,
                "detection_delay_s": delay,
            })

    recall = true_positives / (true_positives + false_negatives)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )

    false_alarm_rate = (
        false_positives / (false_positives + true_negatives)
    )

    # Illustrative cost units, not actual hardware prices.
    sensor_cost = sum(100 + 2 * rate for rate in rates)

    # Normalize against eight sensors operating at 20 Hz.
    cost_fraction = sensor_cost / (8 * 140)

    # Initial scoring rule; agree on this with your teammate.
    score = (
        100 * recall
        - 30 * false_alarm_rate
        - 10 * cost_fraction
    )

    return {
        "config": {
            "sensor_positions": positions,
            "sampling_rates_hz": rates,
        },
        "scenario_set": scenario_set,
        "seed": int(seed),
        "fault_recall": recall,
        "precision": precision,
        "false_alarm_rate": false_alarm_rate,
        "mean_detection_delay_s": (
            float(np.mean(delays)) if delays else None
        ),
        "sensor_cost_units": sensor_cost,
        "score": score,
        "faults_detected": true_positives,
        "faults_missed": false_negatives,
        "false_alarm_windows": false_positives,
        "scenario_results": scenario_results,
    }