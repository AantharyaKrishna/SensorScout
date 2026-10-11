import numpy as np
import pandas as pd

FAULT_TYPES = ("healthy", "bearing_wear", "imbalance", "overheating")


def simulate_machine(
    fault_type="healthy",
    fault_position=3,
    duration_s=120,
    sampling_rate_hz=10,
    fault_start_s=40,
    seed=42,
):
    """Generate repeatable readings at eight possible sensor positions."""
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"fault_type must be one of {FAULT_TYPES}")

    if fault_position not in range(8):
        raise ValueError("fault_position must be an integer from 0 to 7")

    if duration_s <= 0 or sampling_rate_hz <= 0:
        raise ValueError("Duration and sampling rate must be positive")

    if not 0 <= fault_start_s < duration_s:
        raise ValueError("Fault start must fall within the simulation")

    rng = np.random.default_rng(seed)
    time = np.arange(0, duration_s, 1 / sampling_rate_hz)

    # Fault severity grows from zero at onset to one at the end.
    severity = np.clip(
        (time - fault_start_s) / (duration_s - fault_start_s),
        0,
        1,
    )

    if fault_type == "healthy":
        severity = np.zeros_like(time)

    frames = []

    for position in range(8):
        distance = abs(position - fault_position)
        local_effect = np.exp(-distance / 1.5)

        vibration = (
            0.8
            + 0.04 * np.sin(2 * np.pi * 0.1 * time)
            + rng.normal(0, 0.06, len(time))
        )

        temperature = (
            35
            + 0.3 * np.sin(2 * np.pi * 0.01 * time)
            + rng.normal(0, 0.2, len(time))
        )

        if fault_type == "bearing_wear":
            vibration += 3.0 * severity * local_effect
            temperature += 4.0 * severity * local_effect

        elif fault_type == "imbalance":
            vibration += (
                2.2
                * severity
                * local_effect
                * (1 + 0.2 * np.sin(2 * np.pi * 0.5 * time))
            )

        elif fault_type == "overheating":
            temperature += 35.0 * severity * local_effect
            vibration += 0.3 * severity * local_effect

        frames.append(
            pd.DataFrame(
                {
                    "time_s": time,
                    "position": position,
                    "vibration_rms_mm_s": np.maximum(vibration, 0),
                    "temperature_c": temperature,
                    "fault_type": fault_type,
                    "fault_active": (
                        (time >= fault_start_s)
                        & (fault_type != "healthy")
                    ),
                    "severity": severity,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    readings = simulate_machine()
    print(readings.head())
    print(f"\nGenerated {len(readings)} rows")