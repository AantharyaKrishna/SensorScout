
import random

POSITIONS = tuple(range(8))
RATES = (5, 10, 20)


def validate_proposal(config):
    if not isinstance(config, dict):
        raise ValueError("Proposal must be a dictionary")

    positions = config.get("sensor_positions")
    rates = config.get("sampling_rates_hz")

    if not isinstance(positions, list) or not isinstance(rates, list):
        raise ValueError("Positions and rates must be lists")

    if len(positions) != 3 or len(set(positions)) != 3:
        raise ValueError("Exactly 3 unique sensors required")

    if any(type(p) is not int or p not in POSITIONS for p in positions):
        raise ValueError("Invalid sensor position")

    if len(rates) != 3 or any(
        type(r) is not int or r not in RATES for r in rates
    ):
        raise ValueError("Invalid sampling rate")

    return config


class ChallengerAgent:
    def challenge(self, config, rng=None):
        rng = rng or random.Random()
        positions = config["sensor_positions"]
        unmonitored = [p for p in POSITIONS if p not in positions]
        target = rng.choice(unmonitored)

        return {
            "target_position": target,
            "reason": f"Test a potential blind spot at position {target}",
        }


class DiagnosisAgent:
    def diagnose(self, config, challenge):
        target = challenge["target_position"]
        nearest = min(
            config["sensor_positions"],
            key=lambda p: abs(p - target)
        )

        return {
            "weak_position": target,
            "nearest_sensor": nearest,
            "distance": abs(target - nearest),
            "recommendation": "Test sensor relocation",
        }


class DesignerAgent:
    def propose(self, config, diagnosis, rng=None):
        rng = rng or random.Random()

        positions = config["sensor_positions"].copy()
        rates = config["sampling_rates_hz"].copy()
        target = diagnosis["weak_position"]

        if target not in positions:
            candidates = [p for p in positions if p != target]
            replacement = rng.choice(candidates)
            index = positions.index(replacement)
            positions[index] = target

        index = rng.randrange(3)
        rates[index] = rng.choice(RATES)

        return validate_proposal({
            "sensor_positions": positions,
            "sampling_rates_hz": rates,
        })


class StrategistAgent:
    def decide(self, current_metrics, candidate_metrics):
        old_score = current_metrics["score"]
        new_score = candidate_metrics["score"]

        accepted = new_score > old_score

        return {
            "accepted": accepted,
            "old_score": old_score,
            "new_score": new_score,
            "improvement": new_score - old_score,
            "reason": (
                "Candidate improves the score"
                if accepted else "Keep existing configuration"
            ),
        }


import json
import re

from src.contracts import validate_configuration
from src.llm_client import call_model


def ai_designer_proposal(config, diagnosis):
    """Ask DeepSeek to propose an improved sensor configuration."""

    system_prompt = """
You are SensorScout's Designer Agent.

Your job is to suggest sensor placements that may improve
industrial fault detection.

Rules:
- There are 8 possible sensor positions: 0 through 7.
- Select exactly 3 unique positions.
- Choose exactly 3 sampling rates.
- Allowed rates are 5, 10, or 20 Hz.
- Return ONLY a JSON object with:
  sensor_positions
  sampling_rates_hz

Do not include markdown or explanations.
"""

    user_prompt = (
        f"Current configuration: {json.dumps(config)}\n"
        f"Diagnosis: {json.dumps(diagnosis)}\n"
        "Propose a better configuration."
    )

    response = call_model(system_prompt, user_prompt)

    try:
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        proposal = json.loads(cleaned)
        validate_configuration(proposal)
        return proposal

    except (ValueError, TypeError, AttributeError):
        return config.copy()

