
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
    def challenge(self, config, rng=None, metrics=None):
        rng = rng or random.Random()

        if metrics and metrics.get("scenario_results"):
            location_stats = []

            for position in POSITIONS:
                scenarios = [
                    item
                    for item in metrics["scenario_results"]
                    if item["fault_position"] == position
                    and item["fault_type"] != "healthy"
                ]

                if not scenarios:
                    continue

                missed = sum(
                    not item["detected"] for item in scenarios
                )

                delays = [
                    item["detection_delay_s"]
                    for item in scenarios
                    if item["detection_delay_s"] is not None
                ]

                avg_delay = (
                    sum(delays) / len(delays)
                    if delays else None
                )

                location_stats.append({
                    "position": position,
                    "missed_faults": missed,
                    "average_delay_s": avg_delay,
                })

            # Prioritize missed faults, then slow detection
            worst = max(
                location_stats,
                key=lambda item: (
                    item["missed_faults"],
                    item["average_delay_s"]
                    if item["average_delay_s"] is not None
                    else float("inf"),
                ),
            )

            return {
                "target_position": worst["position"],
                "missed_faults": worst["missed_faults"],
                "average_delay_s": worst["average_delay_s"],
                "reason": (
                    f"Position {worst['position']} has "
                    f"{worst['missed_faults']} missed faults "
                    f"and an average detection delay of "
                    f"{worst['average_delay_s']:.2f} seconds."
                ),
            }

        # Fallback if no evaluation data is available
        unmonitored = [
            p for p in POSITIONS
            if p not in config["sensor_positions"]
        ]

        target = rng.choice(unmonitored)

        return {
            "target_position": target,
            "reason": f"Explore position {target}",
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
            "missed_faults": challenge.get("missed_faults"),
            "average_delay_s": challenge.get("average_delay_s"),
            "recommendation": (
                f"Investigate relocating a sensor closer to "
                f"position {target}, while preserving coverage "
                "at other locations."
            ),
        }



class DesignerAgent:
    def propose(self, config, diagnosis, rng=None):
        """Propose a targeted sampling-rate adjustment."""

        positions = config["sensor_positions"].copy()
        rates = config["sampling_rates_hz"].copy()

        target = diagnosis["weak_position"]

        # Find the sensor nearest to the weak location
        nearest_index = min(
            range(len(positions)),
            key=lambda i: abs(positions[i] - target),
        )

        # Try increasing sampling at the nearest sensor
        current_rate = rates[nearest_index]

        if current_rate == 5:
            rates[nearest_index] = 10
        elif current_rate == 10:
            rates[nearest_index] = 20
        else:
            # Already at maximum; try reducing a different sensor
            other_indices = [
                i for i in range(len(rates))
                if i != nearest_index and rates[i] > 5
            ]

            if other_indices:
                i = max(other_indices, key=lambda j: rates[j])
                rates[i] = 10 if rates[i] == 20 else 5

        return validate_proposal({
            "sensor_positions": positions,
            "sampling_rates_hz": rates,
        })




class StrategistAgent:
    def decide(self, current_metrics, candidate_metrics):
        """Compare fault coverage, detection speed, and cost."""

        old_score = current_metrics["score"]
        new_score = candidate_metrics["score"]

        old_recall = current_metrics["fault_recall"]
        new_recall = candidate_metrics["fault_recall"]

        old_delay = current_metrics.get("mean_detection_delay_s")
        new_delay = candidate_metrics.get("mean_detection_delay_s")

        old_cost = current_metrics["sensor_cost_units"]
        new_cost = candidate_metrics["sensor_cost_units"]

        # Never sacrifice fault recall
        recall_ok = new_recall >= old_recall

        # Prefer configurations that improve the existing score
        score_improved = new_score > old_score + 1e-6

        # Track whether detection became faster
        delay_improved = (
            old_delay is not None
            and new_delay is not None
            and new_delay < old_delay
        )

        # Accept a better score only if recall is preserved.
        # A faster configuration can also be accepted if
        # its cost does not increase and recall is preserved.
        accepted = recall_ok and (
            score_improved
            or (
                delay_improved
                and new_cost <= old_cost
                and new_score >= old_score - 0.5
            )
        )

        return {
            "accepted": accepted,
            "old_score": old_score,
            "new_score": new_score,
            "improvement": new_score - old_score,
            "old_delay_s": old_delay,
            "new_delay_s": new_delay,
            "old_cost": old_cost,
            "new_cost": new_cost,
            "reason": (
                "Candidate improves the monitoring trade-off"
                if accepted
                else "Candidate does not meet acceptance criteria"
            ),
        }



import json
import re

from src.contracts import validate_configuration
from src.llm_client import call_model



    
def ai_designer_proposal(config, diagnosis):
    """Generate a proposal using DeepSeek or a rule-based fallback."""

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
        "Propose a configuration specifically targeting the diagnosed "
        "weak position. Consider moving a sensor closer to that location "
        "or changing its sampling rate. Preserve detection performance "
        "at other locations and minimize sensor cost. "
        "Return only the required JSON."
    )

    try:
        response = call_model(system_prompt, user_prompt)

    except Exception as error:
        print(f"DeepSeek unavailable: {error}")
        print("Using rule-based Designer Agent instead.")

        fallback = DesignerAgent().propose(
            config,
            diagnosis,
            random.Random(42),
        )

        return fallback, "rule_based_fallback"

    try:
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        proposal = json.loads(cleaned)
        validate_configuration(proposal)

        return proposal, "deepseek"

    except (ValueError, TypeError, AttributeError) as error:
        print(f"Invalid DeepSeek response: {error}")
        print("Using rule-based Designer Agent instead.")

        fallback = DesignerAgent().propose(
            config,
            diagnosis,
            random.Random(42),
        )

        return fallback, "invalid_response_fallback"
