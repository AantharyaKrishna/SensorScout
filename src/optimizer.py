from src.agents import ai_designer_proposal
import random

from src.agents import (
    ChallengerAgent,
    DiagnosisAgent,
    DesignerAgent,
    StrategistAgent,
    validate_proposal,
)

BASELINE = {
    "sensor_positions": [0, 3, 6],
    "sampling_rates_hz": [5, 10, 20],
}



def mock_evaluator(config, scenario_set="validation", seed=42):
    """Synthetic evaluator for testing the agent workflow only."""
    validate_proposal(config)

    rng = random.Random(seed)

    # Artificial importance of each sensor location
    position_weights = {
        0: 0.12,
        1: 0.22,
        2: 0.07,
        3: 0.04,
        4: 0.09,
        5: 0.06,
        6: 0.18,
        7: 0.03,
    }

    position_score = sum(
        position_weights[pos]
        for pos in config["sensor_positions"]
    )

    rate_bonus = sum(config["sampling_rates_hz"]) / 60

    score = min(
        1.0,
        0.25
        + position_score
        + 0.10 * rate_bonus
        + rng.uniform(0, 0.01),
    )

    return {
        "fault_recall": score,
        "precision": score,
        "false_alarms_per_hour": 1.0 - score,
        "mean_warning_lead_minutes": score * 10,
        "sensor_cost": float(len(config["sensor_positions"]) * 100),
        "score": score,
        "n_scenarios": 20,
    }



def run_optimization(rounds=5, evaluator=None, seed=42):
    """Run repeated sensor-configuration experiments."""
    evaluator = evaluator or mock_evaluator
    rng = random.Random(seed)

    challenger = ChallengerAgent()
    diagnosis_agent = DiagnosisAgent()
    designer = DesignerAgent()
    strategist = StrategistAgent()

    best_config = {
        "sensor_positions": BASELINE["sensor_positions"].copy(),
        "sampling_rates_hz": BASELINE["sampling_rates_hz"].copy(),
    }

    best_metrics = evaluator(
        best_config, scenario_set="validation", seed=seed
    )

    history = [{
        "round": 0,
        "config": best_config.copy(),
        "metrics": best_metrics,
        "best_score": best_metrics["score"],
        "accepted": True,
    }]

    for round_number in range(1, rounds + 1):
        challenge = challenger.challenge(best_config, rng)
        diagnosis = diagnosis_agent.diagnose(best_config, challenge)
        proposal = ai_designer_proposal(best_config, diagnosis)

        candidate_metrics = evaluator(
            proposal, scenario_set="validation", seed=seed
        )

        decision = strategist.decide(best_metrics, candidate_metrics)

        if decision["accepted"]:
            best_config = proposal
            best_metrics = candidate_metrics

        history.append({
            "round": round_number,
            "challenge": challenge,
            "diagnosis": diagnosis,
            "config": proposal,
            "metrics": candidate_metrics,
            "accepted": decision["accepted"],
            "best_score": best_metrics["score"],
        })

    return {
        "baseline": BASELINE,
        "best_config": best_config,
        "best_metrics": best_metrics,
        "history": history,
        "evaluator_mode": (
            "mock" if evaluator is mock_evaluator else "real"
        ),
    }
