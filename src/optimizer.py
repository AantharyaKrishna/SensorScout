
import random

from src.agents import (
    ChallengerAgent,
    DiagnosisAgent,
    DesignerAgent,
    StrategistAgent,
    validate_proposal,
)
from src.evaluator import evaluate_configuration


# Original sensor configuration
BASELINE = {
    "sensor_positions": [0, 3, 6],
    "sampling_rates_hz": [5, 10, 20],
}

POSITIONS = range(8)
RATES = (5, 10, 20)


def mock_evaluator(config, scenario_set="validation", seed=42):
    """
    Compatibility helper using the actual simulator.
    """
    return evaluate_configuration(
        config,
        scenario_set=scenario_set,
        seed=seed,
    )


def config_key(config):
    """
    Identify configurations regardless of sensor order.
    """
    return tuple(
        sorted(
            zip(
                config["sensor_positions"],
                config["sampling_rates_hz"],
            )
        )
    )


def random_configuration(rng):
    """Generate a valid random configuration."""
    return {
        "sensor_positions": rng.sample(list(POSITIONS), 3),
        "sampling_rates_hz": [
            rng.choice(RATES) for _ in range(3)
        ],
    }


def targeted_configurations(config, diagnosis):
    """
    Generate a small number of targeted experiments.

    Keep sensor positions unchanged initially so we do
    not accidentally lose coverage at other locations.
    """
    positions = config["sensor_positions"].copy()
    current_rates = config["sampling_rates_hz"]

    # Fast, demo-focused sampling-rate experiments
    rate_options = [
        (5, 20, 5),
        (10, 5, 10),
        (5, 5, 5),
        (10, 20, 5),
        (5, 20, 10),
    ]

    for rates in rate_options:
        candidate = {
            "sensor_positions": positions.copy(),
            "sampling_rates_hz": list(rates),
        }

        if config_key(candidate) != config_key(config):
            yield candidate

    # Explore one relocation toward the diagnosed weak spot
    weak_position = diagnosis["weak_position"]

    if weak_position not in positions:
        nearest_index = min(
            range(3),
            key=lambda i: abs(positions[i] - weak_position),
        )

        new_positions = positions.copy()
        new_positions[nearest_index] = weak_position

        yield {
            "sensor_positions": new_positions,
            "sampling_rates_hz": current_rates.copy(),
        }


def candidate_priority(current_metrics, candidate_metrics):
    """
    Prefer preserved recall, faster detection,
    lower cost, and higher evaluation score.
    """
    old_recall = current_metrics["fault_recall"]
    new_recall = candidate_metrics["fault_recall"]

    old_delay = current_metrics.get("mean_detection_delay_s")
    new_delay = candidate_metrics.get("mean_detection_delay_s")

    old_cost = current_metrics["sensor_cost_units"]
    new_cost = candidate_metrics["sensor_cost_units"]

    old_score = current_metrics["score"]
    new_score = candidate_metrics["score"]

    recall_preserved = new_recall >= old_recall

    delay_improved = (
        old_delay is not None
        and new_delay is not None
        and new_delay < old_delay
    )

    cost_improved = new_cost < old_cost
    score_improved = new_score > old_score

    both_improved = delay_improved and cost_improved

    return (
        int(recall_preserved),
        int(both_improved),
        int(score_improved),
        int(delay_improved),
        new_score,
        -new_cost,
    )


def choose_targeted_candidate(
    config,
    diagnosis,
    current_metrics,
    evaluator,
    tried_configs,
    seed,
    strategist,
):
    """
    Evaluate targeted experiments and return
    the strongest acceptable candidate.
    """
    best_candidate = None
    best_metrics = None
    best_priority = None

    seen = set()

    for candidate in targeted_configurations(config, diagnosis):
        key = config_key(candidate)

        if key in tried_configs or key in seen:
            continue

        seen.add(key)
        validate_proposal(candidate)

        metrics = evaluator(
            candidate,
            scenario_set="validation",
            seed=seed,
        )

        decision = strategist.decide(
            current_metrics,
            metrics,
        )

        if not decision["accepted"]:
            continue

        priority = candidate_priority(
            current_metrics,
            metrics,
        )

        if best_priority is None or priority > best_priority:
            best_candidate = candidate
            best_metrics = metrics
            best_priority = priority

    return best_candidate, best_metrics


def run_optimization(rounds=5, evaluator=None, seed=42):
    """
    SensorScout fast demo optimization.

    Challenger -> Diagnosis -> Designer -> Strategist

    Uses a local Designer to avoid slow external API calls.
    """
    if evaluator is None:
        evaluator = evaluate_configuration

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
        best_config,
        scenario_set="validation",
        seed=seed,
    )

    tried_configs = {config_key(best_config)}

    history = [
        {
            "round": 0,
            "config": best_config.copy(),
            "metrics": best_metrics,
            "best_score": best_metrics["score"],
            "accepted": True,
            "proposal_source": "baseline",
        }
    ]

    for round_number in range(1, rounds + 1):

        # 1. Challenger identifies the weakness.
        challenge = challenger.challenge(
            best_config,
            rng,
            metrics=best_metrics,
        )

        # 2. Diagnosis explains the weakness.
        diagnosis = diagnosis_agent.diagnose(
            best_config,
            challenge,
        )

        # 3. Local rule-based Designer proposes a change.
        proposal = designer.propose(
            best_config,
            diagnosis,
            rng,
        )

        validate_proposal(proposal)

        proposal_metrics = None

        if config_key(proposal) not in tried_configs:
            proposal_metrics = evaluator(
                proposal,
                scenario_set="validation",
                seed=seed,
            )

        # 4. Evaluate focused alternatives.
        targeted_proposal, targeted_metrics = (
            choose_targeted_candidate(
                best_config,
                diagnosis,
                best_metrics,
                evaluator,
                tried_configs,
                seed,
                strategist,
            )
        )

        candidates = []

        if proposal_metrics is not None:
            proposal_decision = strategist.decide(
                best_metrics,
                proposal_metrics,
            )

            if proposal_decision["accepted"]:
                candidates.append(
                    (
                        proposal,
                        proposal_metrics,
                        "rule_based_designer",
                    )
                )

        if targeted_proposal is not None:
            candidates.append(
                (
                    targeted_proposal,
                    targeted_metrics,
                    "targeted_search",
                )
            )

        # 5. Select the strongest accepted candidate.
        if candidates:
            candidate, candidate_metrics, source = max(
                candidates,
                key=lambda item: candidate_priority(
                    best_metrics,
                    item[1],
                ),
            )

        elif proposal_metrics is not None:
            # Record the rejected Designer experiment.
            candidate = proposal
            candidate_metrics = proposal_metrics
            source = "rule_based_designer"

        else:
            # Avoid repeated configurations.
            candidate = None

            for _ in range(100):
                alternative = random_configuration(rng)

                if config_key(alternative) not in tried_configs:
                    candidate = alternative
                    break

            if candidate is None:
                raise RuntimeError(
                    "Could not generate an untested configuration"
                )

            candidate_metrics = evaluator(
                candidate,
                scenario_set="validation",
                seed=seed,
            )

            source = "random_fallback"

        validate_proposal(candidate)
        tried_configs.add(config_key(candidate))

        # 6. Strategist accepts or rejects the experiment.
        decision = strategist.decide(
            best_metrics,
            candidate_metrics,
        )

        if decision["accepted"]:
            best_config = {
                "sensor_positions": (
                    candidate["sensor_positions"].copy()
                ),
                "sampling_rates_hz": (
                    candidate["sampling_rates_hz"].copy()
                ),
            }

            best_metrics = candidate_metrics

        # 7. Save the experiment.
        history.append(
            {
                "round": round_number,
                "challenge": challenge,
                "diagnosis": diagnosis,
                "config": candidate,
                "metrics": candidate_metrics,
                "decision": decision,
                "accepted": decision["accepted"],
                "best_score": best_metrics["score"],
                "proposal_source": source,
            }
        )

    return {
        "baseline": BASELINE,
        "best_config": best_config,
        "best_metrics": best_metrics,
        "history": history,
        "evaluator_mode": (
            "mock"
            if evaluator is mock_evaluator
            else "simulation"
        ),
    }
