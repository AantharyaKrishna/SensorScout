
import random
from itertools import product

from src.agents import (
    ChallengerAgent,
    DiagnosisAgent,
    DesignerAgent,
    StrategistAgent,
    validate_proposal,
)
from src.evaluator import evaluate_configuration


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

BASELINE = {
    "sensor_positions": [0, 3, 6],
    "sampling_rates_hz": [5, 10, 20],
}

POSITIONS = range(8)
RATES = (5, 10, 20)

MAX_COST_INCREASE = 1.15
MIN_DELAY_IMPROVEMENT = 0.01


# -------------------------------------------------
# EVALUATOR
# -------------------------------------------------

def mock_evaluator(config, scenario_set="validation", seed=42):
    """
    Compatibility wrapper for the real simulator.
    """
    return evaluate_configuration(
        config,
        scenario_set=scenario_set,
        seed=seed,
    )


# -------------------------------------------------
# CONFIGURATION HELPERS
# -------------------------------------------------

def copy_config(config):
    return {
        "sensor_positions": config["sensor_positions"].copy(),
        "sampling_rates_hz": config["sampling_rates_hz"].copy(),
    }


def config_key(config):
    """
    Treat equivalent sensor-position/rate pairs
    as the same configuration.
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
    """
    Generate a valid random sensor configuration.
    """
    return {
        "sensor_positions": rng.sample(list(POSITIONS), 3),
        "sampling_rates_hz": [
            rng.choice(RATES) for _ in range(3)
        ],
    }


# -------------------------------------------------
# TARGETED EXPERIMENT GENERATION
# -------------------------------------------------

def targeted_configurations(config, diagnosis):
    """
    Generate candidate experiments.

    1. Test all 27 sampling-rate combinations
       at the current sensor positions.

    2. Test relocating each sensor toward the
       diagnosed weak position.

    These are simulated experiments, not physical
    changes to real equipment.
    """
    positions = config["sensor_positions"].copy()
    rates = config["sampling_rates_hz"].copy()

    seen = {config_key(config)}

    # Experiment A: sampling-rate combinations
    for combination in product(RATES, repeat=3):
        candidate = {
            "sensor_positions": positions.copy(),
            "sampling_rates_hz": list(combination),
        }

        key = config_key(candidate)

        if key not in seen:
            seen.add(key)
            yield candidate

    # Experiment B: sensor relocation
    weak_position = diagnosis.get("weak_position")

    if (
        weak_position is not None
        and weak_position not in positions
    ):
        for index in range(3):
            new_positions = positions.copy()
            new_positions[index] = weak_position

            if len(set(new_positions)) != 3:
                continue

            for combination in [
                rates,
                [10, 10, 10],
                [20, 10, 10],
                [10, 20, 10],
                [10, 10, 20],
            ]:
                candidate = {
                    "sensor_positions": new_positions.copy(),
                    "sampling_rates_hz": list(combination),
                }

                key = config_key(candidate)

                if key not in seen:
                    seen.add(key)
                    yield candidate


# -------------------------------------------------
# BLIND-SPOT ANALYSIS
# -------------------------------------------------

def location_statistics(metrics, position):
    """
    Calculate detection performance for a fault
    originating at one machine position.
    """
    scenarios = [
        scenario
        for scenario in metrics.get("scenario_results", [])
        if scenario.get("fault_position") == position
        and scenario.get("fault_type") != "healthy"
    ]

    missed = sum(
        1 for scenario in scenarios
        if not scenario.get("detected", False)
    )

    delays = [
        scenario["detection_delay_s"]
        for scenario in scenarios
        if scenario.get("detected", False)
        and scenario.get("detection_delay_s") is not None
    ]

    average_delay = (
        sum(delays) / len(delays)
        if delays
        else None
    )

    return {
        "missed_faults": missed,
        "average_delay_s": average_delay,
    }


# -------------------------------------------------
# ACCEPTANCE POLICY
# -------------------------------------------------

def meets_improvement_policy(
    current_metrics,
    candidate_metrics,
    weak_position=None,
):
    """
    Accept only when:

    - Fault recall is preserved.
    - Average detection delay improves.
    - Cost is within the allowed increase.
    - The diagnosed weak position does not become
      slower or lose previously detected faults.
    """
    old_recall = current_metrics["fault_recall"]
    new_recall = candidate_metrics["fault_recall"]

    if new_recall + 1e-9 < old_recall:
        return False, "Fault recall decreased."

    old_delay = current_metrics.get(
        "mean_detection_delay_s"
    )
    new_delay = candidate_metrics.get(
        "mean_detection_delay_s"
    )

    if old_delay is None or new_delay is None:
        return False, "Detection delay is unavailable."

    if new_delay >= old_delay - MIN_DELAY_IMPROVEMENT:
        return False, (
            "Average detection delay did not improve."
        )

    old_cost = current_metrics["sensor_cost_units"]
    new_cost = candidate_metrics["sensor_cost_units"]

    if new_cost > old_cost * MAX_COST_INCREASE:
        return False, (
            "Sensor cost exceeds the allowed increase."
        )

    if weak_position is not None:
        old_location = location_statistics(
            current_metrics,
            weak_position,
        )
        new_location = location_statistics(
            candidate_metrics,
            weak_position,
        )

        if (
            new_location["missed_faults"]
            > old_location["missed_faults"]
        ):
            return False, (
                "The blind spot has more missed faults."
            )

        old_weak_delay = old_location["average_delay_s"]
        new_weak_delay = new_location["average_delay_s"]

        if (
            old_weak_delay is not None
            and (
                new_weak_delay is None
                or new_weak_delay
                > old_weak_delay + 1e-9
            )
        ):
            return False, (
                "Detection at the diagnosed blind spot "
                "became slower."
            )

    return True, (
        "Detection improved while preserving recall "
        "and maintaining the cost limit."
    )


# -------------------------------------------------
# CANDIDATE RANKING
# -------------------------------------------------

def candidate_priority(
    current_metrics,
    candidate_metrics,
    weak_position=None,
):
    """
    Rank acceptable candidates.

    Prefer improving the diagnosed blind spot,
    then overall detection delay, then score/cost.
    """
    old_delay = current_metrics.get(
        "mean_detection_delay_s"
    )
    new_delay = candidate_metrics.get(
        "mean_detection_delay_s"
    )

    overall_gain = (
        old_delay - new_delay
        if old_delay is not None
        and new_delay is not None
        else float("-inf")
    )

    weak_gain = 0.0

    if weak_position is not None:
        old_location = location_statistics(
            current_metrics,
            weak_position,
        )
        new_location = location_statistics(
            candidate_metrics,
            weak_position,
        )

        old_weak = old_location["average_delay_s"]
        new_weak = new_location["average_delay_s"]

        if old_weak is not None and new_weak is not None:
            weak_gain = old_weak - new_weak

    return (
        weak_gain,
        overall_gain,
        candidate_metrics["score"],
        -candidate_metrics["sensor_cost_units"],
    )


# -------------------------------------------------
# TARGETED SEARCH
# -------------------------------------------------

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
    Evaluate targeted candidates and return the
    strongest candidate satisfying both the
    improvement policy and Strategist.
    """
    best_candidate = None
    best_metrics = None
    best_priority = None

    seen = set()

    weak_position = diagnosis.get("weak_position")

    for candidate in targeted_configurations(
        config,
        diagnosis,
    ):
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

        policy_ok, _ = meets_improvement_policy(
            current_metrics,
            metrics,
            weak_position,
        )

        if not policy_ok:
            continue

        decision = strategist.decide(
            current_metrics,
            metrics,
        )

        if not decision["accepted"]:
            continue

        priority = candidate_priority(
            current_metrics,
            metrics,
            weak_position,
        )

        if (
            best_priority is None
            or priority > best_priority
        ):
            best_candidate = candidate
            best_metrics = metrics
            best_priority = priority

    return best_candidate, best_metrics


# -------------------------------------------------
# MAIN OPTIMIZATION LOOP
# -------------------------------------------------

def run_optimization(rounds=5, evaluator=None, seed=42):
    """
    SensorScout's four-agent optimization process.

    Challenger:
        Identifies a monitoring weakness.

    Diagnosis:
        Explains the weakness.

    Designer:
        Proposes a sensor change.

    Targeted search:
        Tests alternatives using simulation.

    Strategist:
        Decides whether to accept an experiment.

    Fast mode: no external LLM calls.
    """
    if evaluator is None:
        evaluator = evaluate_configuration

    rng = random.Random(seed)

    challenger = ChallengerAgent()
    diagnosis_agent = DiagnosisAgent()
    designer = DesignerAgent()
    strategist = StrategistAgent()

    best_config = copy_config(BASELINE)

    best_metrics = evaluator(
        best_config,
        scenario_set="validation",
        seed=seed,
    )

    tried_configs = {
        config_key(best_config)
    }

    history = [
        {
            "round": 0,
            "config": copy_config(best_config),
            "metrics": best_metrics,
            "best_score": best_metrics["score"],
            "accepted": True,
            "proposal_source": "baseline",
        }
    ]

    for round_number in range(1, rounds + 1):

        # -----------------------------------------
        # 1. CHALLENGER
        # -----------------------------------------

        challenge = challenger.challenge(
            best_config,
            rng,
            metrics=best_metrics,
        )

        # -----------------------------------------
        # 2. DIAGNOSIS
        # -----------------------------------------

        diagnosis = diagnosis_agent.diagnose(
            best_config,
            challenge,
        )

        weak_position = diagnosis.get(
            "weak_position"
        )

        # -----------------------------------------
        # 3. DESIGNER
        # -----------------------------------------

        proposal = designer.propose(
            best_config,
            diagnosis,
            rng,
        )

        validate_proposal(proposal)

        candidates = []

        proposal_key = config_key(proposal)

        if proposal_key not in tried_configs:
            proposal_metrics = evaluator(
                proposal,
                scenario_set="validation",
                seed=seed,
            )

            policy_ok, _ = meets_improvement_policy(
                best_metrics,
                proposal_metrics,
                weak_position,
            )

            strategist_decision = strategist.decide(
                best_metrics,
                proposal_metrics,
            )

            if (
                policy_ok
                and strategist_decision["accepted"]
            ):
                candidates.append(
                    (
                        proposal,
                        proposal_metrics,
                        "rule_based_designer",
                    )
                )

        # -----------------------------------------
        # 4. TARGETED SEARCH
        # -----------------------------------------

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

        if targeted_proposal is not None:
            candidates.append(
                (
                    targeted_proposal,
                    targeted_metrics,
                    "targeted_search",
                )
            )

        # -----------------------------------------
        # 5. SELECT CANDIDATE
        # -----------------------------------------

        if candidates:
            candidate, candidate_metrics, source = max(
                candidates,
                key=lambda item: candidate_priority(
                    best_metrics,
                    item[1],
                    weak_position,
                ),
            )

        else:
            # No acceptable improvement was found.
            # Record an attempted Designer proposal,
            # without changing the best configuration.
            candidate = proposal
            source = "rule_based_designer"

            candidate_metrics = evaluator(
                candidate,
                scenario_set="validation",
                seed=seed,
            )

        validate_proposal(candidate)

        tried_configs.add(
            config_key(candidate)
        )

        # -----------------------------------------
        # 6. STRATEGIST DECISION
        # -----------------------------------------

        strategist_decision = strategist.decide(
            best_metrics,
            candidate_metrics,
        )

        policy_ok, policy_reason = (
            meets_improvement_policy(
                best_metrics,
                candidate_metrics,
                weak_position,
            )
        )

        accepted = (
            strategist_decision["accepted"]
            and policy_ok
        )

        decision = dict(strategist_decision)

        decision["accepted"] = accepted

        if accepted:
            decision["reason"] = (
                "Accepted: "
                + policy_reason
            )

            best_config = copy_config(candidate)
            best_metrics = candidate_metrics

        else:
            decision["reason"] = (
                "Rejected: "
                + policy_reason
                if not policy_ok
                else (
                    "Rejected by Strategist: "
                    + strategist_decision.get(
                        "reason",
                        "Candidate did not meet acceptance criteria.",
                    )
                )
            )

        # -----------------------------------------
        # 7. SAVE AGENT HISTORY
        # -----------------------------------------

        history.append(
            {
                "round": round_number,
                "challenge": challenge,
                "diagnosis": diagnosis,
                "config": copy_config(candidate),
                "metrics": candidate_metrics,
                "decision": decision,
                "accepted": accepted,
                "best_score": best_metrics["score"],
                "proposal_source": source,
            }
        )

    return {
        "baseline": copy_config(BASELINE),
        "best_config": copy_config(best_config),
        "best_metrics": best_metrics,
        "history": history,
        "evaluator_mode": (
            "mock"
            if evaluator is mock_evaluator
            else "simulation"
        ),
    }
