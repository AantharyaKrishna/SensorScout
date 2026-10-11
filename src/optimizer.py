
import random
from itertools import product
from collections import Counter

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
EVALUATIONS_PER_ROUND = 12


# -------------------------------------------------
# EVALUATOR
# -------------------------------------------------

def mock_evaluator(config, scenario_set="validation", seed=42):
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
        "sensor_positions": list(config["sensor_positions"]),
        "sampling_rates_hz": list(config["sampling_rates_hz"]),
    }


def config_key(config):
    """
    Equivalent sensor-position/rate pairs share a key.
    """
    return tuple(sorted(zip(
        config["sensor_positions"],
        config["sampling_rates_hz"],
    )))


def random_configuration(rng):
    return {
        "sensor_positions": rng.sample(list(POSITIONS), 3),
        "sampling_rates_hz": [
            rng.choice(RATES) for _ in range(3)
        ],
    }


# -------------------------------------------------
# EXPERIMENT GENERATION
# -------------------------------------------------

def targeted_configurations(config, diagnosis):
    """
    Generate rate changes and sensor relocations.
    """
    positions = list(config["sensor_positions"])
    rates = list(config["sampling_rates_hz"])
    seen = {config_key(config)}

    for combination in product(RATES, repeat=3):
        candidate = {
            "sensor_positions": positions.copy(),
            "sampling_rates_hz": list(combination),
        }
        key = config_key(candidate)

        if key not in seen:
            seen.add(key)
            yield candidate

    weak_position = diagnosis.get("weak_position")

    if (
        weak_position is not None
        and weak_position in POSITIONS
        and weak_position not in positions
    ):
        for index in range(3):
            new_positions = positions.copy()
            new_positions[index] = weak_position

            if len(set(new_positions)) != 3:
                continue

            for combination in product(RATES, repeat=3):
                candidate = {
                    "sensor_positions": new_positions.copy(),
                    "sampling_rates_hz": list(combination),
                }
                key = config_key(candidate)

                if key not in seen:
                    seen.add(key)
                    yield candidate


def feedback_strategy(rejection_counts, round_number):
    """
    Use previous rejection reasons to select
    the next search approach.
    """
    if not rejection_counts:
        return "balanced"

    cost_failures = sum(
        count
        for reason, count in rejection_counts.items()
        if "cost" in reason.lower()
    )

    delay_failures = sum(
        count
        for reason, count in rejection_counts.items()
        if "delay" in reason.lower()
        or "slower" in reason.lower()
    )

    recall_failures = sum(
        count
        for reason, count in rejection_counts.items()
        if "recall" in reason.lower()
        or "missed" in reason.lower()
    )

    if cost_failures > max(delay_failures, recall_failures):
        return "cost_aware"

    if recall_failures > max(cost_failures, delay_failures):
        return "recall_aware"

    if delay_failures > 0:
        return "delay_focused"

    return "explore" if round_number % 2 == 0 else "balanced"


def candidate_stream(
    config,
    diagnosis,
    strategy,
    rng,
):
    """
    Different strategies change which candidates
    are evaluated first.
    """
    candidates = list(targeted_configurations(
        config, diagnosis
    ))

    current_positions = set(config["sensor_positions"])

    if strategy == "cost_aware":
        candidates.sort(
            key=lambda c: (
                sum(c["sampling_rates_hz"]),
                len(
                    set(c["sensor_positions"])
                    ^ current_positions
                ),
            )
        )

    elif strategy == "delay_focused":
        candidates.sort(
            key=lambda c: (
                -sum(c["sampling_rates_hz"]),
                len(
                    set(c["sensor_positions"])
                    ^ current_positions
                ),
            )
        )

    elif strategy == "recall_aware":
        candidates.sort(
            key=lambda c: (
                -len(
                    set(c["sensor_positions"])
                    ^ current_positions
                ),
                -sum(c["sampling_rates_hz"]),
            )
        )

    elif strategy == "explore":
        rng.shuffle(candidates)

    else:
        # Alternate rate tuning and relocation.
        rate_changes = [
            c for c in candidates
            if c["sensor_positions"]
            == config["sensor_positions"]
        ]
        relocations = [
            c for c in candidates
            if c["sensor_positions"]
            != config["sensor_positions"]
        ]

        candidates = []
        while rate_changes or relocations:
            if rate_changes:
                candidates.append(rate_changes.pop(0))
            if relocations:
                candidates.append(relocations.pop(0))

    return candidates


# -------------------------------------------------
# BLIND-SPOT ANALYSIS
# -------------------------------------------------

def location_statistics(metrics, position):
    scenarios = [
        s for s in metrics.get("scenario_results", [])
        if s.get("fault_position") == position
        and s.get("fault_type") != "healthy"
    ]

    missed = sum(
        not s.get("detected", False)
        for s in scenarios
    )

    delays = [
        s["detection_delay_s"]
        for s in scenarios
        if s.get("detected", False)
        and s.get("detection_delay_s") is not None
    ]

    return {
        "missed_faults": missed,
        "average_delay_s": (
            sum(delays) / len(delays)
            if delays else None
        ),
    }


# -------------------------------------------------
# ACCEPTANCE POLICY
# -------------------------------------------------

def meets_improvement_policy(
    current_metrics,
    candidate_metrics,
    weak_position=None,
):
    if (
        candidate_metrics["fault_recall"] + 1e-9
        < current_metrics["fault_recall"]
    ):
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
        return False, "Average detection delay did not improve."

    if (
        candidate_metrics["sensor_cost_units"]
        > current_metrics["sensor_cost_units"]
        * MAX_COST_INCREASE
    ):
        return False, "Sensor cost exceeds the allowed increase."

    if weak_position is not None:
        old_location = location_statistics(
            current_metrics, weak_position
        )
        new_location = location_statistics(
            candidate_metrics, weak_position
        )

        if (
            new_location["missed_faults"]
            > old_location["missed_faults"]
        ):
            return False, "The blind spot has more missed faults."

        old_weak = old_location["average_delay_s"]
        new_weak = new_location["average_delay_s"]

        if (
            old_weak is not None
            and (
                new_weak is None
                or new_weak > old_weak + 1e-9
            )
        ):
            return False, (
                "Detection at the diagnosed blind spot became slower."
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
    old_delay = current_metrics.get(
        "mean_detection_delay_s"
    )
    new_delay = candidate_metrics.get(
        "mean_detection_delay_s"
    )

    overall_gain = (
        old_delay - new_delay
        if old_delay is not None and new_delay is not None
        else float("-inf")
    )

    weak_gain = 0.0

    if weak_position is not None:
        old_location = location_statistics(
            current_metrics, weak_position
        )
        new_location = location_statistics(
            candidate_metrics, weak_position
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
# TARGETED SEARCH (COMPATIBILITY)
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
    best_candidate = None
    best_metrics = None
    best_priority = None

    for candidate in targeted_configurations(config, diagnosis):
        key = config_key(candidate)

        if key in tried_configs:
            continue

        validate_proposal(candidate)
        tried_configs.add(key)

        metrics = evaluator(
            candidate,
            scenario_set="validation",
            seed=seed,
        )

        policy_ok, _ = meets_improvement_policy(
            current_metrics,
            metrics,
            diagnosis.get("weak_position"),
        )

        if not policy_ok:
            continue

        decision = strategist.decide(current_metrics, metrics)

        if not decision["accepted"]:
            continue

        priority = candidate_priority(
            current_metrics,
            metrics,
            diagnosis.get("weak_position"),
        )

        if best_priority is None or priority > best_priority:
            best_candidate = candidate
            best_metrics = metrics
            best_priority = priority

    return best_candidate, best_metrics


# -------------------------------------------------
# FEEDBACK-DRIVEN OPTIMIZATION
# -------------------------------------------------

def run_optimization(
    rounds=5,
    evaluator=None,
    seed=42,
    evaluations_per_round=EVALUATIONS_PER_ROUND,
):
    """
    Feedback-driven, simulation-guided four-agent loop.

    Fast demo mode does not call an external LLM.
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

    tried_configs = {config_key(best_config)}
    experiment_log = []
    rejection_counts = Counter()

    history = [{
        "round": 0,
        "config": copy_config(best_config),
        "metrics": best_metrics,
        "best_score": best_metrics["score"],
        "accepted": True,
        "proposal_source": "baseline",
        "evaluations": 1,
        "skipped_duplicates": 0,
        "strategy": "baseline",
        "feedback": "Initial sensor configuration.",
    }]

    for round_number in range(1, rounds + 1):
        challenge = challenger.challenge(
            best_config,
            rng,
            metrics=best_metrics,
        )

        diagnosis = diagnosis_agent.diagnose(
            best_config,
            challenge,
        )

        weak_position = diagnosis.get("weak_position")
        strategy = feedback_strategy(
            rejection_counts,
            round_number,
        )

        proposal = designer.propose(
            best_config,
            diagnosis,
            rng,
        )
        validate_proposal(proposal)

        candidates = [
            ("rule_based_designer", proposal)
        ]

        candidates.extend(
            ("targeted_search", candidate)
            for candidate in candidate_stream(
                best_config,
                diagnosis,
                strategy,
                rng,
            )
        )

        # Include exploratory configurations when
        # feedback calls for broader investigation.
        if strategy == "explore":
            candidates.extend(
                ("random_exploration", random_configuration(rng))
                for _ in range(evaluations_per_round * 3)
            )

        round_experiments = []
        skipped_duplicates = 0
        accepted_candidates = []
        evaluated_count = 0

        for source, candidate in candidates:
            key = config_key(candidate)

            if key in tried_configs:
                skipped_duplicates += 1
                continue

            if evaluated_count >= evaluations_per_round:
                break

            validate_proposal(candidate)
            tried_configs.add(key)

            metrics = evaluator(
                candidate,
                scenario_set="validation",
                seed=seed,
            )
            evaluated_count += 1

            policy_ok, policy_reason = meets_improvement_policy(
                best_metrics,
                metrics,
                weak_position,
            )

            strategist_decision = strategist.decide(
                best_metrics,
                metrics,
            )

            accepted = (
                policy_ok
                and strategist_decision["accepted"]
            )

            reason = (
                policy_reason
                if not policy_ok
                else strategist_decision.get(
                    "reason",
                    "Candidate evaluated.",
                )
            )

            experiment = {
                "round": round_number,
                "config": copy_config(candidate),
                "metrics": metrics,
                "source": source,
                "accepted": accepted,
                "reason": reason,
                "strategy": strategy,
            }

            round_experiments.append(experiment)
            experiment_log.append(experiment)

            if accepted:
                accepted_candidates.append(experiment)
            else:
                rejection_counts[reason] += 1

        if accepted_candidates:
            selected = max(
                accepted_candidates,
                key=lambda item: candidate_priority(
                    best_metrics,
                    item["metrics"],
                    weak_position,
                ),
            )

            candidate = selected["config"]
            candidate_metrics = selected["metrics"]
            source = selected["source"]
            accepted = True

            policy_ok, policy_reason = meets_improvement_policy(
                best_metrics,
                candidate_metrics,
                weak_position,
            )

            decision = {
                "accepted": True,
                "reason": "Accepted: " + policy_reason,
            }

            best_config = copy_config(candidate)
            best_metrics = candidate_metrics

        elif round_experiments:
            # Display the best rejected experiment
            # without changing the accepted configuration.
            selected = max(
                round_experiments,
                key=lambda item: candidate_priority(
                    best_metrics,
                    item["metrics"],
                    weak_position,
                ),
            )

            candidate = selected["config"]
            candidate_metrics = selected["metrics"]
            source = selected["source"]
            accepted = False

            decision = {
                "accepted": False,
                "reason": "Rejected: " + selected["reason"],
            }

        else:
            # All available candidates were previously tested.
            candidate = copy_config(best_config)
            candidate_metrics = best_metrics
            source = "no_new_candidate"
            accepted = False

            decision = {
                "accepted": False,
                "reason": (
                    "No new configuration was evaluated "
                    "in this round."
                ),
            }

        feedback = (
            f"Strategy: {strategy}. "
            f"Evaluated {evaluated_count} new configurations; "
            f"skipped {skipped_duplicates} duplicates. "
            f"Accepted improvement: {accepted}."
        )

        history.append({
            "round": round_number,
            "challenge": challenge,
            "diagnosis": diagnosis,
            "config": copy_config(candidate),
            "metrics": candidate_metrics,
            "decision": decision,
            "accepted": accepted,
            "best_score": best_metrics["score"],
            "proposal_source": source,
            "evaluations": evaluated_count,
            "skipped_duplicates": skipped_duplicates,
            "strategy": strategy,
            "feedback": feedback,
            "rejection_reasons": dict(
                Counter(
                    exp["reason"]
                    for exp in round_experiments
                    if not exp["accepted"]
                )
            ),
        })

    return {
        "baseline": copy_config(BASELINE),
        "best_config": copy_config(best_config),
        "best_metrics": best_metrics,
        "history": history,
        "experiment_log": experiment_log,
        "total_evaluations": 1 + len(experiment_log),
        "unique_configurations_tested": len(tried_configs),
        "rejection_summary": dict(rejection_counts),
        "evaluator_mode": (
            "mock"
            if evaluator is mock_evaluator
            else "simulation"
        ),
    }
