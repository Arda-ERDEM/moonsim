from __future__ import annotations

import numpy as np

from moon_gen.planning import config
from moon_gen.planning.types import (
    CandidateSummary,
    DecisionExplanation,
    LayerMap,
    MissionConditions,
    PathResult,
)


def summarize_candidate(
    mode: str,
    result: PathResult,
    layers: LayerMap,
) -> CandidateSummary:
    if not result.exists:
        return {
            "mode": mode,
            "exists": False,
            "path_cost": float("inf"),
            "path_length": float("inf"),
            "mean_risk": float("inf"),
            "mean_uncertainty": float("inf"),
            "turn_count": 0,
        }

    # Calculate statistics along path
    path = result.smoothed_path if result.smoothed_path else result.path
    rr = np.array([p[0] for p in path], dtype=np.int64)
    cc = np.array([p[1] for p in path], dtype=np.int64)

    slope = layers["slope"][rr, cc]
    roughness = layers["roughness"][rr, cc]
    obstacle = layers["obstacle"][rr, cc]
    crater = layers["crater"][rr, cc]
    uncertainty = layers["uncertainty"][rr, cc]

    # Composite risk metric for the path
    mean_risk = float(np.mean(0.35 * slope + 0.20 * roughness + 0.25 * obstacle + 0.20 * crater))
    mean_uncertainty = float(np.mean(uncertainty))

    turns = 0
    if len(path) > 2:
        for i in range(2, len(path)):
            a = path[i - 2]
            b = path[i - 1]
            c = path[i]
            d1 = (b[0] - a[0], b[1] - a[1])
            d2 = (c[0] - b[0], c[1] - b[1])
            if d1 != d2:
                turns += 1

    return {
        "mode": mode,
        "exists": True,
        "path_cost": float(result.cost),
        "path_length": float(result.path_length),
        "mean_risk": mean_risk,
        "mean_uncertainty": mean_uncertainty,
        "turn_count": turns,
    }


def select_autonomous_mode(
    candidates: dict[str, CandidateSummary],
    mission: MissionConditions,
) -> tuple[str | None, DecisionExplanation]:
    """
    Selects the best mode (Safe/Eco/Fast) based on mission conditions.
    """
    valid_modes = [m for m, c in candidates.items() if c["exists"]]
    explanation: DecisionExplanation = {
        "selected_mode": None,
        "available_candidates": candidates,
        "rejected_modes": {},
        "main_decision_factors": [],
        "confidence_penalty_affected": False,
        "fallback_used": False,
    }

    if not valid_modes:
        explanation["main_decision_factors"].append("No valid path in safe/eco/fast.")
        return None, explanation

    # 1. Check constraints based on mission
    high_risk = mission.global_risk >= config.RISK_THRESHOLD
    low_confidence = mission.mean_uncertainty >= (1.0 - config.CONFIDENCE_THRESHOLD)
    low_battery = mission.battery_level <= config.LOW_BATTERY_THRESHOLD
    time_critical = mission.time_priority >= config.TIME_PRIORITY_THRESHOLD

    ranked_preferences: list[str] = []

    # Priority 1: Risk & Safety
    if high_risk:
        ranked_preferences.append("safe")
        explanation["main_decision_factors"].append(
            f"Global terrain risk high ({mission.global_risk:.3f} >= {config.RISK_THRESHOLD:.3f})."
        )

    # Priority 2: Confidence
    if low_confidence:
        if "safe" not in ranked_preferences:
            ranked_preferences.append("safe")
        explanation["confidence_penalty_affected"] = True
        explanation["main_decision_factors"].append(
            f"Uncertainty high ({mission.mean_uncertainty:.3f}), favoring conservative routing."
        )

    # Priority 3: Battery
    if low_battery:
        if "eco" not in ranked_preferences:
            ranked_preferences.append("eco")
        explanation["main_decision_factors"].append(
            f"Battery low ({mission.battery_level:.2f}), favoring energy-aware route."
        )

    # Priority 4: Time (Fast only if not risky/uncertain/low-battery)
    if time_critical and not high_risk and not low_confidence:
        if "fast" not in ranked_preferences:
            ranked_preferences.append("fast")
        explanation["main_decision_factors"].append(
            f"Time priority high ({mission.time_priority:.2f}) with acceptable risk/confidence."
        )

    # If no hard constraints, select based on Utility Score
    if not ranked_preferences:
        # Utility = weighted combination of path features
        # Lower is better
        utility_scores = {}
        for mode in valid_modes:
            candidate = candidates[mode]
            # normalize factors roughly for comparison
            # path cost is already weighted by mode, but we can re-weigh for decision
            utility_scores[mode] = (
                0.40 * (candidate["path_cost"] / 1000.0)
                + 0.25 * (candidate["path_length"] / 500.0)
                + 0.25 * candidate["mean_risk"] * 5.0
                + 0.10 * candidate["mean_uncertainty"] * 5.0
            )

        ranked_preferences = sorted(utility_scores, key=utility_scores.get)
        explanation["main_decision_factors"].append(
            "No hard mission constraint triggered; selected by composite utility score."
        )

    # Fill remainder
    ranked_preferences += [m for m in ("safe", "eco", "fast") if m not in ranked_preferences]

    # Select first valid
    selected_mode: str | None = None
    for mode in ranked_preferences:
        if mode in valid_modes:
            selected_mode = mode
            break

    if selected_mode is None:
        return None, explanation

    # Mark fallback if first choice wasn't valid
    top_preference = ranked_preferences[0]
    if selected_mode != top_preference:
        explanation["fallback_used"] = True

    # Generate rejection reasons
    for mode in ("safe", "eco", "fast"):
        if mode == selected_mode:
            continue

        if not candidates[mode]["exists"]:
            explanation["rejected_modes"][mode] = "No valid path found for this mode."
            continue

        if mode == "safe" and selected_mode != "safe":
            explanation["rejected_modes"][mode] = "Conservative route not required under current mission factors."
        elif mode == "eco" and selected_mode != "eco":
            explanation["rejected_modes"][mode] = "Energy optimization secondary to selected mission priority."
        elif mode == "fast" and selected_mode != "fast":
            explanation["rejected_modes"][mode] = "Direct route less preferred due to risk/uncertainty constraints."

    explanation["selected_mode"] = selected_mode
    return selected_mode, explanation
