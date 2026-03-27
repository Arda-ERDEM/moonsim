from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from moon_gen.planning import config
from moon_gen.planning.decision import (
    select_autonomous_mode,
    summarize_candidate,
)
from moon_gen.planning.maps import (
    build_all_cost_maps,
    compute_terrain_layers,
    load_lunar_image,
    resolve_start_goal,
)
from moon_gen.planning.planner import astar_plan
from moon_gen.planning.types import CandidateSummary, MissionConditions, PathResult
from moon_gen.planning.visualize import render_outputs
from moon_gen.planning.maps import normalize01


def _format_bool(flag: bool) -> str:
    return "yes" if flag else "no"


def _plan_mode_with_fallback(
    *,
    mode: str,
    cost_map: np.ndarray,
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    turn_penalty: float,
    distance_weight: float,
    elevation_map: np.ndarray,
    cell_size: float,
) -> PathResult:
    """Plan one mode, relaxing obstacle inflation if no path is found."""
    if mode == "safe":
        inflation_candidates = (2.5, 2.0, 1.5, 1.0, 0.5, 0.0)
        elevation_weight = 0.3
    elif mode == "eco":
        inflation_candidates = (0.5,)
        elevation_weight = 5.0
    else:  # fast
        inflation_candidates = (0.0,)
        elevation_weight = 0.1

    last_result: PathResult | None = None
    for inflation in inflation_candidates:
        result = astar_plan(
            cost_map=cost_map,
            blocked=blocked,
            start=start,
            goal=goal,
            turn_penalty=turn_penalty,
            distance_weight=distance_weight,
            elevation_map=elevation_map,
            elevation_weight=elevation_weight,
            obstacle_inflation=inflation,
            cell_size=cell_size,
        )
        last_result = result
        if result.exists:
            return result

    # Fallback return keeps previous behavior if all attempts fail.
    assert last_result is not None
    return last_result


def generate_all_candidates(
    image_input: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    block_threshold: float | None = None,
    cell_size: float = 1.0,
) -> dict[str, Any]:
    """Generate all three candidate routes without selecting one."""
    elevation_map = np.asarray(image_input, dtype=np.float32)
    finite = np.isfinite(elevation_map)
    if not finite.all():
        fill = float(np.nanmedian(elevation_map[finite])) if finite.any() else 0.0
        elevation_map = np.where(finite, elevation_map, fill)

    image = normalize01(elevation_map)
    layers = compute_terrain_layers(image)
    cost_maps = build_all_cost_maps(layers)

    blocked = layers["hard_block"] >= 0.5
    if block_threshold is not None:
        blocked = blocked | (layers["obstacle"] >= block_threshold)

    if 0 <= start[0] < blocked.shape[0] and 0 <= start[1] < blocked.shape[1]:
        blocked[start] = False
    if 0 <= goal[0] < blocked.shape[0] and 0 <= goal[1] < blocked.shape[1]:
        blocked[goal] = False

    mode_weights = {
        "safe": config.SAFE_WEIGHTS,
        "eco": config.ECO_WEIGHTS,
        "fast": config.FAST_WEIGHTS,
    }

    plans: dict[str, dict[str, Any]] = {}
    for mode in ("safe", "eco", "fast"):
        weights = mode_weights[mode]
        result = _plan_mode_with_fallback(
            mode=mode,
            cost_map=cost_maps[mode],
            blocked=blocked,
            start=start,
            goal=goal,
            turn_penalty=weights["turn_penalty"],
            distance_weight=weights["distance_weight"],
            elevation_map=elevation_map,
            cell_size=cell_size,
        )
        summary = summarize_candidate(mode, result, layers)
        plans[mode] = {"result": result, "summary": summary}

    global_risk = float(np.mean(0.35 * layers["slope"] + 0.20 * layers["roughness"] + 0.25 * layers["obstacle"] + 0.20 * layers["crater"]))
    mean_uncertainty = float(np.mean(layers["uncertainty"]))

    mission = MissionConditions(
        battery_level=config.BATTERY_LEVEL,
        time_priority=config.TIME_PRIORITY,
        global_risk=global_risk,
        mean_uncertainty=mean_uncertainty,
    )

    return {
        "plans": plans,
        "layers": layers,
        "mission": mission,
        "global_risk": global_risk,
        "mean_uncertainty": mean_uncertainty,
    }


def plan_mission(
    image_input: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    block_threshold: float | None = None,
    cell_size: float = 1.0,
) -> dict[str, Any]:
    """Programmatic planner entry point."""
    elevation_map = np.asarray(image_input, dtype=np.float32)
    finite = np.isfinite(elevation_map)
    if not finite.all():
        fill = float(np.nanmedian(elevation_map[finite])) if finite.any() else 0.0
        elevation_map = np.where(finite, elevation_map, fill)

    image = normalize01(elevation_map)
    layers = compute_terrain_layers(image)
    cost_maps = build_all_cost_maps(layers)

    blocked = layers["hard_block"] >= 0.5
    if block_threshold is not None:
        blocked = blocked | (layers["obstacle"] >= block_threshold)

    if 0 <= start[0] < blocked.shape[0] and 0 <= start[1] < blocked.shape[1]:
        blocked[start] = False
    if 0 <= goal[0] < blocked.shape[0] and 0 <= goal[1] < blocked.shape[1]:
        blocked[goal] = False

    mode_weights = {
        "safe": config.SAFE_WEIGHTS,
        "eco": config.ECO_WEIGHTS,
        "fast": config.FAST_WEIGHTS,
    }

    plans: dict[str, dict[str, Any]] = {}
    for mode in ("safe", "eco", "fast"):
        weights = mode_weights[mode]
        result = _plan_mode_with_fallback(
            mode=mode,
            cost_map=cost_maps[mode],
            blocked=blocked,
            start=start,
            goal=goal,
            turn_penalty=weights["turn_penalty"],
            distance_weight=weights["distance_weight"],
            elevation_map=elevation_map,
            cell_size=cell_size,
        )
        summary = summarize_candidate(mode, result, layers)
        plans[mode] = {"result": result, "summary": summary}

    global_risk = float(np.mean(0.35 * layers["slope"] + 0.20 * layers["roughness"] + 0.25 * layers["obstacle"] + 0.20 * layers["crater"]))
    mean_uncertainty = float(np.mean(layers["uncertainty"]))

    mission = MissionConditions(
        battery_level=config.BATTERY_LEVEL,
        time_priority=config.TIME_PRIORITY,
        global_risk=global_risk,
        mean_uncertainty=mean_uncertainty,
    )

    candidate_summaries: dict[str, CandidateSummary] = {mode: plans[mode]["summary"] for mode in plans}
    selected_mode, explanation = select_autonomous_mode(candidate_summaries, mission)

    return {
        "shape": tuple(int(v) for v in image.shape),
        "start": start,
        "goal": goal,
        "plans": plans,
        "selected_mode": selected_mode,
        "explanation": explanation,
        "layers": layers,
    }


def run(image_path: str | None = None) -> dict[str, Any]:
    """
    Main planning pipeline entry point.

    Future Extensibility Hooks:
    1. Replace gradient-based slope with NASA DEM slope in `maps.compute_terrain_layers`.
    2. Swap crater proxy with learned crater detector while preserving `layers['crater']` API.
    3. Replace static A* in `planner.astar_plan` with D* Lite for dynamic replanning.
    4. Use uncertainty map for online risk-aware replanning by updating cost maps in-loop.
    5. Bridge `run()` I/O to ROS topics/services or simulator APIs without changing core logic.
    """

    # 1. Load Image
    image, resolved_path = load_lunar_image(image_path)
    
    # 2. Compute Features
    layers = compute_terrain_layers(image)
    
    # 3. Build Cost Maps
    cost_maps = build_all_cost_maps(layers)

    # 4. Resolve Start/Goal
    start, goal = resolve_start_goal(
        image.shape,
        config.START_FRACTION,
        config.GOAL_FRACTION,
    )

    # Ensure start/goal are not on obstacles (simple fix)
    blocked = layers["hard_block"] >= 0.5
    blocked[start] = False
    blocked[goal] = False

    # 5. Run A* for each mode
    mode_weights = {
        "safe": config.SAFE_WEIGHTS,
        "eco": config.ECO_WEIGHTS,
        "fast": config.FAST_WEIGHTS,
    }

    plans: dict[str, dict] = {}
    for mode in ("safe", "eco", "fast"):
        weights = mode_weights[mode]
        result = astar_plan(
            cost_map=cost_maps[mode],
            blocked=blocked,
            start=start,
            goal=goal,
            turn_penalty=weights["turn_penalty"],
            distance_weight=weights["distance_weight"],
        )
        summary = summarize_candidate(mode, result, layers)
        plans[mode] = {
            "result": result,
            "summary": summary,
        }

    # 6. Autonomous Decision
    # Synthesize mission conditions from computed metrics + configured settings
    global_risk = float(np.mean(0.35 * layers["slope"] + 0.20 * layers["roughness"] + 0.25 * layers["obstacle"] + 0.20 * layers["crater"]))
    mean_uncertainty = float(np.mean(layers["uncertainty"]))

    mission = MissionConditions(
        battery_level=config.BATTERY_LEVEL,
        time_priority=config.TIME_PRIORITY,
        global_risk=global_risk,
        mean_uncertainty=mean_uncertainty,
    )

    candidate_summaries: dict[str, CandidateSummary] = {mode: plans[mode]["summary"] for mode in plans}
    selected_mode, explanation = select_autonomous_mode(candidate_summaries, mission)

    # 7. Terminal Output
    print("=" * 70)
    print("LUNAR MULTI-MODE PATH PLANNING")
    print("=" * 70)
    print(f"Image source: {resolved_path}")
    print(f"Image shape: {image.shape}")
    print(f"Start: {start} | Goal: {goal}")
    print(f"Slope source: gradient-based approximation from image intensity")
    print("-" * 70)

    for mode in ("safe", "eco", "fast"):
        summary = plans[mode]["summary"]
        print(f"{mode.upper()} path found: {_format_bool(summary['exists'])}")
        print(f"{mode.upper()} path length: {summary['path_length']:.2f}")
        print(f"{mode.upper()} path cost: {summary['path_cost']:.3f}")
        print(f"{mode.upper()} mean risk: {summary['mean_risk']:.3f}")
        print(f"{mode.upper()} mean uncertainty: {summary['mean_uncertainty']:.3f}")
        print("-")

    print("-" * 70)
    print(f"Autonomous selected: {selected_mode}")
    print("Main decision factors:")
    for factor in explanation["main_decision_factors"]:
        print(f"  - {factor}")

    print("Rejected modes and reasons:")
    for mode, reason in explanation["rejected_modes"].items():
        print(f"  - {mode}: {reason}")

    print(f"Confidence penalty affected decision: {_format_bool(explanation['confidence_penalty_affected'])}")
    print(f"Fallback used: {_format_bool(explanation['fallback_used'])}")

    # 8. Visualization
    vis_outputs = render_outputs(
        image=image,
        layers=layers,
        cost_maps=cost_maps,
        plans=plans,
        selected_mode=selected_mode,
        start=start,
        goal=goal,
    )

    if vis_outputs:
        print("Generated figures:")
        for key, value in vis_outputs.items():
            print(f"  - {key}: {value}")

    return {
        "image_path": str(resolved_path),
        "shape": tuple(int(v) for v in image.shape),
        "start": start,
        "goal": goal,
        "plans": plans,
        "selected_mode": selected_mode,
        "explanation": explanation,
        "visualization": vis_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-mode lunar rover path planning")
    parser.add_argument("--image", type=str, default=None, help="Optional path to lunar input image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.image)
