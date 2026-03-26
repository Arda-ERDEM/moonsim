"""
Pathfinding module: Theta* (Any-Angle Pathfinding) with mode-specific optimizations.

Replaces grid-locked A* with smooth trajectory generation for lunar rover navigation.
Supports three distinct planning modes:
- SAFE: Large obstacle inflation for maximum clearance
- ECO: Heavy elevation penalties to minimize energy consumption
- FAST: Minimal inflation with maximum line-of-sight shortcuts for speed
"""

from __future__ import annotations

import numpy as np

from moon_gen.planning.types import PathResult
from moon_gen.planning.thetastar import theta_star_plan


def astar_plan(
    cost_map: np.ndarray,
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    turn_penalty: float = 0.5,
    distance_weight: float = 1.0,
    elevation_map: np.ndarray | None = None,
    elevation_weight: float = 0.0,
    obstacle_inflation: float = 0.0,
) -> PathResult:
    """
    Plan a path using Theta* (Any-Angle Pathfinding).
    
    This function maintains the original interface while using the advanced Theta*
    algorithm for smooth trajectory generation.
    
    Args:
        cost_map: 2D array of traversal costs
        blocked: 2D binary array indicating obstacles
        start: (row, col) start position
        goal: (row, col) goal position
        turn_penalty: Cost for direction changes (used in turn logic)
        distance_weight: Weight for geometric distance in cost calculation
        elevation_map: Optional 2D elevation/height data for slope-aware planning
        elevation_weight: Weight for elevation changes (for Eco mode)
        obstacle_inflation: Radius to dilate obstacles (for Safe mode)
        
    Returns:
        PathResult containing the optimized path and metadata
    """
    # Delegate to Theta* implementation
    return theta_star_plan(
        cost_map=cost_map,
        blocked=blocked,
        elevation_map=elevation_map,
        start=start,
        goal=goal,
        turn_penalty=turn_penalty,
        distance_weight=distance_weight,
        elevation_weight=elevation_weight,
        obstacle_inflation=obstacle_inflation,
    )
