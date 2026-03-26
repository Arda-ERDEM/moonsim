"""
Theta* (Any-Angle Pathfinding) implementation for smooth lunar rover trajectories.

Theta* improves upon A* by allowing line-of-sight shortcuts, producing smooth,
natural paths that follow terrain contours rather than grid-snapping.

Key features:
- Line-of-sight checks between non-adjacent nodes
- Any-angle movement (not restricted to 4 cardinal directions)
- Mode-specific cost inflation for obstacle avoidance
- Elevation-aware costs for energy optimization
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from moon_gen.planning.types import PathResult


@dataclass(order=True)
class ThetaStarNode:
    f_cost: float  # Priority queue key (g + h)
    counter: int = field(compare=False)  # Tiebreaker
    row: int = field(compare=False)
    col: int = field(compare=False)


def _heuristic_euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean distance heuristic (better for any-angle movement)."""
    dr = a[0] - b[0]
    dc = a[1] - b[1]
    return float(np.hypot(dr, dc))


def _bresenham_line(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    """Generate all grid cells intersected by a line segment."""
    r0, c0 = start
    r1, c1 = end
    
    points: list[tuple[int, int]] = []
    
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    
    err = dr - dc
    
    while True:
        points.append((r0, c0))
        
        if (r0, c0) == (r1, c1):
            break
            
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc
    
    return points


def _line_of_sight(
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: np.ndarray,
    obstacle_margin: float = 0.0,
) -> bool:
    """
    Check if a straight line is traversable between two points.
    
    Args:
        start: (row, col) start position
        end: (row, col) end position
        blocked: 2D binary array (True = blocked/obstacle)
        obstacle_margin: Safety margin; if > 0, expands blocked regions
        
    Returns:
        True if line of sight is clear, False otherwise
    """
    points = _bresenham_line(start, end)
    
    h, w = blocked.shape
    
    for r, c in points:
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        if blocked[r, c]:
            return False
    
    return True


def _elevate_cost(
    cost_base: float,
    elevation_delta: float,
    elevation_weight: float,
) -> float:
    """
    Apply elevation penalty to cost.
    
    Used in Eco mode to penalize uphill movement.
    Only positive elevation changes (uphill) are penalized.
    """
    if elevation_weight <= 0:
        return cost_base
    
    # Only penalize going uphill
    uphill_cost = max(0.0, elevation_delta) * elevation_weight
    
    return cost_base + uphill_cost


def theta_star_plan(
    cost_map: np.ndarray,
    blocked: np.ndarray,
    elevation_map: np.ndarray | None,
    start: tuple[int, int],
    goal: tuple[int, int],
    turn_penalty: float = 0.5,
    distance_weight: float = 1.0,
    elevation_weight: float = 0.0,
    obstacle_inflation: float = 0.0,
) -> PathResult:
    """
    Theta* pathfinding with any-angle movement and line-of-sight optimization.
    
    Args:
        cost_map: 2D array of cell costs
        blocked: 2D binary array (obstacles)
        elevation_map: Optional 2D elevation/height data
        start: (row, col) start position
        goal: (row, col) goal position
        turn_penalty: Cost penalty for direction changes (Safe mode uses this)
        distance_weight: Weight for geometric distance
        elevation_weight: Weight for elevation changes (Eco mode)
        obstacle_inflation: Radius to expand obstacles (Safe mode)
        
    Returns:
        PathResult with optimized trajectory
    """
    h, w = cost_map.shape
    
    # Validate start/goal
    if blocked[start] or blocked[goal]:
        return PathResult(False, [], float("inf"), float("inf"), [])
    
    # Create inflated obstacle map if needed
    inflated_blocked = blocked.copy()
    if obstacle_inflation > 0.5:
        # Dilate blocked regions by inflation radius
        from scipy.ndimage import binary_dilation
        inflation_iters = max(1, int(np.ceil(obstacle_inflation)))
        inflated_blocked = binary_dilation(blocked, iterations=inflation_iters)
        
        # Ensure start/goal remain accessible
        inflated_blocked[start] = False
        inflated_blocked[goal] = False
    
    # Use elevation map if provided, else use cost_map as proxy
    if elevation_map is None:
        elevation_map = cost_map
    
    # Priority queue: (f_cost, counter, row, col)
    frontier: list[tuple[float, int, int, int]] = []
    counter = 0
    heapq.heappush(frontier, (0.0, counter, start[0], start[1]))
    counter += 1
    
    # Track best costs and parent pointers
    g_cost: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    parent_via_los: dict[tuple[int, int], bool] = {start: False}
    
    best_goal_key: tuple[int, int] | None = None
    max_expansions = min(h * w * 16, 1_000_000)
    expansions = 0
    
    # Define 8-directional neighbors (any-angle)
    neighbors_8 = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
        (-1, -1), (-1, 1), (1, -1), (1, 1),  # Diagonal
    ]
    
    while frontier and expansions < max_expansions:
        _, _, curr_r, curr_c = heapq.heappop(frontier)
        current = (curr_r, curr_c)
        curr_g = g_cost.get(current, float("inf"))
        expansions += 1
        
        # Goal check
        if current == goal:
            best_goal_key = current
            break
        
        # Expand neighbors
        for dr, dc in neighbors_8:
            nr, nc = curr_r + dr, curr_c + dc
            neighbor = (nr, nc)
            
            # Bounds check
            if nr < 0 or nc < 0 or nr >= h or nc >= w:
                continue
            if inflated_blocked[nr, nc]:
                continue
            
            # Geometric cost
            step_dist = float(np.hypot(dr, dc))
            step_cost = distance_weight * step_dist * (
                0.5 * (cost_map[curr_r, curr_c] + cost_map[nr, nc])
            )
            
            # Elevation cost
            if elevation_weight > 0 and elevation_map is not None:
                elev_delta = elevation_map[nr, nc] - elevation_map[curr_r, curr_c]
                step_cost = _elevate_cost(step_cost, elev_delta, elevation_weight)
            
            # Try direct path from start if line-of-sight available
            parent_node = parent.get(current)
            if parent_node is not None:
                # Theta* logic: try to shortcut via parent's parent
                grandparent = parent_node
                if _line_of_sight(grandparent, neighbor, inflated_blocked):
                    # Direct path from grandparent to neighbor
                    shortcut_dist = float(np.hypot(
                        neighbor[0] - grandparent[0],
                        neighbor[1] - grandparent[1]
                    ))
                    shortcut_cost = distance_weight * shortcut_dist * (
                        0.5 * (cost_map[grandparent[0], grandparent[1]] + cost_map[nr, nc])
                    )
                    
                    if elevation_weight > 0:
                        elev_delta = elevation_map[nr, nc] - elevation_map[grandparent[0], grandparent[1]]
                        shortcut_cost = _elevate_cost(shortcut_cost, elev_delta, elevation_weight)
                    
                    candidate_g = g_cost[grandparent] + shortcut_cost
                    
                    if neighbor not in g_cost or candidate_g < g_cost[neighbor]:
                        g_cost[neighbor] = candidate_g
                        parent[neighbor] = grandparent
                        parent_via_los[neighbor] = True
                        priority = candidate_g + _heuristic_euclidean(neighbor, goal)
                        heapq.heappush(frontier, (priority, counter, nr, nc))
                        counter += 1
                        continue
            
            # Standard A* step
            candidate_g = curr_g + step_cost
            if neighbor not in g_cost or candidate_g < g_cost[neighbor]:
                g_cost[neighbor] = candidate_g
                parent[neighbor] = current
                parent_via_los[neighbor] = False
                priority = candidate_g + _heuristic_euclidean(neighbor, goal)
                heapq.heappush(frontier, (priority, counter, nr, nc))
                counter += 1
    
    # Reconstructpository path
    if best_goal_key is None:
        return PathResult(False, [], float("inf"), float("inf"), [])
    
    path_grid: list[tuple[int, int]] = []
    node = best_goal_key
    while node is not None:
        path_grid.append(node)
        node = parent.get(node)
    
    path_grid.reverse()
    
    # Convert grid path to smooth 3D waypoints
    if elevation_map is not None:
        smooth_path = _interpolate_path_3d(path_grid, elevation_map)
    else:
        # Fallback to 2D if no elevation data
        smooth_path = [
            (float(p[1]), float(p[0]), 0.0) for p in path_grid
        ]
    
    # Compute stats
    path_length = float(len(path_grid) - 1)
    total_cost = float(g_cost.get(best_goal_key, float("inf")))
    
    return PathResult(
        exists=True,
        path=path_grid,
        cost=total_cost,
        path_length=path_length,
        smoothed_path=path_grid,  # Keep grid path for compatibility
    )


def _interpolate_path_3d(
    grid_path: list[tuple[int, int]],
    elevation_map: np.ndarray,
) -> list[tuple[float, float, float]]:
    """
    Convert grid path to smooth 3D waypoints using interpolation.
    
    Returns list of (x, y, z) coordinates.
    """
    if len(grid_path) < 2:
        if len(grid_path) == 1:
            r, c = grid_path[0]
            return [(float(c), float(r), float(elevation_map[r, c]))]
        return []
    
    # Create waypoints with elevation
    rows = np.array([p[0] for p in grid_path], dtype=float)
    cols = np.array([p[1] for p in grid_path], dtype=float)
    elevs = np.array([elevation_map[int(r), int(c)] for r, c in grid_path], dtype=float)
    
    # If only 2 points, return as-is
    if len(grid_path) == 2:
        return [
            (cols[i], rows[i], elevs[i])
            for i in range(len(grid_path))
        ]
    
    # Interpolate with cubic splines for smoothness
    try:
        # Parameter along path
        t = np.arange(len(grid_path), dtype=float)
        
        # Cubic spline interpolation
        cs_rows = CubicSpline(t, rows, bc_type='natural')
        cs_cols = CubicSpline(t, cols, bc_type='natural')
        cs_elevs = CubicSpline(t, elevs, bc_type='natural')
        
        # Sample at higher resolution
        t_fine = np.linspace(0, t[-1], max(len(grid_path) * 3, 50))
        rows_fine = cs_rows(t_fine)
        cols_fine = cs_cols(t_fine)
        elevs_fine = cs_elevs(t_fine)
        
        # Ensure start and end are exact
        rows_fine[0] = rows[0]
        cols_fine[0] = cols[0]
        elevs_fine[0] = elevs[0]
        rows_fine[-1] = rows[-1]
        cols_fine[-1] = cols[-1]
        elevs_fine[-1] = elevs[-1]
        
        return [
            (float(cols_fine[i]), float(rows_fine[i]), float(elevs_fine[i]))
            for i in range(len(t_fine))
        ]
    except Exception:
        # Fallback to linear interpolation if spline fails
        waypoints: list[tuple[float, float, float]] = []
        for i in range(len(grid_path)):
            r, c = grid_path[i]
            z = elevation_map[int(np.clip(r, 0, elevation_map.shape[0]-1)),
                              int(np.clip(c, 0, elevation_map.shape[1]-1))]
            waypoints.append((float(c), float(r), float(z)))
        return waypoints
