from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np


@dataclass
class MissionConditions:
    """Inputs to the autonomous decision logic."""
    battery_level: float
    time_priority: float
    global_risk: float
    mean_uncertainty: float


@dataclass
class PathResult:
    """Result of a planning attempt."""
    exists: bool
    path: list[tuple[int, int]]
    cost: float
    path_length: float
    smoothed_path: list[tuple[int, int]]


class CandidateSummary(TypedDict):
    """Features of a generated path for decision making."""
    mode: str
    exists: bool
    path_cost: float
    path_length: float
    mean_risk: float
    mean_uncertainty: float
    turn_count: int


class DecisionExplanation(TypedDict):
    """Why a specific mode was chosen."""
    selected_mode: str | None
    available_candidates: dict[str, CandidateSummary]
    rejected_modes: dict[str, str]
    main_decision_factors: list[str]
    confidence_penalty_affected: bool
    fallback_used: bool


class LayerMap(TypedDict):
    """Container for computed terrain layers."""
    image: np.ndarray
    slope: np.ndarray
    roughness: np.ndarray
    obstacle: np.ndarray
    crater: np.ndarray
    confidence: np.ndarray
    uncertainty: np.ndarray
    obstacle_signal: np.ndarray
