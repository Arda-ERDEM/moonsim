from __future__ import annotations

import os

# -----------------------------------------------------------------------------
# GLOBAL SETTINGS
# -----------------------------------------------------------------------------
IMAGE_CANDIDATES = [
    "img/default_moon_dem.tif",
    "img/moon.png",
    "moon.png",
    "default_moon_dem.tif",
]

BATTERY_LEVEL = 85.0       # Percent (0-100)
TIME_PRIORITY = 0.4        # Priority (0-1) - High means rush
RISK_THRESHOLD = 0.65      # Global risk threshold for autonomous fallback
CONFIDENCE_THRESHOLD = 0.6  # If mean confidence is below this, prefer Safe mode
LOW_BATTERY_THRESHOLD = 30.0
TIME_PRIORITY_THRESHOLD = 0.7

# Grid processing settings
GRID_MAX_DIMENSION = 200   # Downsample large images for speed
ROUGHNESS_SIGMA = 1.0      # Local variance smoothing
OBSTACLE_PERCENTILE = 92   # Percentile for obstacle thresholding
OBSTACLE_DILATION_ITERS = 1
OBSTACLE_CLOSE_ITERS = 1
OBSTACLE_BLOCK_THRESHOLD = 0.8  # cells with obstacle value > this are blocked

# Hard mobility constraints (non-traversable, not just penalized)
# These represent rover physics limits and are enforced for all modes.
MAX_CLIMBABLE_SLOPE = 0.28
HARD_SLOPE_THRESHOLD = 0.78
HARD_PIT_DEPTH_THRESHOLD = 0.72
HARD_CRATER_SIGNAL_THRESHOLD = 0.58

CRATER_BOWL_SIGMA = 5.0    # Gaussian differential for bowl detection
CRATER_DARK_PERCENTILE = 15  # Dark region threshold
CONFIDENCE_CONTEXT_SIGMA = 2.0

# -----------------------------------------------------------------------------
# MODE WEIGHTS
# -----------------------------------------------------------------------------
# Normalized weights [0-10ish]. Higher = more avoidance/penalty.

# 1. SAFE MODE
# Prioritizes: Low slope, low risk, high confidence.
# Less sensitive to distance.
SAFE_WEIGHTS = {
    "slope": 8.0,
    "roughness": 4.0,
    "obstacle": 10.0,
    "crater": 8.0,
    "uncertainty": 5.0,
    "turn_penalty": 2.0,     # Moderate turn penalty
    "distance_weight": 1.2,  # Low distance penalty (willing to detour)
}

# 2. ECO MODE
# Prioritizes: Flat terrain, smoothness, few turns (energy efficiency).
ECO_WEIGHTS = {
    "slope": 4.0,
    # Very high roughness penalty (vibration/slip energy)
    "roughness": 8.0,
    "obstacle": 6.0,
    "crater": 3.0,
    "uncertainty": 1.0,
    "turn_penalty": 5.0,     # High turn penalty (turning costs energy)
    "distance_weight": 1.0,  # Minimize distance generally
}

# 3. FAST MODE
# Prioritizes: Shortest path, speed.
# Willing to take moderate risks if it saves time.
FAST_WEIGHTS = {
    "slope": 2.0,            # Can handle steeper slopes
    "roughness": 1.5,        # Can handle bumps
    "obstacle": 10.0,        # Still must avoid rocks
    "crater": 4.0,           # Avoid big holes
    "uncertainty": 0.5,      # Ignored if path is clear
    "turn_penalty": 0.5,     # Agile
    "distance_weight": 3.0,  # High distance penalty -> straight line preference
}

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------
VISUALIZE = True
VISUALIZE_SAVE = True
# Set True to pop up window (blocks execution until closed)
VISUALIZE_SHOW = False
VISUALIZE_DPI = 120
VISUALIZE_OUTPUT_DIR = "img"
OVERVIEW_FIGURE_NAME = "planning_overview.png"
SELECTION_FIGURE_NAME = "planning_selected_route.png"

# -----------------------------------------------------------------------------
# PATH SMOOTHING
# -----------------------------------------------------------------------------
SMOOTHING_ENABLED = True
LINE_OF_SIGHT_CHECK_STEP = 1  # Check every Nth node for LOS
start_fraction = (0.1, 0.1)  # Relative position
goal_fraction = (0.9, 0.8)   # Relative position

# Fraction of image dimensions
START_FRACTION = (0.1, 0.15)
GOAL_FRACTION = (0.85, 0.85)

# Mode Color Mapping for UI
MODE_COLORS = {
    "safe": "#2ecc71",
    "eco": "#f1c40f",
    "fast": "#e74c3c",
}
