from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    gaussian_filter,
)
import tifffile
from pyqtgraph.Qt import QtGui, QtCore

from moon_gen.planning import config
from moon_gen.planning.types import LayerMap


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    min_v = float(values[finite].min())
    max_v = float(values[finite].max())
    if np.isclose(max_v, min_v):
        out[finite] = 0.0
        return out
    out[finite] = (values[finite] - min_v) / (max_v - min_v)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _downsample_mean_2d(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image.shape
    largest = max(h, w)
    if largest <= max_dim:
        return image.astype(np.float32)

    scale = int(np.ceil(largest / max_dim))
    new_h = h // scale
    new_w = w // scale
    if new_h < 2 or new_w < 2:
        return image.astype(np.float32)

    trimmed = image[:new_h * scale, :new_w * scale]
    reshaped = trimmed.reshape(new_h, scale, new_w, scale)
    return reshaped.mean(axis=(1, 3)).astype(np.float32)


def _load_png_jpg(path: Path) -> np.ndarray:
    image = QtGui.QImage(str(path))
    if image.isNull():
        raise ValueError(f"image could not be decoded: {path}")

    image = image.convertToFormat(
        QtGui.QImage.Format.Format_Grayscale8,
        QtCore.Qt.ImageConversionFlag.MonoOnly,
    )

    w = image.width()
    h = image.height()
    ptr = image.constBits()
    if ptr is None:
        raise ValueError("failed to read image buffer")
    ptr.setsize(image.sizeInBytes())

    padding = image.bytesPerLine() - w
    values = np.asarray(ptr, dtype=np.uint8).reshape((h, w + padding))
    if padding:
        values = values[:, :-padding]
    return values.astype(np.float32)


def load_lunar_image(image_path: str | Path | None = None) -> tuple[np.ndarray, Path]:
    if image_path is not None:
        candidates = [Path(image_path)]
    else:
        candidates = [Path(p) for p in config.IMAGE_CANDIDATES]

    selected: Path | None = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            selected = candidate
            break

    if selected is None:
        raise FileNotFoundError("No lunar input image found in configured candidates")

    suffix = selected.suffix.casefold()
    if suffix in (".tif", ".tiff"):
        image = tifffile.imread(str(selected))
        if image.ndim > 2:
            image = image[..., 0]
        image = np.asarray(image, dtype=np.float32)
    else:
        # Fallback to PNG loader
        image = _load_png_jpg(selected)

    image = _downsample_mean_2d(image, config.GRID_MAX_DIMENSION)
    image = normalize01(image)
    return image, selected


def resolve_start_goal(
    shape: tuple[int, int],
    start_fraction: tuple[float, float],
    goal_fraction: tuple[float, float],
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape

    si = int(np.clip(round(start_fraction[0] * (h - 1)), 0, h - 1))
    sj = int(np.clip(round(start_fraction[1] * (w - 1)), 0, w - 1))
    gi = int(np.clip(round(goal_fraction[0] * (h - 1)), 0, h - 1))
    gj = int(np.clip(round(goal_fraction[1] * (w - 1)), 0, w - 1))

    return (si, sj), (gi, gj)


def compute_terrain_layers(image: np.ndarray) -> LayerMap:
    """
    1. Slope (gradient magnitude)
    2. Roughness (local variance)
    3. Obstacle (high slope/roughness)
    4. Crater (morphology + dark regions)
    5. Confidence (local contrast + threshold proximity)
    """
    if image.ndim != 2:
        raise ValueError("terrain image must be 2D")

    base = normalize01(image)

    # 1. Slope (approximate via gradient)
    gy, gx = np.gradient(base)
    slope = normalize01(np.hypot(gx, gy))

    # 2. Roughness (local variance = E[X^2] - E[X]^2)
    mean = gaussian_filter(base, sigma=config.ROUGHNESS_SIGMA)
    mean_sq = gaussian_filter(base**2, sigma=config.ROUGHNESS_SIGMA)
    roughness = normalize01(np.clip(mean_sq - mean**2, 0.0, None))

    # 3. Obstacle Risk (combination of high slope + roughness)
    obstacle_signal = 0.65 * slope + 0.35 * roughness
    threshold = np.percentile(obstacle_signal, config.OBSTACLE_PERCENTILE)
    obstacle_mask = obstacle_signal >= threshold
    obstacle_mask = binary_dilation(obstacle_mask, iterations=config.OBSTACLE_DILATION_ITERS)
    obstacle_mask = binary_closing(obstacle_mask, iterations=config.OBSTACLE_CLOSE_ITERS)
    obstacle = obstacle_mask.astype(np.float32)

    # 4. Crater Risk (dark regions + bowl shape)
    # Dark regions often shadows in craters
    dark_threshold = np.percentile(base, config.CRATER_DARK_PERCENTILE)
    dark_region = normalize01(np.clip((dark_threshold - base), 0.0, None))
    # Bowl shape: difference of gaussians or Laplacian-of-Gaussian proxy
    # Simple proxy: locally depressed vs surroundings
    bowl_response = normalize01(
        np.clip(gaussian_filter(base, sigma=config.CRATER_BOWL_SIGMA) - base, 0.0, None)
    )
    crater = normalize01(0.55 * dark_region + 0.45 * bowl_response)

    # 5. Confidence / Uncertainty
    # Ambiguous regions near obstacle threshold = lower confidence
    # Low local contrast = lower texture information = higher uncertainty?
    # Let's say: High contrast + Clear separation from threshold = High Confidence
    threshold_sep = normalize01(np.abs(obstacle_signal - threshold))
    local_contrast = normalize01(np.abs(base - gaussian_filter(base, sigma=config.CONFIDENCE_CONTEXT_SIGMA)))
    confidence = normalize01(0.65 * threshold_sep + 0.35 * local_contrast)
    uncertainty = normalize01(1.0 - confidence)

    return {
        "image": base,
        "slope": slope,
        "roughness": roughness,
        "obstacle": obstacle,
        "crater": crater,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "obstacle_signal": normalize01(obstacle_signal),
    }


def build_cost_map(layers: LayerMap, weights: dict[str, float]) -> np.ndarray:
    """
    Weighted sum of features based on mode config.
    """
    cell_cost = (
        weights["slope"] * layers["slope"]
        + weights["roughness"] * layers["roughness"]
        + weights["obstacle"] * layers["obstacle"]
        + weights["crater"] * layers["crater"]
        + weights["uncertainty"] * layers["uncertainty"]
    )
    # Normalize base cost map to [0, 1] before blockage logic
    cell_cost = normalize01(cell_cost)

    # Hard blockage logic: if obstacle > threshold, set cost extremely high
    huge_penalty = 1e6
    blocked = layers["obstacle"] >= config.OBSTACLE_BLOCK_THRESHOLD
    
    # Scale cost to avoid float precision issues with huge penalties in A*
    # 1.0 is min traversal cost per step
    cell_cost = 1.0 + 9.0 * cell_cost  # Map [0,1] -> [1, 10] range roughly
    cell_cost = np.where(blocked, huge_penalty, cell_cost)
    
    return cell_cost.astype(np.float32)


def build_all_cost_maps(layers: LayerMap) -> dict[str, np.ndarray]:
    return {
        "safe": build_cost_map(layers, config.SAFE_WEIGHTS),
        "eco": build_cost_map(layers, config.ECO_WEIGHTS),
        "fast": build_cost_map(layers, config.FAST_WEIGHTS),
    }
