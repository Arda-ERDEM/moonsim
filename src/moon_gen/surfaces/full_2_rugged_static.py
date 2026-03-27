import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, zoom

from moon_gen.lib.craters import crater_density_mature, make_crater, waste_gaussian
from moon_gen.lib.heightmaps import perlin_multiscale_grid, surface_psd_nominal, surface_psd_rough
from moon_gen.lib.utils import SurfaceType

__depends__ = [
    "moon_gen.lib.utils",
    "moon_gen.lib.craters",
    "moon_gen.lib.heightmaps",
]

DEFAULT_SEED = 25032026
SEED_ENV_VAR = "MOON_GEN_SEED"
DEFAULT_GRID_SIZE = max(
    257, int(os.getenv("MOON_GEN_DEFAULT_GRID_SIZE", "513")))
DEFAULT_AREA_SIZE = float(os.getenv("MOON_GEN_AREA_SIZE", "2400.0"))
DEFAULT_HEIGHT_RANGE_METERS = float(os.getenv("MOON_GEN_Z_RANGE", "180.0"))
DEM_PATH_ENV_VAR = "MOON_GEN_DEM_PATH"


@contextmanager
def _seeded_numpy(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)

    env_value = os.getenv(SEED_ENV_VAR)
    if env_value is not None:
        try:
            return int(env_value)
        except ValueError:
            pass

    return DEFAULT_SEED


def _background_height(x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    nominal = perlin_multiscale_grid(
        x_coords, y_coords, octaves=5, psd=surface_psd_nominal)
    rough = perlin_multiscale_grid(
        x_coords + 41.0, y_coords - 17.0, octaves=4, psd=surface_psd_rough)
    return 0.70 * nominal + 0.30 * rough


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_dem_path() -> Path:
    env_path = os.getenv(DEM_PATH_ENV_VAR)
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p

    return _project_root() / "img" / "default_moon_dem.tif"


def _robust_normalize(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float64)

    valid = values[finite]
    lo = float(np.percentile(valid, low))
    hi = float(np.percentile(valid, high))

    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        out = np.zeros_like(values, dtype=np.float64)
        out[finite] = 0.5
        return out

    clipped = np.clip(values, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    normalized = np.nan_to_num(normalized, nan=0.5, posinf=1.0, neginf=0.0)
    return np.asarray(normalized, dtype=np.float64)


def _load_dem_patch(nx: int, ny: int) -> np.ndarray | None:
    dem_path = _resolve_dem_path()
    if not dem_path.exists():
        return None

    dem = tifffile.imread(str(dem_path))
    if dem.ndim > 2:
        dem = dem[..., 0]
    if dem.ndim != 2:
        return None

    dem = np.asarray(dem, dtype=np.float64)
    finite = np.isfinite(dem)
    if not finite.any():
        return None
    if not finite.all():
        dem = np.where(finite, dem, np.nanmedian(dem[finite]))

    # Use a center-biased crop to keep major lunar macro-features and avoid nodata borders.
    h, w = dem.shape
    patch_h = max(nx * 2, min(h, int(0.72 * h)))
    patch_w = max(ny * 2, min(w, int(0.72 * w)))
    y0 = max(0, (h - patch_h) // 2)
    x0 = max(0, (w - patch_w) // 2)
    dem = dem[y0:y0 + patch_h, x0:x0 + patch_w]

    dem = _robust_normalize(dem, low=1.0, high=99.0)

    zoom_y = nx / dem.shape[0]
    zoom_x = ny / dem.shape[1]
    dem = zoom(dem, (zoom_y, zoom_x), order=1)
    dem = np.asarray(dem[:nx, :ny], dtype=np.float64)

    return _robust_normalize(dem, low=0.5, high=99.5)


def _build_dem_relief(dem_patch: np.ndarray, nx: int) -> np.ndarray:
    # Multi-scale decomposition keeps long slopes and crater rims at the same time.
    macro = gaussian_filter(dem_patch, sigma=max(1.2, nx / 40.0))
    meso = dem_patch - gaussian_filter(dem_patch, sigma=max(0.8, nx / 120.0))
    micro = dem_patch - gaussian_filter(dem_patch, sigma=max(0.6, nx / 220.0))

    relief = 0.62 * macro + 0.30 * meso + 0.08 * micro
    return _robust_normalize(relief, low=0.5, high=99.5)


def _add_hills_and_ridges(x_coords: np.ndarray, y_coords: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    xx_grid, yy_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
    span = max(np.ptp(x_coords), np.ptp(y_coords))

    result = z_grid.copy()
    # Low-count broad hills to avoid artificial bumpy noise.
    for _ in range(6):
        cx = float(np.random.uniform(np.min(x_coords), np.max(x_coords)))
        cy = float(np.random.uniform(np.min(y_coords), np.max(y_coords)))
        sigma = float(np.random.uniform(0.06, 0.14) * span)
        amp = float(np.random.uniform(0.02, 0.06))
        result += amp * np.exp(-((xx_grid - cx) ** 2 +
                               (yy_grid - cy) ** 2) / (2.0 * sigma * sigma))

    ridge = 0.012 * np.sin(0.005 * xx_grid + 0.0035 * yy_grid)
    ridge += 0.010 * np.cos(0.0042 * xx_grid - 0.0031 * yy_grid)
    result += ridge

    return result


def _add_crater_field(x_coords: np.ndarray, y_coords: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    terrain_span = float(np.ptp(x_coords))
    distribution = crater_density_mature
    distribution.d_min = 3.2 * terrain_span / len(x_coords)

    crater_count = max(12, int(0.32 * distribution.number(x_coords, y_coords)))
    result = z_grid

    for _ in range(5):
        radius = float((0.07 + 0.06 * np.random.random()) * terrain_span)
        center = (
            float(np.random.uniform(np.min(x_coords), np.max(x_coords))),
            float(np.random.uniform(np.min(y_coords), np.max(y_coords))),
        )
        result = make_crater(x_coords, y_coords, result, radius, center)

    epochs = 4
    for epoch in reversed(range(epochs)):
        for _ in range(crater_count // epochs):
            diameter = float(distribution.diameter(float(np.random.random())))
            center = (
                float(np.random.uniform(np.min(x_coords), np.max(x_coords))),
                float(np.random.uniform(np.min(y_coords), np.max(y_coords))),
            )
            result = make_crater(x_coords, y_coords,
                                 result, 0.5 * diameter, center)

        if epoch > 0:
            result = waste_gaussian(
                result, terrain_span / len(x_coords), 0.15 * (epoch / epochs))

    return result


def _finalize_height(z_grid: np.ndarray) -> np.ndarray:
    z_grid = _robust_normalize(z_grid, low=0.8, high=99.2)
    z_grid = np.power(z_grid, 1.06)
    z_grid *= DEFAULT_HEIGHT_RANGE_METERS
    z_grid -= float(np.nanmin(z_grid))
    return z_grid


def surface(n: int | None = None, seed: int | None = None) -> SurfaceType:
    if n is None:
        n = DEFAULT_GRID_SIZE

    nx = ny = int(n)

    area_x = area_y = DEFAULT_AREA_SIZE
    x_coords = np.linspace(-area_x / 2, area_x / 2, nx)
    y_coords = np.linspace(-area_y / 2, area_y / 2, ny)

    resolved_seed = _resolve_seed(seed)
    with _seeded_numpy(resolved_seed):
        dem_patch = _load_dem_patch(nx, ny)
        if dem_patch is None:
            dem_relief = _robust_normalize(_background_height(
                x_coords, y_coords), low=1.0, high=99.0)
        else:
            dem_relief = _build_dem_relief(dem_patch, nx)

        z_grid = dem_relief
        z_grid = _add_crater_field(x_coords, y_coords, z_grid)
        z_grid = _add_hills_and_ridges(x_coords, y_coords, z_grid)
        z_grid += 0.12 * _background_height(x_coords, y_coords)
        z_grid = waste_gaussian(z_grid, np.ptp(x_coords) / len(x_coords), 0.08)

        z_grid = _finalize_height(z_grid)

    return x_coords, y_coords, np.asarray(z_grid, dtype=np.float64)
