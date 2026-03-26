import os
from contextlib import contextmanager

import numpy as np

from moon_gen.lib.utils import SurfaceType
from moon_gen.lib.craters import (
    make_crater,
    waste_gaussian,
    crater_density_mature,
)
from moon_gen.lib.heightmaps import (
    perlin_multiscale_grid,
    surface_psd_rough,
    surface_psd_nominal,
    surface_psd_smooth,
)

__depends__ = [
    "moon_gen.lib.utils",
    "moon_gen.lib.craters",
    "moon_gen.lib.heightmaps",
]

DEFAULT_SEED = 25032026
SEED_ENV_VAR = "MOON_GEN_SEED"
DEFAULT_GRID_SIZE = max(161, int(os.getenv("MOON_GEN_DEFAULT_GRID_SIZE", "257")))


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
        x_coords,
        y_coords,
        octaves=5,
        psd=surface_psd_nominal,
    )
    rough = perlin_multiscale_grid(
        x_coords + 57.0,
        y_coords - 23.0,
        octaves=4,
        psd=surface_psd_rough,
    )
    smooth = perlin_multiscale_grid(
        x_coords - 89.0,
        y_coords + 41.0,
        octaves=3,
        psd=surface_psd_smooth,
    )

    return 0.52 * nominal + 0.34 * rough + 0.14 * smooth


def surface(n: int | None = None, seed: int | None = None) -> SurfaceType:
    if n is None:
        n = DEFAULT_GRID_SIZE

    nx = ny = int(n)
    ny += 1

    area_x = area_y = 24.0
    x_coords = np.linspace(-area_x / 2, area_x / 2, nx)
    y_coords = np.linspace(-area_y / 2, area_y / 2, ny)

    resolved_seed = _resolve_seed(seed)
    with _seeded_numpy(resolved_seed):
        z_grid = _background_height(x_coords, y_coords)

        crater_distribution = crater_density_mature
        crater_distribution.d_min = 4 * np.ptp(x_coords) / len(x_coords)
        crater_count = max(1, int(0.60 * crater_distribution.number(x_coords, y_coords)))

        terrain_span = np.ptp(x_coords)
        for _ in range(4):
            crater_radius = (0.11 + 0.09 * np.random.random()) * terrain_span
            crater_center = (
                np.ptp(x_coords) * np.random.random() + np.min(x_coords),
                np.ptp(y_coords) * np.random.random() + np.min(y_coords),
            )
            z_grid = make_crater(x_coords, y_coords, z_grid, crater_radius, crater_center)

        epochs = 4
        for epoch in reversed(range(epochs)):
            for _ in range(crater_count // epochs):
                crater_diameter = crater_distribution.diameter(np.random.random())
                crater_center = (
                    np.ptp(x_coords) * np.random.random() + np.min(x_coords),
                    np.ptp(y_coords) * np.random.random() + np.min(y_coords),
                )
                z_grid = make_crater(
                    x_coords,
                    y_coords,
                    z_grid,
                    crater_diameter / 2,
                    crater_center,
                )

            if epoch > 0:
                z_grid = waste_gaussian(
                    z_grid,
                    np.ptp(x_coords) / len(x_coords),
                    0.24 * (epoch / epochs),
                )

        z_grid += np.random.normal(scale=1.0e-2 * np.ptp(x_coords) / len(x_coords), size=z_grid.shape)

        for _ in range(crater_count % epochs):
            crater_diameter = crater_distribution.diameter(np.random.random())
            crater_center = (
                np.ptp(x_coords) * np.random.random() + np.min(x_coords),
                np.ptp(y_coords) * np.random.random() + np.min(y_coords),
            )
            z_grid = make_crater(
                x_coords,
                y_coords,
                z_grid,
                crater_diameter / 2,
                crater_center,
            )

        z_grid = waste_gaussian(z_grid, np.ptp(x_coords) / len(x_coords), 0.12)

    xx_grid, yy_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
    slope_plane = (
        0.17 * (xx_grid - np.min(x_coords)) / np.ptp(x_coords)
        + 0.08 * (yy_grid - np.min(y_coords)) / np.ptp(y_coords)
    )
    undulation = 0.025 * np.sin(0.45 * xx_grid + 0.26 * yy_grid) + 0.018 * np.cos(0.92 * xx_grid - 0.37 * yy_grid)
    z_grid = z_grid + slope_plane + undulation

    z_grid -= z_grid.min()
    z_max = z_grid.max()
    if z_max > 0:
        z_grid /= z_max
        z_grid = np.power(z_grid, 1.05)

    return x_coords, y_coords, z_grid