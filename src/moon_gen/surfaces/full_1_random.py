import numpy as np

from moon_gen.lib.utils import SurfaceType
from moon_gen.lib.craters import (  # noqa: F401
    make_crater, waste_gaussian,
    crater_density_fresh, crater_density_young,
    crater_density_mature, crater_density_old,
)
from moon_gen.lib.heightmaps import (  # noqa: F401
    perlin_multiscale_grid,
    surface_psd_rough, surface_psd_nominal, surface_psd_smooth,
)

__depends__ = [
    "moon_gen.lib.utils",
    "moon_gen.lib.craters",
    "moon_gen.lib.heightmaps"
]


def parametric_surface(
        x, y,
    epochs=5,
    ocatves=7,
    psd=surface_psd_nominal,
    distribution=crater_density_mature,
    weathering_strength=0.20,
    relief_gain=1.10,
    flatland_strength=0.30,
    crater_fraction=0.95,
    target_relief=0.95,
    large_crater_count=4,
):

    print("generating background")
    z_nominal = perlin_multiscale_grid(
        x,
        y,
        octaves=ocatves,
        psd=psd,
    )

    z_rough = perlin_multiscale_grid(
        x + 37.0,
        y + 19.0,
        octaves=max(3, ocatves - 1),
        psd=surface_psd_rough,
    )

    z_smooth = perlin_multiscale_grid(
        x - 71.0,
        y + 43.0,
        octaves=max(3, ocatves - 2),
        psd=surface_psd_smooth,
    )

    z = 0.55 * z_nominal + 0.30 * z_rough + 0.15 * z_smooth

    distribution.d_min = 4*np.ptp(x)/len(x)
    nb_craters = max(1, int(crater_fraction * distribution.number(x, y)))
    print(f"generating {nb_craters} craters")

    terrain_span = np.ptp(x)
    for _ in range(large_crater_count):
        large_radius = (0.12 + 0.08*np.random.random()) * terrain_span
        center = (
            np.ptp(x) * np.random.random() + np.min(x),
            np.ptp(y) * np.random.random() + np.min(y)
        )
        z = make_crater(x, y, z, large_radius, center)

    # create older craters first and weather them
    for w in reversed(range(epochs)):
        for _ in range(nb_craters//epochs):
            d = distribution.diameter(np.random.random())
            center = (np.ptp(x) * np.random.random() + np.min(x),
                      np.ptp(y) * np.random.random() + np.min(y))
            z = make_crater(x, y, z, d/2, center)

        if w > 0:
            z = waste_gaussian(
                z,
                np.ptp(x)/len(x),
                weathering_strength * (w/epochs)
            )

    # apply micro-meteorite impacts
    z += np.random.normal(scale=9e-3*np.ptp(x)/len(x), size=z.shape)

    # create the last remaining craters unweathered
    for _ in range(nb_craters % epochs):
        d = distribution.diameter(np.random.random())
        center = (np.ptp(x) * np.random.random() + np.min(x),
                  np.ptp(y) * np.random.random() + np.min(y))
        z = make_crater(x, y, z, d/2, center)

    z = waste_gaussian(z, np.ptp(x)/len(x), 0.12)
    z *= relief_gain

    z -= z.min()
    z_max = z.max()
    if z_max > 0:
        z *= target_relief / z_max
        z_norm = z / target_relief
        flat_weight = np.clip((0.45 - z_norm) / 0.45, 0.0, 1.0)
        z *= (1.0 - flatland_strength * flat_weight)
        z[z < 0.02*target_relief] = 0.0
        z -= z.min()

    print("done")

    return z


# def surface(n=1025) -> SurfaceType:
# def surface(n=513) -> SurfaceType:
# def surface(n=129) -> SurfaceType:
def surface(n=None) -> SurfaceType:
    '''
    create a random lunar surface using:
     - mutliscale perlin grid with a lunar highland PSD
     - randomly placed craters
     - gaussian-blur style mass wasting
    '''
    if n is None:
        n = int(np.sqrt(300_000))

    nx = ny = int(n)
    ny += 1
    ax = ay = 20
    epochs = 5

    cx, cy = 100*np.random.random((2,))
    x = np.linspace(-ax/2, ax/2, nx)
    y = np.linspace(-ay/2, ay/2, ny)

    z = parametric_surface(
        x+cx,
        y+cy,
        epochs,
        ocatves=7,
        psd=surface_psd_nominal,
        distribution=crater_density_mature,
        weathering_strength=0.20,
        relief_gain=1.10,
        flatland_strength=0.30,
        crater_fraction=0.95,
        target_relief=0.95,
        large_crater_count=4,
    )

    return x, y, z
