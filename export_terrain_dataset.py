from pathlib import Path

import numpy as np

from moon_gen.surfaces.full_2_rugged_static import surface


def main() -> None:
    x, y, z = surface()
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), z.ravel())).astype(np.float32)

    out = Path("exports")
    out.mkdir(exist_ok=True)

    np.save(out / "x.npy", x)
    np.save(out / "y.npy", y)
    np.save(out / "z.npy", z)
    np.savez_compressed(out / "terrain_xyz_grid.npz", x=x, y=y, z=z)

    np.save(out / "terrain_xyz_points.npy", points)
    np.savetxt(
        out / "terrain_xyz_points.csv",
        points,
        delimiter=",",
        header="x,y,z",
        comments="",
        fmt="%.6f",
    )

    dx = float(np.ptp(x) / (len(x) - 1))
    dy = float(np.ptp(y) / (len(y) - 1))

    print(f"points={points.shape[0]}")
    print(f"grid_shape={z.shape}")
    print(f"dx={dx}")
    print(f"dy={dy}")
    print(f"z_min={float(np.nanmin(z))}")
    print(f"z_max={float(np.nanmax(z))}")
    print(f"output_dir={out.resolve()}")


if __name__ == "__main__":
    main()
