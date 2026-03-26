import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault('QT_IMAGEIO_MAXALLOC', '4096')

import pyqtgraph as pg

from moon_gen.surface_plotter import SurfacePlotter


parser = argparse.ArgumentParser()
parser.add_argument('FILE', nargs='?', type=str, help='input file')
parser.add_argument('-n', '--newest', action='store_true',
                    help='use the most recently modified file '
                    'in the module\'s `surfaces` folder')
args = parser.parse_args()


def get_most_recent_file() -> str:
    surfaces_dir = Path(__file__).with_name('surfaces')
    files = sorted(
        (
            p for p in surfaces_dir.iterdir()
            if p.is_file()
            and p.suffix == '.py'
            and p.name != '__init__.py'
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(files[0])


def get_project_dem_file() -> str | None:
    project_root = Path(__file__).resolve().parents[2]
    bundled_dem = project_root / 'img' / 'default_moon_dem.tif'
    if bundled_dem.exists():
        return str(bundled_dem)

    dem_files = sorted(
        (
            p for p in project_root.iterdir()
            if p.is_file() and p.suffix.casefold() in ('.tif', '.tiff')
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if dem_files:
        return str(dem_files[0])

    return None


def get_default_startup_file() -> str:
    surfaces_dir = Path(__file__).with_name('surfaces')
    default_file = surfaces_dir / 'full_2_rugged_static.py'

    if default_file.exists():
        return str(default_file)

    dem_file = get_project_dem_file()
    if dem_file is not None:
        return dem_file

    fallback_file = surfaces_dir / 'full_1_random.py'

    if fallback_file.exists():
        return str(fallback_file)

    return get_most_recent_file()


pg.mkQApp("GLSurfacePlot Example")
with SurfacePlotter() as w:
    if args.FILE is not None:
        w.plotSurfaceFromFile(args.FILE)
    elif args.newest:
        file = get_most_recent_file()
        w.plotSurfaceFromFile(file)
    else:
        w.plotSurfaceFromFile(get_default_startup_file(), prompt_user=False)

    sys.exit(pg.exec())
