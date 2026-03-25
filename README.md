# Moon Surface Generator

[![Testing](https://github.com/mbiselx/moon_gen/actions/workflows/python-testing.yml/badge.svg)](https://github.com/mbiselx/moon_gen/actions/workflows/python-testing.yml)
[![Formatting](https://github.com/mbiselx/moon_gen/actions/workflows/python-lint_and_format.yml/badge.svg)](https://github.com/mbiselx/moon_gen/actions/workflows/python-lint_and_format.yml) 



The following is a small app allowing one to visualize a surface (such as an elevation model), defined in a python file. 
The purpose is to aid in the development of algorithms to generate nice-looking lunar-like surfaces for lunar rover surface operation simulations. 


## Preview

| ![surface 1](img/Screenshot(1).png) | ![surface 2](img/Screenshot(2).png) | ![surface 3](img/Screenshot(3).png) |
| --- | --- | --- |
| a somewhat older surface, with randomly placed craters | a surface with a fresh crater | a surface with a randomly generated height map |


## Running

After installation, the project can be run as module:
```bash
python -m moon_gen
```

By default, the GUI loads the bundled TIFF DEM at `img/default_moon_dem.tif`,
so a fresh clone starts with the same TIFF-based view.

For weaker laptops, you can cap preview resolution (while still loading large
input TIFF files) with an environment variable:
```powershell
$env:MOON_GEN_MAX_HEIGHTMAP_DIMENSION=512; python -m moon_gen
```
```bash
MOON_GEN_MAX_HEIGHTMAP_DIMENSION=512 python -m moon_gen
```

If you have a large external DEM (for example
`Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif`), place it in the project root
and pass it explicitly:
```bash
python -m moon_gen Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif
```

Use `Ctrl+S` in the viewer to export as either:
- 16-bit TIFF (`.tif`, preferred for DEM quality)
- 8-bit PNG (`.png`, smaller files)


## TODOs

This project is still a work in progress. As such, there are a number of features I would still like to implement. Some are listed below : 
- [ ] better crater and ejecta modelling (more scientifically accurate shapes)
- [ ] better mass wasting (using a dffusion equation, rather than smoothing)
- [ ] use real DEMs for base terrain
- [x] non-crater procedural base terrain
- [ ] generate albedo maps based on crater placement
- [x] export generated surfaces for use in Gazebo
- [ ] export generated surfaces to a standard DEM format, for use in Gazebo