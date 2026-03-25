#!/usr/bin/env python
'''
A widget for interactively plotting surfaces.
'''

import os
import sys
import logging
import importlib
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

import pyqtgraph.opengl as gl
if TYPE_CHECKING:
    from PyQt6 import QtCore, QtGui, QtWidgets
else:
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from moon_gen.lib.utils import SurfaceFunctionType, SurfaceType


class SurfacePlotter(QtWidgets.QFrame):

    _MAX_HEIGHTMAP_DIMENSION = max(
        64,
        int(os.environ.get('MOON_GEN_MAX_HEIGHTMAP_DIMENSION', '768'))
    )
    _HEIGHTMAP_CLIP_LOW_PERCENTILE = 1.0
    _HEIGHTMAP_CLIP_HIGH_PERCENTILE = 99.0
    _HEIGHTMAP_GAMMA = 0.9

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._module: ModuleType | None = None

        self.vw = gl.GLViewWidget(self)

        self.grid = gl.GLGridItem()
        self.vw.addItem(self.grid)

        self._surfaceData: SurfaceType = (
            np.arange(0, 2.),
            np.arange(0, 2.),
            np.zeros((2, 2))
        )

        x, y, z = self._surfaceData
        self.surf = gl.GLSurfacePlotItem(
            x=x,
            y=y,
            z=z,
            shader='shaded'
        )
        self.vw.addItem(self.surf)

        self.setAcceptDrops(True)

        self._reloadAction = QtGui.QAction('&Regenerate surface', self)
        self._reloadAction.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self._reloadAction.setShortcut(QtGui.QKeySequence('Ctrl+R'))
        self._reloadAction.triggered.connect(self.reloadSurface)
        self.addAction(self._reloadAction)

        self._gridVizAction = QtGui.QAction('&Toggle grid on/off', self)
        self._gridVizAction.setCheckable(True)
        self._gridVizAction.setChecked(self.grid.visible())
        self._gridVizAction.setShortcut(QtGui.QKeySequence('Ctrl+G'))
        self._gridVizAction.toggled.connect(self.grid.setVisible)
        self.addAction(self._gridVizAction)

        self._shaderAction = QtGui.QAction('&Shader on/off', self)
        self._shaderAction.setCheckable(True)
        self._shaderAction.setChecked(self.surf.shader().name == 'normalColor')
        self._shaderAction.toggled.connect(self.toggleShader)
        self.addAction(self._shaderAction)

        self._sep = QtGui.QAction(self)
        self._sep.setSeparator(True)
        self.addAction(self._sep)

        self._exportAction = QtGui.QAction('&Export surface', self)
        self._exportAction.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_ToolBarHorizontalExtensionButton
        ))
        self._exportAction.setShortcut(QtGui.QKeySequence('Ctrl+S'))
        self._exportAction.triggered.connect(self.exportSurface)
        self.addAction(self._exportAction)

        self.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.vw)
        self.layout().setContentsMargins(*4*[0])

        # error message
        self._err_message = QtWidgets.QErrorMessage(self)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(100, 100)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(500, 300)

    def __enter__(self) -> 'SurfacePlotter':
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.show()
        return self

    def __exit__(self, *args):
        pass

    def toggleShader(self, active: bool):
        self.surf.setShader('normalColor' if active else 'shaded')

    def dragEnterEvent(self, a0: QtGui.QDragEnterEvent) -> None:
        '''accept any .py files dragged into this widget'''
        if a0.mimeData().hasText():
            txt = a0.mimeData().text()
            self._logger.info(txt)

            if os.path.isfile(txt.removeprefix('file:///')):
                return a0.accept()

        return a0.ignore()

    def dropEvent(self, a0: QtGui.QDropEvent) -> None:
        '''
        Run the files dropped onto this widget and plot the resulting surface.
        '''
        filename = a0.mimeData().text().removeprefix('file:///')
        self.plotSurfaceFromFile(filename)

    def _downsample_heightmap(self, values: np.ndarray) -> np.ndarray:
        height, width = values.shape
        max_size = self._MAX_HEIGHTMAP_DIMENSION
        largest_dim = max(height, width)

        if largest_dim <= max_size:
            return np.asarray(values, dtype=np.float32)

        scale = largest_dim / max_size
        target_height = max(2, int(round(height / scale)))
        target_width = max(2, int(round(width / scale)))

        row_idx = np.linspace(0, height - 1, target_height, dtype=np.int64)
        col_idx = np.linspace(0, width - 1, target_width, dtype=np.int64)

        return np.asarray(values[np.ix_(row_idx, col_idx)], dtype=np.float32)

    def _normalize_heightmap(self, values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError('heightmap does not contain finite values')

        finite_values = values[finite]
        lo = np.percentile(finite_values, self._HEIGHTMAP_CLIP_LOW_PERCENTILE)
        hi = np.percentile(finite_values, self._HEIGHTMAP_CLIP_HIGH_PERCENTILE)

        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
            normalized = np.zeros_like(values, dtype=np.float32)
            normalized[finite] = 0.5
            return normalized

        clipped = np.clip(values, lo, hi)
        normalized = (clipped - lo) / (hi - lo)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

        if not np.isclose(self._HEIGHTMAP_GAMMA, 1.0):
            normalized = np.power(normalized, self._HEIGHTMAP_GAMMA)

        return normalized.astype(np.float32)

    def plotSurfaceFromFile(self, filename: str, *, prompt_user: bool = True):

        if filename.casefold().endswith('.py'):
            self.plotSurfaceFromModule(filename)
        elif filename.casefold().endswith(
            ('.png', '.jpg', '.jepg', '.tif', '.tiff')
        ):
            self.plotSurfaceFromHeightmap(filename, prompt_user=prompt_user)
        else:
            ermsg = f"unsupported filetype `{filename.rsplit('.', 1)[-1]}`"
            self._err_message.showMessage(ermsg, 'warning')
            self._logger.warning(ermsg)

    def plotSurfaceFromModule(self, filename: str):
        '''plot the surface defined in a python file'''
        modulepath = os.path.dirname(filename)
        modulename = os.path.basename(filename)[:-3]

        try:
            # try to reload the module if it exists
            module = importlib.reload(sys.modules[modulename])
        except KeyError:
            # otherwise load it for the first time
            sys.path.append(modulepath)
            module = importlib.import_module(modulename)

        # try to retrieve a surface from the module
        try:
            surface_func: SurfaceFunctionType = module.surface
            self._surfaceData = surface_func()
            x, y, z = self._surfaceData
            self.surf.setData(x=x, y=y, z=z)
            self._module = module
            if surface_func.__doc__ is not None:
                self.setToolTip(surface_func.__doc__)
            else:
                self.setToolTip('')
        except Exception as e:
            ermsg = f"failed to plot surface from module ({e})"
            self._err_message.showMessage(ermsg, 'error')
            self._logger.error(ermsg)
            self._logger.exception(e)

    def plotSurfaceFromHeightmap(self, filename: str, *, prompt_user: bool = True):
        '''plot the surface defined in a heightmap image file'''
        try:
            file_suffix = os.path.splitext(filename)[1].casefold()

            if file_suffix in ('.tif', '.tiff'):
                import tifffile

                height_data = tifffile.memmap(filename)
                if height_data.ndim > 2:
                    height_data = height_data[..., 0]
                if height_data.ndim != 2:
                    raise ValueError('unsupported TIFF dimensions')
                height_data = self._downsample_heightmap(height_data)
                normalized = self._normalize_heightmap(height_data)
                h, w = normalized.shape
            else:
                surface_image = QtGui.QImage(filename)
                if surface_image.isNull():
                    raise ValueError('image could not be decoded by Qt')

                surface_image = surface_image.convertToFormat(
                    QtGui.QImage.Format.Format_Grayscale8,
                    QtCore.Qt.ImageConversionFlag.MonoOnly
                )

                w = surface_image.width()
                h = surface_image.height()
                ptr = surface_image.constBits()
                if ptr is None:
                    raise ValueError('image buffer unavailable')
                ptr.setsize(surface_image.sizeInBytes())

                # QImage has some end-of-line padding, so that each line
                # is word-aligned
                padding = surface_image.bytesPerLine() - w
                zz = np.asarray(ptr, dtype=np.uint8).reshape((h, w+padding))
                if padding:
                    zz = zz[:, :-padding]
                zz = self._downsample_heightmap(zz)
                normalized = zz / 255.0
                h, w = normalized.shape

            default_x_range = 20.0
            default_y_range = default_x_range / w * h
            default_z_range = 1.0

            if prompt_user:
                x_range, _ = QtWidgets.QInputDialog.getDouble(
                    self,
                    "X range",
                    "please input width of heightmap image (in meters)",
                    default_x_range, 0, 10000
                )
                y_range, _ = QtWidgets.QInputDialog.getDouble(
                    self,
                    "Y range",
                    "please input height of heightmap image (in meters)",
                    default_y_range, 0, 10000
                )
                z_range, _ = QtWidgets.QInputDialog.getDouble(
                    self,
                    "Z range",
                    "please input depth of heightmap image (in meters)",
                    default_z_range, 0, 10000
                )
            else:
                x_range = default_x_range
                y_range = default_y_range
                z_range = default_z_range

            x = np.linspace(-x_range/2, x_range/2, w)
            y = np.linspace(-y_range/2, y_range/2, h)

            z = np.flipud(normalized.T).astype(float) * z_range

            self._surfaceData = x, y, z
            x, y, z = self._surfaceData
            self.surf.setData(x=x, y=y, z=z)
            self._module = None

            self.setToolTip(os.path.basename(filename))

        except Exception as e:
            ermsg = f"failed to load heightmap image ({e})"
            self._err_message.showMessage(ermsg, 'error')
            self._logger.error(ermsg)
            self._logger.exception(e)

    def reloadSurface(self):
        if self._module is not None:
            self.reloadSurfaceModule()
        elif self._surfaceData is not None:
            self.reloadSurfaceImage()
        else:
            ermsg = "No surface to reload"
            self._err_message.showMessage(ermsg, 'warning')
            self._logger.warn(ermsg)

    def reloadSurfaceModule(self):
        '''reload the surface defined in the current python file'''
        if self._module is None:
            return

        def recursive_reload(module: ModuleType):
            # try to reload the module if it exists
            module = importlib.reload(module)

            # check & reload dependencies
            if hasattr(module, '__depends__'):
                for submodule_name in module.__depends__:
                    submodule = importlib.import_module(submodule_name)
                    recursive_reload(submodule)

                # make sure reloaded dependencies take effect
                module = importlib.reload(module)

            # return the module
            return module

        # try to reload the module if it exists
        self._module = recursive_reload(self._module)

        # try to retrieve a surface from the module
        self._surfaceData = self._module.surface()
        x, y, z = self._surfaceData
        self.surf.setData(x=x, y=y, z=z)

    def reloadSurfaceImage(self):
        x, y, z, *c = self._surfaceData

        x_range, _ = QtWidgets.QInputDialog.getDouble(
            self,
            "X range",
            "please input width of heightmap image (in meters)",
            np.ptp(x), 0, 10000
        )
        y_range, _ = QtWidgets.QInputDialog.getDouble(
            self,
            "Y range",
            "please input height of heightmap image (in meters)",
            np.ptp(y), 0, 10000
        )
        z_range, _ = QtWidgets.QInputDialog.getDouble(
            self,
            "Z range",
            "please input depth of heightmap image (in meters)",
            np.ptp(z), 0, 10000
        )

        x = np.linspace(-x_range/2, x_range/2, len(x))
        y = np.linspace(-y_range/2, y_range/2, len(y))
        z *= (z_range/np.ptp(z))

        self._surfaceData = x, y, z
        x, y, z = self._surfaceData
        self.surf.setData(x=x, y=y, z=z)

    def exportSurface(self, *, filename: str | None = None):
        '''export a surface to a PNG or TIFF heightmap file'''
        x, y, z, *c = self._surfaceData

        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                'save heightmap',
                f'./heightmap_{int(np.ptp(x))}'
                f'_{int(np.ptp(y))}_{np.ptp(z):.1f}.tif',
                'TIFF (*.tif *.tiff);;PNG (*.png)'
            )

        if filename in ('', None):
            return

        file_suffix = os.path.splitext(filename)[1].casefold()
        if file_suffix not in ('.png', '.tif', '.tiff'):
            filename += '.tif'
            file_suffix = '.tif'

        z_span = np.ptp(z)
        if np.isclose(z_span, 0.0):
            normalized = np.zeros_like(z, dtype=np.float32)
        else:
            normalized = ((z - z.min()) / z_span).astype(np.float32)

        if file_suffix == '.png':
            zz: np.ndarray = (normalized * 255).astype(np.uint8)
            zz = np.flipud(zz).transpose()

            img = QtGui.QImage(
                zz.data.tobytes(),
                z.shape[0],
                z.shape[1],
                z.shape[0],
                QtGui.QImage.Format.Format_Grayscale8
            )
            img.save(filename)
            return

        import tifffile

        zz16 = (normalized * 65535).astype(np.uint16)
        zz16 = np.flipud(zz16).transpose()
        tifffile.imwrite(filename, zz16, photometric='minisblack')
