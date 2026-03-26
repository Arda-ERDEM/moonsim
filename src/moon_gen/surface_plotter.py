#!/usr/bin/env python
'''
A widget for interactively plotting surfaces.
'''

import os
import sys
import logging
import importlib
import heapq
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter
import pyqtgraph as pg

import pyqtgraph.opengl as gl
if TYPE_CHECKING:
    from PyQt6 import QtCore, QtGui, QtWidgets
else:
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from moon_gen.lib.utils import SurfaceFunctionType, SurfaceType
from moon_gen.planning.main import plan_mission, generate_all_candidates
from moon_gen.planning import config as planning_config


class _StdoutBridge(QtCore.QObject):
    text_emitted = QtCore.pyqtSignal(str)


class _StdoutTee:
    def __init__(self, original_stream, bridge: _StdoutBridge):
        self._original_stream = original_stream
        self._bridge = bridge

    def write(self, text: str):
        if self._original_stream is not None:
            self._original_stream.write(text)
        if text:
            self._bridge.text_emitted.emit(text)

    def flush(self):
        if self._original_stream is not None:
            self._original_stream.flush()


class SurfacePlotter(QtWidgets.QFrame):

    _MAX_HEIGHTMAP_DIMENSION = max(
        64,
        int(os.environ.get('MOON_GEN_MAX_HEIGHTMAP_DIMENSION', '768'))
    )
    _HEIGHTMAP_CLIP_LOW_PERCENTILE = 1.0
    _HEIGHTMAP_CLIP_HIGH_PERCENTILE = 99.0
    _HEIGHTMAP_GAMMA = 0.9
    _PATHFIND_MAX_GRID_DIMENSION = max(
        64,
        int(os.environ.get('MOON_GEN_PATHFIND_MAX_GRID_DIMENSION', '180'))
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._module: ModuleType | None = None

        self._missionPathItem: gl.GLLinePlotItem | None = None
        self._waypointItem: gl.GLScatterPlotItem | None = None
        self._roverItem: gl.GLMeshItem | None = None
        self._plannedPath: np.ndarray | None = None
        self._roverPathCursor = 0
        self._missionTimer = QtCore.QTimer(self)
        self._missionTimer.timeout.connect(self._advanceRover)
        self._missionHazardMap: np.ndarray | None = None
        self._waypointsInitialized = False
        self._startWaypoint: tuple[float, float] | None = None
        self._endWaypoint: tuple[float, float] | None = None
        self._nextPickTarget = 'start'
        self._mapImageItem: pg.ImageItem | None = None
        self._mapWaypointScatter: pg.ScatterPlotItem | None = None

        # NEW: Multi-route selection system
        self._candidate_routes: dict[str, dict] = {}  # {safe/eco/fast: {path, cost, etc}}
        self._candidate_path_items: dict[str, gl.GLLinePlotItem] = {}  # For visualization
        self._current_selected_mode: str | None = None  # Currently selected route
        self._mission_data: dict | None = None  # Full mission data from planner
        self._consoleWidget: QtWidgets.QPlainTextEdit | None = None
        self._stdoutBridge: _StdoutBridge | None = None
        self._stdoutOriginal = None
        self._stdoutTee: _StdoutTee | None = None

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

        self._tabs = QtWidgets.QTabWidget(self)
        self._tabs.setMinimumWidth(280)
        self._setupMissionPlannerTab()
        self._installMissionConsoleHook()

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

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(self.vw, 1)
        self.layout().addWidget(self._tabs)
        self.layout().setContentsMargins(*4*[0])

        # error message
        self._err_message = QtWidgets.QErrorMessage(self)
        self._onSurfaceChanged()

    def _setupMissionPlannerTab(self):
        planner_tab = QtWidgets.QWidget(self)
        planner_layout = QtWidgets.QVBoxLayout(planner_tab)

        self._mapPlot = pg.PlotWidget(planner_tab)
        self._mapPlot.setMinimumHeight(260)
        self._mapPlot.setAspectLocked(True)
        self._mapPlot.showGrid(x=True, y=True, alpha=0.2)
        self._mapPlot.setLabel('bottom', 'X', units='m')
        self._mapPlot.setLabel('left', 'Y', units='m')
        self._mapPlot.setMenuEnabled(False)
        self._mapPlot.scene().sigMouseClicked.connect(self._onPlannerMapClicked)
        planner_layout.addWidget(self._mapPlot)

        self._startCoordLabel = QtWidgets.QLabel('Start: not selected', planner_tab)
        self._endCoordLabel = QtWidgets.QLabel('End: not selected', planner_tab)
        planner_layout.addWidget(self._startCoordLabel)
        planner_layout.addWidget(self._endCoordLabel)

        pick_row = QtWidgets.QHBoxLayout()
        self._pickStartButton = QtWidgets.QPushButton('Pick Start', planner_tab)
        self._pickStartButton.clicked.connect(self._activateStartPick)
        self._pickEndButton = QtWidgets.QPushButton('Pick End', planner_tab)
        self._pickEndButton.clicked.connect(self._activateEndPick)
        pick_row.addWidget(self._pickStartButton)
        pick_row.addWidget(self._pickEndButton)
        planner_layout.addLayout(pick_row)

        self._avoidanceSpin = QtWidgets.QDoubleSpinBox(planner_tab)
        self._avoidanceSpin.setDecimals(2)
        self._avoidanceSpin.setRange(0.0, 5.0)
        self._avoidanceSpin.setSingleStep(0.1)
        self._avoidanceSpin.setValue(1.8)

        self._slopeSpin = QtWidgets.QDoubleSpinBox(planner_tab)
        self._slopeSpin.setDecimals(2)
        self._slopeSpin.setRange(0.0, 5.0)
        self._slopeSpin.setSingleStep(0.1)
        self._slopeSpin.setValue(1.6)

        self._roverSpeedSpin = QtWidgets.QSpinBox(planner_tab)
        self._roverSpeedSpin.setRange(1, 15)
        self._roverSpeedSpin.setValue(3)

        form = QtWidgets.QFormLayout()
        form.addRow('Crater Avoid', self._avoidanceSpin)
        form.addRow('Slope Avoid', self._slopeSpin)
        form.addRow('Rover Speed', self._roverSpeedSpin)
        planner_layout.addLayout(form)

        button_row = QtWidgets.QHBoxLayout()
        self._clearWaypointsButton = QtWidgets.QPushButton('Clear Points', planner_tab)
        self._clearWaypointsButton.clicked.connect(self._clearWaypoints)
        self._swapWaypointsButton = QtWidgets.QPushButton('Swap', planner_tab)
        self._swapWaypointsButton.clicked.connect(self._swapWaypoints)
        self._planMissionButton = QtWidgets.QPushButton('Plan Path', planner_tab)
        self._planMissionButton.clicked.connect(self.planMissionPath)
        button_row.addWidget(self._clearWaypointsButton)
        button_row.addWidget(self._swapWaypointsButton)
        button_row.addWidget(self._planMissionButton)
        planner_layout.addLayout(button_row)

        mission_row = QtWidgets.QHBoxLayout()
        self._startMissionButton = QtWidgets.QPushButton('Start Rover', planner_tab)
        self._startMissionButton.clicked.connect(self.startRoverMission)
        self._stopMissionButton = QtWidgets.QPushButton('Stop', planner_tab)
        self._stopMissionButton.clicked.connect(self.stopRoverMission)
        mission_row.addWidget(self._startMissionButton)
        mission_row.addWidget(self._stopMissionButton)
        planner_layout.addLayout(mission_row)

        # NEW: Route Selection Buttons
        route_selection_label = QtWidgets.QLabel('Select Route:', planner_tab)
        route_selection_label.setStyleSheet("font-weight: bold;")
        planner_layout.addWidget(route_selection_label)

        route_buttons_row = QtWidgets.QHBoxLayout()
        
        self._buttonSafe = QtWidgets.QPushButton('SAFE', planner_tab)
        self._buttonSafe.setStyleSheet("color: white; background-color: #2ecc71; font-weight: bold;")
        self._buttonSafe.clicked.connect(lambda: self._selectRoute('safe'))
        self._buttonSafe.setEnabled(False)
        
        self._buttonEco = QtWidgets.QPushButton('ECO', planner_tab)
        self._buttonEco.setStyleSheet("color: white; background-color: #f1c40f; font-weight: bold;")
        self._buttonEco.clicked.connect(lambda: self._selectRoute('eco'))
        self._buttonEco.setEnabled(False)
        
        self._buttonFast = QtWidgets.QPushButton('FAST', planner_tab)
        self._buttonFast.setStyleSheet("color: white; background-color: #e74c3c; font-weight: bold;")
        self._buttonFast.clicked.connect(lambda: self._selectRoute('fast'))
        self._buttonFast.setEnabled(False)
        
        self._buttonAuto = QtWidgets.QPushButton('AUTO', planner_tab)
        self._buttonAuto.setStyleSheet("color: white; background-color: #3498db; font-weight: bold;")
        self._buttonAuto.clicked.connect(self._selectRouteAuto)
        self._buttonAuto.setEnabled(False)
        
        route_buttons_row.addWidget(self._buttonSafe)
        route_buttons_row.addWidget(self._buttonEco)
        route_buttons_row.addWidget(self._buttonFast)
        route_buttons_row.addWidget(self._buttonAuto)
        planner_layout.addLayout(route_buttons_row)

        console_label = QtWidgets.QLabel('Route Logs:', planner_tab)
        console_label.setStyleSheet("font-weight: bold;")
        planner_layout.addWidget(console_label)

        self._consoleWidget = QtWidgets.QPlainTextEdit(planner_tab)
        self._consoleWidget.setReadOnly(True)
        self._consoleWidget.setMinimumHeight(140)
        self._consoleWidget.setPlaceholderText('Planner terminal output will appear here in real-time...')
        planner_layout.addWidget(self._consoleWidget)

        self._missionStatusLabel = QtWidgets.QLabel(
            'Pick Start and End directly on the map, then click Plan Path.',
            planner_tab
        )
        self._missionStatusLabel.setWordWrap(True)
        planner_layout.addWidget(self._missionStatusLabel)
        planner_layout.addStretch(1)

        self._tabs.addTab(planner_tab, 'Moon Route')

    def _setMissionStatus(self, text: str):
        self._missionStatusLabel.setText(text)

    def _installMissionConsoleHook(self):
        if self._consoleWidget is None or self._stdoutTee is not None:
            return

        self._stdoutBridge = _StdoutBridge(self)
        self._stdoutBridge.text_emitted.connect(self._appendMissionConsoleText)

        self._stdoutOriginal = sys.stdout
        self._stdoutTee = _StdoutTee(self._stdoutOriginal, self._stdoutBridge)
        sys.stdout = self._stdoutTee

    def _appendMissionConsoleText(self, text: str):
        if self._consoleWidget is None:
            return

        self._consoleWidget.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self._consoleWidget.insertPlainText(text)
        self._consoleWidget.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._stdoutTee is not None and self._stdoutOriginal is not None and sys.stdout is self._stdoutTee:
            sys.stdout = self._stdoutOriginal
        self._stdoutTee = None
        self._stdoutOriginal = None
        super().closeEvent(event)

    def _activateStartPick(self):
        self._nextPickTarget = 'start'
        self._setMissionStatus('Click on map to choose START waypoint.')

    def _activateEndPick(self):
        self._nextPickTarget = 'end'
        self._setMissionStatus('Click on map to choose END waypoint.')

    def _clearWaypoints(self):
        self.stopRoverMission()
        self._startWaypoint = None
        self._endWaypoint = None
        self._plannedPath = None
        self._roverPathCursor = 0
        self._nextPickTarget = 'start'
        self._refreshWaypointLabels()
        self._refreshMapWaypointOverlay()

        if self._missionPathItem is not None:
            self.vw.removeItem(self._missionPathItem)
            self._missionPathItem = None
        if self._waypointItem is not None:
            self.vw.removeItem(self._waypointItem)
            self._waypointItem = None
        if self._roverItem is not None:
            self.vw.removeItem(self._roverItem)
            self._roverItem = None

        # Clear candidate routes
        self._clearRouteVisualization()
        self._candidate_routes.clear()
        self._mission_data = None
        self._current_selected_mode = None

        # Disable route selection buttons
        self._buttonSafe.setEnabled(False)
        self._buttonEco.setEnabled(False)
        self._buttonFast.setEnabled(False)
        self._buttonAuto.setEnabled(False)

        self._setMissionStatus('Waypoints cleared. Pick START then END on map.')

    def _refreshWaypointLabels(self):
        if self._startWaypoint is None:
            self._startCoordLabel.setText('Start: not selected')
        else:
            self._startCoordLabel.setText(
                f'Start: x={self._startWaypoint[0]:.2f}, y={self._startWaypoint[1]:.2f}'
            )

        if self._endWaypoint is None:
            self._endCoordLabel.setText('End: not selected')
        else:
            self._endCoordLabel.setText(
                f'End: x={self._endWaypoint[0]:.2f}, y={self._endWaypoint[1]:.2f}'
            )

    def _refreshMapWaypointOverlay(self):
        if self._mapWaypointScatter is None:
            self._mapWaypointScatter = pg.ScatterPlotItem(size=12)
            self._mapPlot.addItem(self._mapWaypointScatter)

        points = []
        if self._startWaypoint is not None:
            points.append({
                'pos': self._startWaypoint,
                'brush': pg.mkBrush(40, 220, 60, 220),
                'pen': pg.mkPen(20, 80, 20, 255),
                'symbol': 'o',
                'size': 12
            })
        if self._endWaypoint is not None:
            points.append({
                'pos': self._endWaypoint,
                'brush': pg.mkBrush(235, 80, 70, 220),
                'pen': pg.mkPen(100, 20, 20, 255),
                'symbol': 't',
                'size': 12
            })
        self._mapWaypointScatter.setData(points)

    def _onPlannerMapClicked(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        view_box = self._mapPlot.getViewBox()
        if view_box is None:
            return

        point = view_box.mapSceneToView(ev.scenePos())
        x, y, z, *c = self._surfaceData
        x_val = float(np.clip(point.x(), x.min(), x.max()))
        y_val = float(np.clip(point.y(), y.min(), y.max()))

        if self._nextPickTarget == 'start':
            self._startWaypoint = (x_val, y_val)
            self._nextPickTarget = 'end'
            self._setMissionStatus('Start selected. Click to choose END waypoint.')
        else:
            self._endWaypoint = (x_val, y_val)
            self._nextPickTarget = 'start'
            self._setMissionStatus('End selected. Click Plan Path to generate route.')

        self._refreshWaypointLabels()
        self._refreshMapWaypointOverlay()

    def _updateWaypointRanges(self):
        x, y, z, *c = self._surfaceData
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())

        if self._mapImageItem is None:
            self._mapImageItem = pg.ImageItem(axisOrder='row-major')
            self._mapPlot.addItem(self._mapImageItem)

        z_view = np.asarray(z.T, dtype=float)
        self._mapImageItem.setImage(z_view, autoLevels=True)
        self._mapImageItem.setRect(QtCore.QRectF(x_min, y_min, np.ptp(x), np.ptp(y)))

        self._mapPlot.setXRange(x_min, x_max, padding=0.02)
        self._mapPlot.setYRange(y_min, y_max, padding=0.02)

        if self._startWaypoint is not None:
            self._startWaypoint = (
                float(np.clip(self._startWaypoint[0], x_min, x_max)),
                float(np.clip(self._startWaypoint[1], y_min, y_max)),
            )
        if self._endWaypoint is not None:
            self._endWaypoint = (
                float(np.clip(self._endWaypoint[0], x_min, x_max)),
                float(np.clip(self._endWaypoint[1], y_min, y_max)),
            )

        self._refreshWaypointLabels()
        self._refreshMapWaypointOverlay()

        self._missionHazardMap = None

    def _clearMissionGraphics(self):
        self.stopRoverMission()

        if self._missionPathItem is not None:
            self.vw.removeItem(self._missionPathItem)
            self._missionPathItem = None

        if self._waypointItem is not None:
            self.vw.removeItem(self._waypointItem)
            self._waypointItem = None

        if self._roverItem is not None:
            self.vw.removeItem(self._roverItem)
            self._roverItem = None

        self._plannedPath = None
        self._roverPathCursor = 0

    def _onSurfaceChanged(self):
        self._clearMissionGraphics()
        self._updateWaypointRanges()
        self._setMissionStatus('Surface ready. Define two waypoints and plan mission.')

    def _swapWaypoints(self):
        self._startWaypoint, self._endWaypoint = self._endWaypoint, self._startWaypoint
        self._refreshWaypointLabels()
        self._refreshMapWaypointOverlay()

    def _xyToGridIndex(self, x_value: float, y_value: float) -> tuple[int, int]:
        x, y, z, *c = self._surfaceData
        i = int(np.abs(x - x_value).argmin())
        j = int(np.abs(y - y_value).argmin())
        return i, j

    def _gridIndexToPoint(self, i: int, j: int) -> np.ndarray:
        x, y, z, *c = self._surfaceData
        z_offset = 0.02 * max(np.ptp(z), 1e-3)
        return np.array([x[i], y[j], z[i, j] + z_offset], dtype=float)

    def _buildHazardMap(self, z: np.ndarray) -> np.ndarray:
        dzdx, dzdy = np.gradient(z)
        slope = np.hypot(dzdx, dzdy)
        slope_scale = np.percentile(slope, 95) + 1e-9
        slope_norm = np.clip(slope / slope_scale, 0.0, 4.0)

        local_mean = gaussian_filter(z, sigma=2.2)
        pit_depth = np.clip(local_mean - z, 0.0, None)
        pit_scale = np.percentile(pit_depth, 95) + 1e-9
        pit_norm = np.clip(pit_depth / pit_scale, 0.0, 4.0)

        slope_weight = self._slopeSpin.value()
        crater_weight = self._avoidanceSpin.value()

        return 1.0 + slope_weight * slope_norm + crater_weight * pit_norm

    def _aStarPath(
        self,
        start_idx: tuple[int, int],
        end_idx: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        x, y, z, *c = self._surfaceData
        nx, ny = z.shape

        hazard = self._missionHazardMap
        if hazard is None or hazard.shape != z.shape:
            hazard = self._buildHazardMap(z)
            self._missionHazardMap = hazard

        stride = max(
            1,
            int(np.ceil(max(nx, ny) / self._PATHFIND_MAX_GRID_DIMENSION))
        )

        if stride > 1:
            row_idx = np.arange(0, nx, stride, dtype=np.int64)
            col_idx = np.arange(0, ny, stride, dtype=np.int64)
            if row_idx[-1] != nx - 1:
                row_idx = np.append(row_idx, nx - 1)
            if col_idx[-1] != ny - 1:
                col_idx = np.append(col_idx, ny - 1)

            hazard_grid = hazard[np.ix_(row_idx, col_idx)]
            start_idx = (
                int(np.abs(row_idx - start_idx[0]).argmin()),
                int(np.abs(col_idx - start_idx[1]).argmin()),
            )
            end_idx = (
                int(np.abs(row_idx - end_idx[0]).argmin()),
                int(np.abs(col_idx - end_idx[1]).argmin()),
            )
            nx, ny = hazard_grid.shape
        else:
            hazard_grid = hazard
            row_idx = np.arange(nx, dtype=np.int64)
            col_idx = np.arange(ny, dtype=np.int64)

        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
            di = a[0] - b[0]
            dj = a[1] - b[1]
            return float(np.hypot(di, dj))

        frontier: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(frontier, (0.0, start_idx))

        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_idx: None}
        cost_so_far: dict[tuple[int, int], float] = {start_idx: 0.0}

        max_expansions = min(nx * ny, 120_000)
        expansions = 0

        while frontier and expansions < max_expansions:
            _, current = heapq.heappop(frontier)
            expansions += 1

            if current == end_idx:
                break

            ci, cj = current
            for di, dj in neighbors:
                ni, nj = ci + di, cj + dj
                if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
                    continue

                step_len = float(np.hypot(di, dj))
                avg_hazard = 0.5 * (hazard_grid[ci, cj] + hazard_grid[ni, nj])
                new_cost = cost_so_far[current] + step_len * avg_hazard
                nxt = (ni, nj)

                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + heuristic(nxt, end_idx)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if end_idx not in came_from:
            return None

        path: list[tuple[int, int]] = []
        node: tuple[int, int] | None = end_idx
        while node is not None:
            path.append((int(row_idx[node[0]]), int(col_idx[node[1]])))
            node = came_from[node]

        path.reverse()
        return self._densifyIndexPath(path)

    def _densifyIndexPath(
        self,
        path: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if len(path) <= 1:
            return path

        dense_path: list[tuple[int, int]] = [path[0]]
        for (ai, aj), (bi, bj) in zip(path[:-1], path[1:]):
            di = bi - ai
            dj = bj - aj
            steps = max(abs(di), abs(dj))
            if steps <= 1:
                if dense_path[-1] != (bi, bj):
                    dense_path.append((bi, bj))
                continue

            for step in range(1, steps + 1):
                ti = ai + int(round(di * step / steps))
                tj = aj + int(round(dj * step / steps))
                if dense_path[-1] != (ti, tj):
                    dense_path.append((ti, tj))

        return dense_path

    def _setRoverPosition(self, point: np.ndarray):
        if self._roverItem is None:
            x, y, z, *c = self._surfaceData
            radius = 0.01 * max(np.ptp(x), np.ptp(y))
            mesh = gl.MeshData.sphere(rows=10, cols=16, radius=radius)
            self._roverItem = gl.GLMeshItem(
                meshdata=mesh,
                smooth=True,
                color=(0.05, 0.85, 1.0, 0.95),
                shader='shaded',
                glOptions='opaque'
            )
            self.vw.addItem(self._roverItem)

        self._roverItem.resetTransform()
        self._roverItem.translate(float(point[0]), float(point[1]), float(point[2]))

    def planMissionPath(self):
        if self._startWaypoint is None or self._endWaypoint is None:
            self._setMissionStatus('Pick both START and END waypoint on map first.')
            return

        x, y, z, *c = self._surfaceData
        start_idx = self._xyToGridIndex(self._startWaypoint[0], self._startWaypoint[1])
        end_idx = self._xyToGridIndex(self._endWaypoint[0], self._endWaypoint[1])

        if start_idx == end_idx:
            self._setMissionStatus('Start and end waypoint are the same.')
            return

        # Override planning config with UI values
        planning_config.SAFE_WEIGHTS['slope'] = self._slopeSpin.value() * 2.0
        planning_config.SAFE_WEIGHTS['obstacle'] = self._avoidanceSpin.value() * 2.0
        
        self._setMissionStatus('Generating all three candidate routes (Safe, Eco, Fast)...')
        QtWidgets.QApplication.processEvents()

        try:
            # IMPORT the new function from planning.main
            from moon_gen.planning.main import generate_all_candidates
            
            # Generate ALL three routes WITHOUT selecting
            result = generate_all_candidates(
                image_input=z,
                start=start_idx,
                goal=end_idx
            )
        except Exception as e:
            self._logger.error(f"Planning failed: {e}", exc_info=True)
            self._setMissionStatus(f'Planning Error: {e}')
            return

        # Store the mission data
        self._mission_data = result
        plans = result.get('plans', {})
        
        # Ensure all three routes exist
        valid_modes = [m for m in ('safe', 'eco', 'fast') if m in plans and plans[m]['summary']['exists']]
        if not valid_modes:
            self._setMissionStatus('No valid routes found. Adjust terrain avoidance weights.')
            return

        # Clear previous route visualization
        self._clearRouteVisualization()
        
        # Generate and visualize all three routes
        self._candidate_routes = {}
        colors = {'safe': (0.2, 0.9, 0.4, 1.0), 'eco': (0.2, 0.5, 0.9, 1.0), 'fast': (0.9, 0.3, 0.2, 1.0)}
        
        for mode in ('safe', 'eco', 'fast'):
            if mode not in plans:
                continue
                
            plan = plans[mode]
            summary = plan['summary']
            
            if not summary['exists']:
                continue
            
            # Get the path
            path_result = plan['result']
            path_indices = path_result.smoothed_path if path_result.smoothed_path else path_result.path
            
            # Convert to 3D points
            points = []
            for r, c in path_indices:
                r = int(np.clip(r, 0, z.shape[0]-1))
                c = int(np.clip(c, 0, z.shape[1]-1))
                points.append(self._gridIndexToPoint(r, c))
            
            if not points:
                continue
                
            path_points = np.vstack(points)
            
            # Store route data
            self._candidate_routes[mode] = {
                'path_points': path_points,
                'path_indices': path_indices,
                'summary': summary,
                'line_item': None,
            }
            
            # Draw the route (not highlighted yet)
            color = colors.get(mode, (1, 1, 1, 1))
            line_item = gl.GLLinePlotItem(
                pos=path_points,
                color=color,
                width=2.0,
                antialias=True,
                mode='line_strip'
            )
            self.vw.addItem(line_item)
            self._candidate_routes[mode]['line_item'] = line_item
            self._candidate_path_items[mode] = line_item

        # Print to console
        print("=" * 70)
        print("MULTI-ROUTE PATH PLANNING COMPLETE")
        print("=" * 70)
        for mode in ('safe', 'eco', 'fast'):
            if mode in self._candidate_routes:
                summary = self._candidate_routes[mode]['summary']
                print(f"{mode.upper():5} | Length: {summary['path_length']:6.1f} | Risk: {summary['mean_risk']:.3f} | Cost: {summary['path_cost']:8.2f}")

        # Enable route selection buttons
        for btn_name in ('safe', 'eco', 'fast'):
            if btn_name in self._candidate_routes:
                getattr(self, f'_button{btn_name.capitalize()}').setEnabled(True)
        self._buttonAuto.setEnabled(True)

        # Update status
        self._setMissionStatus('All routes GENERATED. Click SAFE, ECO, FAST, or AUTO to select.')
        self._current_selected_mode = None

    def _clearRouteVisualization(self):
        """Remove all route line items from the 3D view."""
        for mode in self._candidate_path_items.values():
            if mode is not None:
                try:
                    self.vw.removeItem(mode)
                except:
                    pass
        self._candidate_path_items.clear()

    def _selectRoute(self, mode: str):
        """Manually select a route (SAFE, ECO, or FAST)."""
        if mode not in self._candidate_routes:
            self._setMissionStatus(f'{mode.upper()} route is not available.')
            return

        self._current_selected_mode = mode
        route_data = self._candidate_routes[mode]
        path_points = route_data['path_points']
        summary = route_data['summary']

        # Update the active path (for rover movement)
        self._plannedPath = path_points
        self._roverPathCursor = 0

        # Set rover to start
        if len(path_points) > 0:
            self._setRoverPosition(path_points[0])

        # Highlight the selected route (make it thicker/brighter)
        self._updateRouteHighlight(mode)

        # Update status
        status_msg = (
            f"SELECTED: {mode.upper()} | "
            f"Length: {summary['path_length']:.1f} | "
            f"Risk: {summary['mean_risk']:.3f} | "
            f"Click 'Start Rover' to execute."
        )
        self._setMissionStatus(status_msg)
        print(f"[SELECTION] User selected {mode.upper()} route")

    def _selectRouteAuto(self):
        """Automatically select the best route using the decision logic."""
        if not self._mission_data:
            self._setMissionStatus('No mission data. Plan path first.')
            return

        plans = self._mission_data.get('plans', {})
        mission = self._mission_data.get('mission')
        
        # Get candidate summaries
        candidate_summaries = {mode: plans[mode]['summary'] for mode in plans if plans[mode]['summary']['exists']}
        
        if not candidate_summaries:
            self._setMissionStatus('No valid routes for AUTO selection.')
            return

        # Use the decision logic
        from moon_gen.planning.decision import select_autonomous_mode
        
        selected_mode, explanation = select_autonomous_mode(candidate_summaries, mission)

        if not selected_mode or selected_mode not in self._candidate_routes:
            self._setMissionStatus('AUTO failed to select a route.')
            return

        # Highlight explanation
        factors = explanation.get('main_decision_factors', [])
        reason = factors[0] if factors else "Optimal selection."

        self._current_selected_mode = selected_mode
        route_data = self._candidate_routes[selected_mode]
        path_points = route_data['path_points']

        # Update active path
        self._plannedPath = path_points
        self._roverPathCursor = 0

        # Set rover to start
        if len(path_points) > 0:
            self._setRoverPosition(path_points[0])

        # Highlight selected route
        self._updateRouteHighlight(selected_mode)

        # Status with explanation
        status_msg = (
            f"AUTO SELECTED: {selected_mode.upper()}\n"
            f"Reason: {reason}\n"
            f"Ready for execution."
        )
        self._setMissionStatus(status_msg)
        print(f"[AUTO DECISION] Selected {selected_mode.upper()}")
        print(f"  Reason: {reason}")

    def _updateRouteHighlight(self, selected_mode: str):
        """Update line widths/colors to highlight the selected route."""
        colors = {'safe': (0.2, 0.9, 0.4, 1.0), 'eco': (0.2, 0.5, 0.9, 1.0), 'fast': (0.9, 0.3, 0.2, 1.0)}
        
        for mode in ('safe', 'eco', 'fast'):
            if mode not in self._candidate_routes:
                continue
                
            line_item = self._candidate_routes[mode]['line_item']
            if line_item is None:
                continue
            
            # Selected route: thicker, full opacity
            # Other routes: thinner, slightly dimmed
            if mode == selected_mode:
                line_item.setData(width=4.0, color=colors[mode])
            else:
                # Dim non-selected routes
                dim_color = tuple(c * 0.5 + 0.2 for c in colors[mode][:3]) + (0.6,)
                line_item.setData(width=1.5, color=dim_color)

    def startRoverMission(self):
        if self._plannedPath is None or len(self._plannedPath) < 2:
            if not self._current_selected_mode:
                self._setMissionStatus('No route selected. Generate paths and select SAFE, ECO, FAST, or AUTO.')
                return

        if self._plannedPath is None or len(self._plannedPath) < 2:
            return

        self._missionTimer.start(35)
        self._setMissionStatus('Rover moving autonomously between waypoints.')

    def stopRoverMission(self):
        if self._missionTimer.isActive():
            self._missionTimer.stop()

    def _advanceRover(self):
        if self._plannedPath is None or len(self._plannedPath) == 0:
            self.stopRoverMission()
            return

        speed = max(1, int(self._roverSpeedSpin.value()))
        self._roverPathCursor = min(
            self._roverPathCursor + speed,
            len(self._plannedPath) - 1
        )

        self._setRoverPosition(self._plannedPath[self._roverPathCursor])

        if self._roverPathCursor >= len(self._plannedPath) - 1:
            self.stopRoverMission()
            self._setMissionStatus('Mission complete: rover reached end waypoint.')

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

        scale_factor = int(np.ceil(largest_dim / max_size))
        target_height = height // scale_factor
        target_width = width // scale_factor

        if target_height < 1 or target_width < 1:
            return np.asarray(values, dtype=np.float32)

        trimmed = values[:target_height * scale_factor, :target_width * scale_factor]
        reshaped = trimmed.reshape(target_height, scale_factor, target_width, scale_factor)
        downsampled = reshaped.mean(axis=(1, 3))

        return np.asarray(downsampled, dtype=np.float32)

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
            self._onSurfaceChanged()
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
            self._onSurfaceChanged()

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
        self._onSurfaceChanged()

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
        self._onSurfaceChanged()

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
