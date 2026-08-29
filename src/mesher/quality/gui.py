"""Native Qt desktop UI for exploring Tri3 and Quad4 element quality."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .checker import ElementType
from .gui_model import (
    QualityThresholds,
    classify_quality,
    evaluate_element,
    format_metric,
    preset_nodes,
)


FloatArray = NDArray[np.float64]

_BACKGROUND = "#0f172a"
_PANEL = "#172033"
_PANEL_ALT = "#1e293b"
_TEXT = "#e5edf8"
_MUTED = "#94a3b8"
_GRID = "#2b3950"
_ACCENT = "#38bdf8"
_PASS = "#22c55e"
_WARNING = "#f59e0b"
_INVALID = "#ef4444"


class ElementCanvas(QWidget):
    """Widget that renders and edits one element in normalized coordinates."""

    nodes_changed = Signal(object)

    _WORLD_LIMIT = 1.0
    _PADDING = 56.0
    _NODE_RADIUS = 10.0

    def __init__(self, nodes: FloatArray, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._nodes = np.asarray(nodes, dtype=np.float64).copy()
        self._active_node: int | None = None
        self._status_key = "pass"
        self.setMinimumSize(480, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_nodes(self, nodes: FloatArray) -> None:
        self._nodes = np.asarray(nodes, dtype=np.float64).copy()
        self._active_node = None
        self.update()

    def set_status(self, status_key: str) -> None:
        self._status_key = status_key
        self.update()

    def _drawing_scale(self) -> float:
        return max(80.0, min(self.width(), self.height()) - 2 * self._PADDING)

    def _world_to_canvas(self, point: Sequence[float]) -> QPointF:
        scale = self._drawing_scale() / (2 * self._WORLD_LIMIT)
        return QPointF(
            self.width() / 2 + float(point[0]) * scale,
            self.height() / 2 - float(point[1]) * scale,
        )

    def _canvas_to_world(self, point: QPointF) -> FloatArray:
        scale = self._drawing_scale() / (2 * self._WORLD_LIMIT)
        world = np.array(
            [
                (point.x() - self.width() / 2) / scale,
                (self.height() / 2 - point.y()) / scale,
            ],
            dtype=np.float64,
        )
        return np.clip(world, -self._WORLD_LIMIT, self._WORLD_LIMIT)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        positions = np.array(
            [[point.x(), point.y()] for point in map(self._world_to_canvas, self._nodes)]
        )
        cursor = np.array([event.position().x(), event.position().y()])
        distances = np.linalg.norm(positions - cursor, axis=1)
        closest = int(np.argmin(distances))
        self._active_node = (
            closest if distances[closest] <= 2.0 * self._NODE_RADIUS else None
        )
        self.setFocus()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._active_node is None or not event.buttons() & Qt.MouseButton.LeftButton:
            return
        self._nodes[self._active_node] = self._canvas_to_world(event.position())
        self.nodes_changed.emit(self._nodes.copy())
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        deltas = {
            Qt.Key.Key_Left: (-1.0, 0.0),
            Qt.Key.Key_Right: (1.0, 0.0),
            Qt.Key.Key_Up: (0.0, 1.0),
            Qt.Key.Key_Down: (0.0, -1.0),
        }
        if self._active_node is None or event.key() not in deltas:
            super().keyPressEvent(event)
            return
        step = 0.05 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.01
        self._nodes[self._active_node] = np.clip(
            self._nodes[self._active_node] + np.asarray(deltas[event.key()]) * step,
            -self._WORLD_LIMIT,
            self._WORLD_LIMIT,
        )
        self.nodes_changed.emit(self._nodes.copy())
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(_PANEL))

        painter.setPen(QPen(QColor(_GRID), 1))
        for coordinate in np.linspace(-1.0, 1.0, 9):
            painter.drawLine(
                self._world_to_canvas((coordinate, -1.0)),
                self._world_to_canvas((coordinate, 1.0)),
            )
            painter.drawLine(
                self._world_to_canvas((-1.0, coordinate)),
                self._world_to_canvas((1.0, coordinate)),
            )

        painter.setPen(QPen(QColor("#46566e"), 1))
        painter.drawLine(
            self._world_to_canvas((-1.0, 0.0)),
            self._world_to_canvas((1.0, 0.0)),
        )
        painter.drawLine(
            self._world_to_canvas((0.0, -1.0)),
            self._world_to_canvas((0.0, 1.0)),
        )

        positions = [self._world_to_canvas(point) for point in self._nodes]
        status_color = {
            "pass": _PASS,
            "warning": _WARNING,
            "invalid": _INVALID,
        }.get(self._status_key, _ACCENT)
        fill = QColor(status_color)
        fill.setAlpha(52)
        painter.setBrush(fill)
        painter.setPen(QPen(QColor(status_color), 3))
        painter.drawPolygon(QPolygonF(positions))

        for index, point in enumerate(positions):
            selected = index == self._active_node
            radius = self._NODE_RADIUS + (2 if selected else 0)
            painter.setBrush(QColor(_ACCENT if selected else _TEXT))
            painter.setPen(QPen(QColor(_BACKGROUND), 3))
            painter.drawEllipse(point, radius, radius)
            painter.setPen(QColor(_TEXT))
            font = painter.font()
            font.setBold(True)
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(point + QPointF(17, -16), f"P{index + 1}")

        painter.setPen(QColor(_MUTED))
        font = painter.font()
        font.setBold(False)
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(
            QPointF(22, self.height() - 20),
            "Drag a point · Arrow keys nudge · Shift + arrow moves faster",
        )


class ElementQualityWindow(QMainWindow):
    """Standalone native window for interactively checking one element."""

    def __init__(self, initial_type: ElementType = ElementType.TRI3) -> None:
        super().__init__()
        self._element_type = initial_type
        self._nodes = preset_nodes(initial_type)
        self._thresholds = QualityThresholds()
        self._metric_labels: dict[str, QLabel] = {}
        self._coordinate_labels: list[QLabel] = []

        self.setWindowTitle("Mesher · Element Quality Explorer")
        self.resize(1120, 720)
        self.setMinimumSize(900, 620)
        self._build_layout()
        self._set_element_type(initial_type)

    def _build_layout(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(14)

        title = QLabel("Element Quality Explorer")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel(
            "Move the nodes and inspect Mesher's quality metrics in real time."
        )
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self._tri_button = QPushButton("Triangle · 3 nodes")
        self._tri_button.clicked.connect(
            lambda: self._set_element_type(ElementType.TRI3)
        )
        toolbar.addWidget(self._tri_button)
        self._quad_button = QPushButton("Quadrilateral · 4 nodes")
        self._quad_button.clicked.connect(
            lambda: self._set_element_type(ElementType.QUAD4)
        )
        toolbar.addWidget(self._quad_button)
        toolbar.addStretch()
        reset_button = QPushButton("Reset element")
        reset_button.clicked.connect(self._reset_element)
        toolbar.addWidget(reset_button)
        layout.addLayout(toolbar)

        content = QHBoxLayout()
        content.setSpacing(14)
        self._canvas = ElementCanvas(self._nodes)
        self._canvas.nodes_changed.connect(self._nodes_changed)
        content.addWidget(self._canvas, stretch=1)
        content.addWidget(self._build_sidebar())
        layout.addLayout(content, stretch=1)

        self.setStyleSheet(
            f"""
            QWidget#root {{ background: {_BACKGROUND}; color: {_TEXT}; }}
            QLabel#title {{ color: {_TEXT}; font-size: 25px; font-weight: 700; }}
            QLabel#subtitle {{ color: {_MUTED}; font-size: 12px; margin-bottom: 5px; }}
            QPushButton {{
                background: {_PANEL_ALT}; color: {_TEXT}; border: 0;
                border-radius: 7px; padding: 9px 14px; font-size: 12px;
            }}
            QPushButton:hover {{ background: #334155; }}
            QPushButton[selected="true"] {{
                background: {_ACCENT}; color: {_BACKGROUND}; font-weight: 700;
            }}
            QFrame#sidebar {{ background: {_PANEL}; border-radius: 10px; }}
            QFrame#status {{ background: {_PANEL_ALT}; border-radius: 7px; }}
            QLabel {{ color: {_TEXT}; }}
            QLabel[role="section"] {{ font-size: 14px; font-weight: 700; }}
            QLabel[role="muted"] {{ color: {_MUTED}; font-size: 11px; }}
            QLabel[role="metric"] {{ font-family: monospace; font-size: 14px; font-weight: 700; }}
            QLabel[role="coordinate"] {{ font-family: monospace; font-size: 11px; }}
            QLineEdit {{
                background: {_BACKGROUND}; color: {_TEXT}; border: 1px solid #475569;
                border-radius: 5px; padding: 5px 7px; selection-background-color: {_ACCENT};
            }}
            """
        )

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(330)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 18, 18, 18)
        sidebar_layout.setSpacing(8)

        self._element_name_label = QLabel()
        self._element_name_label.setProperty("role", "section")
        sidebar_layout.addWidget(self._element_name_label)

        status = QFrame()
        status.setObjectName("status")
        status_layout = QVBoxLayout(status)
        status_layout.setContentsMargins(14, 12, 14, 12)
        self._status_label = QLabel()
        self._status_label.setStyleSheet("font-size: 17px; font-weight: 700;")
        status_layout.addWidget(self._status_label)
        self._status_detail_label = QLabel()
        self._status_detail_label.setProperty("role", "muted")
        self._status_detail_label.setWordWrap(True)
        status_layout.addWidget(self._status_detail_label)
        sidebar_layout.addWidget(status)

        metrics = (
            ("Minimum Jacobian", "jacobian_minimum"),
            ("Maximum Jacobian", "jacobian_maximum"),
            ("Minimum scaled Jacobian", "scaled_jacobian_minimum"),
            ("Maximum scaled Jacobian", "scaled_jacobian_maximum"),
            ("Aspect ratio", "aspect_ratio"),
        )
        metric_grid = QGridLayout()
        metric_grid.setVerticalSpacing(8)
        for row, (name, key) in enumerate(metrics):
            name_label = QLabel(name)
            name_label.setProperty("role", "muted")
            metric_grid.addWidget(name_label, row, 0)
            value_label = QLabel("—")
            value_label.setProperty("role", "metric")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._metric_labels[key] = value_label
            metric_grid.addWidget(value_label, row, 1)
        sidebar_layout.addLayout(metric_grid)

        sidebar_layout.addSpacing(7)
        threshold_title = QLabel("Quality thresholds")
        threshold_title.setProperty("role", "section")
        sidebar_layout.addWidget(threshold_title)
        threshold_grid = QGridLayout()
        minimum_label = QLabel("Min. scaled Jacobian")
        minimum_label.setProperty("role", "muted")
        threshold_grid.addWidget(minimum_label, 0, 0)
        self._minimum_scaled_edit = QLineEdit("0.30")
        self._minimum_scaled_edit.setFixedWidth(74)
        self._minimum_scaled_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        threshold_grid.addWidget(self._minimum_scaled_edit, 0, 1)
        maximum_label = QLabel("Max. aspect ratio")
        maximum_label.setProperty("role", "muted")
        threshold_grid.addWidget(maximum_label, 1, 0)
        self._maximum_aspect_edit = QLineEdit("4.00")
        self._maximum_aspect_edit.setFixedWidth(74)
        self._maximum_aspect_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        threshold_grid.addWidget(self._maximum_aspect_edit, 1, 1)
        self._minimum_scaled_edit.editingFinished.connect(self._thresholds_changed)
        self._maximum_aspect_edit.editingFinished.connect(self._thresholds_changed)
        sidebar_layout.addLayout(threshold_grid)

        sidebar_layout.addSpacing(7)
        coordinate_title = QLabel("Node coordinates")
        coordinate_title.setProperty("role", "section")
        sidebar_layout.addWidget(coordinate_title)
        self._coordinates_layout = QVBoxLayout()
        self._coordinates_layout.setSpacing(3)
        sidebar_layout.addLayout(self._coordinates_layout)
        sidebar_layout.addStretch()
        return sidebar

    def _set_element_type(self, element_type: ElementType) -> None:
        self._element_type = element_type
        self._nodes = preset_nodes(element_type)
        self._canvas.set_nodes(self._nodes)
        self._element_name_label.setText(
            "TRI3 element" if element_type is ElementType.TRI3 else "QUAD4 element"
        )
        self._set_selected(self._tri_button, element_type is ElementType.TRI3)
        self._set_selected(self._quad_button, element_type is ElementType.QUAD4)
        self._rebuild_coordinate_labels()
        self._refresh_quality()

    @staticmethod
    def _set_selected(button: QPushButton, selected: bool) -> None:
        button.setProperty("selected", selected)
        button.style().unpolish(button)
        button.style().polish(button)

    def _reset_element(self) -> None:
        self._nodes = preset_nodes(self._element_type)
        self._canvas.set_nodes(self._nodes)
        self._refresh_quality()

    def _nodes_changed(self, nodes: object) -> None:
        self._nodes = np.asarray(nodes, dtype=np.float64)
        self._refresh_quality()

    def _thresholds_changed(self) -> None:
        try:
            thresholds = QualityThresholds(
                minimum_scaled_jacobian=float(self._minimum_scaled_edit.text()),
                maximum_aspect_ratio=float(self._maximum_aspect_edit.text()),
            )
        except ValueError as error:
            QMessageBox.critical(self, "Invalid quality threshold", str(error))
            self._minimum_scaled_edit.setText(
                f"{self._thresholds.minimum_scaled_jacobian:.2f}"
            )
            self._maximum_aspect_edit.setText(
                f"{self._thresholds.maximum_aspect_ratio:.2f}"
            )
            return
        self._thresholds = thresholds
        self._refresh_quality()

    def _rebuild_coordinate_labels(self) -> None:
        while self._coordinates_layout.count():
            item = self._coordinates_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._coordinate_labels = []
        for _index in range(len(self._nodes)):
            label = QLabel()
            label.setProperty("role", "coordinate")
            self._coordinate_labels.append(label)
            self._coordinates_layout.addWidget(label)

    def _refresh_quality(self) -> None:
        quality = evaluate_element(self._nodes)
        status = classify_quality(quality, self._thresholds)
        values = {
            "jacobian_minimum": quality.jacobian_minimum,
            "jacobian_maximum": quality.jacobian_maximum,
            "scaled_jacobian_minimum": quality.scaled_jacobian_minimum,
            "scaled_jacobian_maximum": quality.scaled_jacobian_maximum,
            "aspect_ratio": quality.aspect_ratio,
        }
        for key, value in values.items():
            self._metric_labels[key].setText(format_metric(value))
        self._status_label.setText(status.label)
        self._status_detail_label.setText(status.detail)
        status_color = {"pass": _PASS, "warning": _WARNING, "invalid": _INVALID}[
            status.key
        ]
        self._status_label.setStyleSheet(
            f"color: {status_color}; font-size: 17px; font-weight: 700;"
        )
        self._canvas.set_status(status.key)
        for index, (label, point) in enumerate(
            zip(self._coordinate_labels, self._nodes, strict=True)
        ):
            label.setText(f"P{index + 1}   x {point[0]: .4f}   y {point[1]: .4f}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the native Mesher Tri3/Quad4 element quality explorer.",
    )
    parser.add_argument(
        "--element",
        choices=("tri", "quad"),
        default="tri",
        help="element shown at startup (default: tri)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the desktop GUI and block until its window is closed."""

    arguments = build_argument_parser().parse_args(argv)
    initial_type = ElementType.TRI3 if arguments.element == "tri" else ElementType.QUAD4
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Mesher Element Quality Explorer")
    window = ElementQualityWindow(initial_type)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - exercised as an application
    raise SystemExit(main())
