# The SustainaPrint UI (Figure 7 of the paper):  python ui.py [model.stl]
import os
import sys
import time

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QApplication, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                            QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget)

from sustainaprint import save, segment, simulate, surface, tetrahedralize

# weak filament presets: Young's modulus in MPa, Poisson's ratio
MATERIALS = {"Polymaker PolyTerra PLA": (1591, 0.36), "Recycled PLA": (1800, 0.36), "Ultimaker PLA": (2346, 0.36),
             "Ultimaker Tough PLA": (1820, 0.36), "PolyMax PLA": (1879, 0.36), "PolyLite PLA": (2636, 0.36)}
MODES = {"Fine-grained": "fine", "Layer-aligned": "layer"}
STRONG, WEAK, BOX = (0.9, 0.45, 0.45), (0.56, 0.64, 0.68), (1.0, 0.6, 0.2)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SustainaPrint UI")
        self.resize(980, 780)
        self.surface = self.mesh = self.stress = self.strong = None
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        self.plotter = QtInteractor(root)
        layout.addWidget(self.plotter.interactor, stretch=1)
        row = QHBoxLayout()
        layout.addLayout(row)
        for text, action in (("Load 3D Model", self.load), ("Update Bounding Boxes", self.draw_setup),
                             ("Run Simulation", self.run), ("Download Results", self.download)):
            button = QPushButton(text)
            button.clicked.connect(action)
            row.addWidget(button)
        grid = QGridLayout()
        layout.addLayout(grid)
        self.boxes = []
        for i, name in enumerate(("Fixed", "Force")):
            grid.addWidget(QLabel(name + " Bounding Box"), i, 0)
            self.boxes.append([QLineEdit() for _ in range(6)])
            for j, field in enumerate(self.boxes[i]):
                grid.addWidget(field, i, j + 1)
        self.slider_label = QLabel(alignment=Qt.AlignCenter)
        self.slider = QSlider(Qt.Horizontal, minimum=0, maximum=10, value=9, tickPosition=QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.resegment)
        layout.addWidget(self.slider_label)
        layout.addWidget(self.slider)
        row = QHBoxLayout()
        layout.addLayout(row)
        row.addWidget(QLabel("Force Direction:"))
        self.direction = QLineEdit("0 0 -1")
        self.direction.editingFinished.connect(self.draw_setup)
        self.material = QComboBox()
        self.material.addItems(MATERIALS)
        self.material.currentTextChanged.connect(self.preset)
        self.mode = QComboBox()
        self.mode.addItems(MODES)
        self.mode.currentIndexChanged.connect(self.resegment)
        self.islands = QPushButton("Remove Islands", checkable=True)
        self.shell = QPushButton("Print with Uniform Exterior", checkable=True)
        for widget in (self.direction, self.material, self.mode, self.islands, self.shell):
            row.addWidget(widget)
        self.islands.toggled.connect(self.resegment)
        self.shell.toggled.connect(self.resegment)
        row = QHBoxLayout()  # material properties and heuristic settings
        layout.addLayout(row)
        self.E, self.nu, self.island_pct, self.layers = (QLineEdit(text) for text in ("1591", "0.36", "3", "1"))
        for label, field in (("Weak filament E (MPa):", self.E), ("ν:", self.nu),
                             ("Max island (% of volume):", self.island_pct), ("Shell layers:", self.layers)):
            row.addWidget(QLabel(label))
            row.addWidget(field)
        row.addStretch()
        self.resegment()
        self.status("Load a 3D model (STL or OBJ) to begin.")

    def status(self, text):
        self.statusBar().showMessage(text)
        QApplication.processEvents()

    def load(self, path=None):
        path = path or QFileDialog.getOpenFileName(self, "Open 3D Model", os.path.join(os.path.dirname(__file__), "examples"),
                                                   "3D model (*.stl *.obj)")[0]
        if not path:
            return
        self.path, self.mesh, self.stress, self.strong = path, None, None, None
        surface = pv.read(path)
        self.origin = np.array(surface.bounds[::2])  # the model is shown with its minimum corner at the origin
        self.surface = surface.translate(-self.origin)
        x, y, z = self.surface.bounds[1::2]
        self.set_box(0, [0, 0, 0, x, y, 0.1 * z])  # default: held at the bottom, pushed down on top
        self.set_box(1, [0, 0, 0.9 * z, x, y, z])
        self.draw_setup()
        self.plotter.view_isometric()
        self.status(f"Loaded {os.path.basename(path)}. Set the bounding boxes and the force direction, then Run Simulation.")

    def box(self, i):
        v = np.array([float(field.text()) for field in self.boxes[i]])
        return np.r_[np.minimum(v[:3], v[3:]), np.maximum(v[:3], v[3:])]

    def set_box(self, i, values):
        for field, value in zip(self.boxes[i], values):
            field.setText(f"{value:g}")

    def force_direction(self):
        d = np.array(self.direction.text().split(), float)
        return d / np.linalg.norm(d)

    def draw_setup(self):
        if self.surface is None:
            return
        p = self.plotter
        p.clear()
        p.add_mesh(self.surface, color="#B0BEC5", opacity=0.7)
        for i in range(2):
            x0, y0, z0, x1, y1, z1 = self.box(i)
            p.add_mesh(pv.Box((x0, x1, y0, y1, z0, z1)), color=BOX, opacity=0.3)
        x0, y0, z0, x1, y1, z1 = self.box(1)
        centre, d = np.array([x0 + x1, y0 + y1, z0 + z1]) / 2, self.force_direction()
        size = 0.2 * np.linalg.norm(self.surface.bounds[1::2])
        p.add_mesh(pv.Arrow(centre - d * size, d, scale=size), color="red")
        p.add_point_labels([centre], ["Force Direction"], show_points=False, font_size=12)
        p.show_grid(xtitle="X Axis", ytitle="Y Axis", ztitle="Z Axis")
        p.render()

    def run(self):
        if self.surface is None:
            return self.status("Load a model first.")
        start = time.time()
        if self.mesh is None:
            self.status("Meshing with fTetWild...")
            self.mesh = tetrahedralize(self.path)
        self.status(f"Solving with SfePy ({len(self.mesh[1])} tetrahedra)...")
        shift = np.r_[self.origin, self.origin]
        self.stress, _ = simulate(*self.mesh, self.box(0) + shift, self.box(1) + shift, self.force_direction(),
                                  E=float(self.E.text()), nu=float(self.nu.text()))
        self.resegment()
        self.status(f"Simulation completed in {time.time() - start:.1f}s. {self.summary()}")

    def resegment(self, *_):
        self.slider_label.setText(f"Percentage Weak Filament: {10 * self.slider.value()}%")
        if self.stress is None:
            return
        points, tets = self.mesh
        self.strong = segment(self.stress, points, tets, self.slider.value() / 10, MODES[self.mode.currentText()],
                              shell_layers=int(self.layers.text()) if self.shell.isChecked() else 0,
                              island_max_fraction=float(self.island_pct.text()) / 100 if self.islands.isChecked() else 0)
        p = self.plotter
        p.clear()
        for mask, color in ((self.strong, STRONG), (~self.strong, WEAK)):
            if mask.any():
                p.add_mesh(surface(points - self.origin, tets[mask]), color=color, opacity=0.7)
        p.show_grid(xtitle="X Axis", ytitle="Y Axis", ztitle="Z Axis")
        p.render()
        self.status(self.summary())

    def summary(self):
        return f"Strong filament: {100 * self.strong.mean():.1f}% of the tetrahedra. Move the slider or change the options to re-segment."

    def preset(self, name):
        self.E.setText(str(MATERIALS[name][0]))
        self.nu.setText(str(MATERIALS[name][1]))

    def download(self):
        if self.strong is None:
            return self.status("Run the simulation first.")
        folder = QFileDialog.getExistingDirectory(self, "Save high_stress.stl and low_stress.stl to")
        if folder:
            save(*self.mesh, self.strong, folder)
            self.status(f"Saved high_stress.stl and low_stress.stl to {folder}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    if len(sys.argv) > 1:
        window.load(sys.argv[1])
    sys.exit(app.exec())
