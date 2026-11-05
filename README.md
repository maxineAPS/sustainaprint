# SustainaPrint

Code for *SustainaPrint: Making the Most of Eco-Friendly Filaments* (UIST 2025,
[doi:10.1145/3746059.3747640](https://doi.org/10.1145/3746059.3747640)).

SustainaPrint simulates the stress in a 3D model, prints the most stressed regions in a strong
(virgin) filament and the rest in a weaker eco-friendly one, so most of the part is eco-friendly
without losing strength. It writes one STL per filament for a dual-extrusion printer.
A browser version runs at https://maxineaps.github.io/sustainaprint-project-site/.

## Install

Python 3.10 or newer:

    pip install -r requirements.txt

## Run

    python ui.py examples/wall_hook.stl

1. Load a model (STL or OBJ). It is shown with its lower corner at the origin.
2. Type the fixed bounding box (where the part is held) and the force bounding box (where the
   load acts) as x0 y0 z0 x1 y1 z1, and the force direction, then Update Bounding Boxes.
3. Run Simulation. The model is meshed with fTetWild and solved with SfePy using the weak
   filament's properties (pick a preset or type E and ν). This takes from seconds to a couple of
   minutes.
4. Move the slider to the percentage of weak filament, choose fine-grained (per tetrahedron) or
   layer-aligned (one planar split along Z, the print direction) segmentation, and optionally
   Remove Islands (strong pieces smaller than the given share of the volume become weak) and
   Print with Uniform Exterior (the outermost tetrahedra are printed in one filament).
5. Download Results writes high_stress.stl (strong filament) and low_stress.stl (eco-friendly
   filament). Slice them together.

From Python:

```python
from sustainaprint import tetrahedralize, simulate, segment, save

points, tets = tetrahedralize("examples/wall_hook.stl")
stress, _ = simulate(points, tets, fixed_box=[-1, -1, -1, 36, 0.5, 70],   # the back face, against the wall
                     force_box=[-1, 24, 20, 36, 38, 30], direction=[0, 0, -1])  # inside the curve, pulled down
strong = segment(stress, points, tets, weak_fraction=0.9)  # True for the tetrahedra printed in the strong filament
save(points, tets, strong, "out")
```

Boxes are in the file's coordinates. `segment` also takes `mode="layer"`, `shell_layers`,
`shell_material` ("strong" or "weak") and `island_max_fraction`; `simulate` takes `magnitude`,
`E` (MPa) and `nu`.

## Models

`examples/` has three CC0 models from Printables, see `examples/ATTRIBUTION.md`.

## License

MIT, see LICENSE. The dependencies are separate projects with their own licenses: NumPy, SciPy and
SfePy (BSD), PyVista and pyvistaqt (MIT), wildmeshing/fTetWild (MPL 2.0) and PySide6 (LGPL 3).
