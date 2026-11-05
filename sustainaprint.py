# SustainaPrint: strong filament where the stress is high, eco-friendly filament everywhere else.
# Paper: https://doi.org/10.1145/3746059.3747640
import os
import tempfile

import numpy as np
import pyvista as pv
import scipy.sparse
from scipy.sparse.csgraph import connected_components
from sfepy.base.base import output
from sfepy.discrete import Equation, Equations, FieldVariable, Integral, Material, Problem
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.discrete.fem import FEDomain, Field, Mesh
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.terms import Term

output.set_output(quiet=True)


def tetrahedralize(path):
    """Tetrahedral mesh (points, tets) of an STL/OBJ file, fTetWild with its default settings."""
    import wildmeshing
    path, cwd, stdout = os.path.abspath(path), os.getcwd(), os.dup(1)
    with tempfile.TemporaryDirectory() as tmp, open(os.devnull, "w") as null:  # fTetWild writes scratch files and logs
        os.chdir(tmp)
        os.dup2(null.fileno(), 1)
        try:
            mesher = wildmeshing.Tetrahedralizer(coarsen=False)
            mesher.load_mesh(path)
            mesher.tetrahedralize()
            points, tets = mesher.get_tet_mesh()[:2]
        finally:
            os.dup2(stdout, 1)
            os.close(stdout)
            os.chdir(cwd)
    return np.asarray(points, float), np.asarray(tets, np.int64)


def inside(box):
    x0, y0, z0, x1, y1, z1 = box
    return f"vertices in (x > {x0}) & (x < {x1}) & (y > {y0}) & (y < {y1}) & (z > {z0}) & (z < {z1})"


def simulate(points, tets, fixed_box, force_box, direction, magnitude=1.0, E=1591.0, nu=0.36):
    """Linear elasticity with the weak filament's E (MPa) and nu: the part is held still inside
    fixed_box and pushed inside force_box. Boxes are [x0, y0, z0, x1, y1, z1]. Returns the von
    Mises stress of every tetrahedron and the displacement of every node."""
    mesh = Mesh.from_data("part", points, None, [tets], [np.zeros(len(tets), np.int32)], ["3_4"])
    domain = FEDomain("part", mesh)
    part = domain.create_region("Part", "all")
    held = domain.create_region("Held", inside(fixed_box), "facet")
    pushed = domain.create_region("Pushed", inside(force_box), "facet")
    field = Field.from_args("displacement", np.float64, 3, part, approx_order=1)
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")
    solid = Material("solid", D=stiffness_from_youngpoisson(3, E, nu))
    load = Material("load", val=(magnitude * np.asarray(direction, float) / np.linalg.norm(direction)).reshape(3, 1))
    elastic = Term.new("dw_lin_elastic(solid.D, v, u)", Integral("i", order=1), part, solid=solid, v=v, u=u)
    traction = Term.new("dw_surface_ltr(load.val, v)", Integral("s", order=2), pushed, load=load, v=v)
    problem = Problem("sustainaprint", equations=Equations([Equation("balance", elastic - traction)]))
    problem.set_bcs(ebcs=Conditions([EssentialBC("held", held, {"u.all": 0.0})]))
    problem.set_solver(Newton({"i_max": 1, "eps_a": 1e-10}, lin_solver=ScipyDirect({})))
    state = problem.solve(save_results=False)
    s11, s22, s33, s12, s13, s23 = problem.evaluate("ev_cauchy_stress.1.Part(solid.D, u)", mode="el_avg", solid=solid, copy_materials=False).reshape(-1, 6).T
    von_mises = np.sqrt(np.maximum(0, ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) / 2 + 3 * (s12 ** 2 + s13 ** 2 + s23 ** 2)))
    return von_mises, state.get_state_parts()["u"].reshape(-1, 3)


def segment(stress, points, tets, weak_fraction=0.9, mode="fine", shell_layers=0, shell_material="strong", island_max_fraction=0):
    """True for every tetrahedron printed in the strong filament."""
    strong = most_stressed(stress, weak_fraction)
    if mode == "layer":
        strong = layer_aligned(strong, points, tets)
    if shell_layers or island_max_fraction:
        neighbours, on_surface = face_neighbours(tets)
    if shell_layers:
        strong = outer_shell(strong, neighbours, on_surface, shell_layers, shell_material)
    if island_max_fraction:
        strong = remove_islands(strong, neighbours, volumes(points, tets), island_max_fraction)
    return strong


def most_stressed(stress, weak_fraction):
    """The top (1 - weak_fraction) of the tetrahedra by stress, by default the top 10%."""
    count = int(np.floor((1 - weak_fraction) * len(stress) + 0.5))
    strong = np.zeros(len(stress), bool)
    strong[np.argsort(-stress, kind="stable")[:count]] = True
    return strong


def layer_aligned(strong, points, tets):
    """One planar split along z, the print direction: everything above the lowest strong
    tetrahedron or everything below the highest one, whichever needs less strong filament."""
    if not strong.any():
        return strong
    z = points[tets].mean(axis=1)[:, 2]
    above, below = z >= z[strong].min(), z <= z[strong].max()
    volume = volumes(points, tets)
    return above if volume[above].sum() <= volume[below].sum() else below


def outer_shell(strong, neighbours, on_surface, layers, material="strong"):
    """The outermost layers of tetrahedra in one filament, for a uniform look."""
    shell, front = on_surface.copy(), on_surface.copy()
    for _ in range(layers - 1):
        front = (neighbours @ front) & ~shell
        shell |= front
    return strong | shell if material == "strong" else strong & ~shell


def remove_islands(strong, neighbours, volume, max_fraction):
    """Strong pieces smaller than max_fraction of the part's volume become weak."""
    ids = np.flatnonzero(strong)
    _, piece = connected_components(neighbours[ids][:, ids], directed=False)
    small = np.bincount(piece, weights=volume[ids])[piece] < max_fraction * volume.sum()
    strong = strong.copy()
    strong[ids[small]] = False
    return strong


def face_neighbours(tets):
    """Which tetrahedra share a face (sparse adjacency) and which ones lie on the surface."""
    faces = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    owner = np.tile(np.arange(len(tets)), 4)
    _, face, count = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    face = face.ravel()
    on_surface = np.zeros(len(tets), bool)
    on_surface[owner[count[face] == 1]] = True
    order = np.argsort(face, kind="stable")
    shared = face[order][1:] == face[order][:-1]
    a, b = owner[order][:-1][shared], owner[order][1:][shared]
    neighbours = scipy.sparse.csr_matrix((np.ones(2 * len(a), bool), (np.r_[a, b], np.r_[b, a])), shape=(len(tets), len(tets)))
    return neighbours, on_surface


def volumes(points, tets):
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    return np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6


def surface(points, tets):
    """The boundary of these tetrahedra as a PyVista surface."""
    return pv.UnstructuredGrid({pv.CellType.TETRA: tets}, points).extract_surface()


def save(points, tets, strong, folder):
    """high_stress.stl (strong filament) and low_stress.stl (eco-friendly filament) for the slicer."""
    os.makedirs(folder, exist_ok=True)
    for name, chosen in (("high_stress.stl", strong), ("low_stress.stl", ~strong)):
        if chosen.any():
            surface(points, tets[chosen]).save(os.path.join(folder, name))
