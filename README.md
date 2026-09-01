# mesher

`mesher` is a typed Python library for planar mixed Tri3/Quad4 meshes,
layered Wedge6/Hex8 extrusion, mesh quality checks, visualization, and the
process-flow geometry-to-CDB workflow.

The package is organized by responsibility:

- `mesher.mesh2d` owns planar models, generators, circular operations,
  quality checks, and 2D visualization.
- `mesher.mesh3d` owns solid models, process-flow extrusion, and 3D visualization.
- `mesher.process_flow` translates Standard V1 geometry and coordinates the
  complete 2D-to-3D process-flow pipeline.

See [Architecture](docs/architecture.md) for the dependency rules and
[Migrating to 0.2](docs/migration-0.2.md) for breaking import and field changes.

## Installation

Install the NumPy-only core:

```bash
python -m pip install .
```

Install optional capabilities independently:

```bash
python -m pip install '.[visualization]'
python -m pip install '.[gui]'
python -m pip install '.[process-flow]'
```

The basic `import mesher` path does not import PyVista, Matplotlib, PySide6,
or `process_flow_kernel`.

## Mesh models

The package root exports only the central mesh models and the planar topology enum:

```python
from mesher import ElementType2D, Mesh2D, Mesh3D
```

`Mesh2D` owns contiguous `float64` nodes with shape `(n, 3)` and `int32`
connectivity with shape `(m, 4)`. XY input is accepted and normalized with a
zero Z coordinate. Tri3 rows use `[n0, n1, n2, n2]`; `element_types` exposes
the inferred Tri3/Quad4 topology.

`Mesh3D` owns `(n, 3)` nodes, fixed-width `(m, 8)` connectivity,
`element_comps`, and `comps`. Wedge-like connectivity remains padded to eight
slots by repeating the third bottom and top nodes so the repository CDB format
stays stable.

## 2D generation and circular features

```python
from mesher.mesh2d.circular import extend_circular_mesh, imprint_circle
from mesher.mesh2d.generators import generate_rectilinear_mesh

mesh = generate_rectilinear_mesh(
    target_edge_size=1.0,
    x_coordinates=[-10.0, 0.0, 10.0],
    y_coordinates=[-10.0, 0.0, 10.0],
)

imprint_circle(
    mesh,
    center=(0.0, 0.0),
    radius=5.0,
    band_width=1.0,
    target_edge_size=0.5,
)

extend_circular_mesh(
    mesh,
    element_size=1.0,
    center_x=0.0,
    center_y=0.0,
    inner_radius=5.0,
    outer_radius=10.0,
)
```

Both circular operations are transactional and in place: they return the same
`Mesh2D` after success and leave it unchanged on failure.

## Process-flow 3D extrusion

```python
from mesher.mesh3d.extrusion import Dragger

dragger = Dragger()
dragger.set_2D(mesh.nodes, mesh.elements)
solid = dragger.build(layer_infos, element_size=1.0)
```

`layer_infos` is the ordered dictionary structure returned by
`StandardV1Translator.get_3D_pattern()`. Component id zero represents an empty
planar element, and adjacent layers reuse their shared top/bottom nodes.

## Process-flow pipeline

Install the `process-flow` extra, then build from a Standard V1 structure:

```python
from mesher.process_flow import SymmetryMode, build_mesh_from_structure

mesh = build_mesh_from_structure(
    structure,
    element_size=100.0,
    symmetry=SymmetryMode.UPPER_RIGHT_QUARTER,
    progress=optional_event_callback,
)
```

Supported symmetry values are `full`, `upper_half`, `right_half`, and
`upper_right_quarter`.

Run the worker through its console script or module path:

```bash
mesher-process-flow geometry.json 100 output.cdb upper_right_quarter
python -m mesher.process_flow.worker geometry.json 100 output.cdb full
```

Success writes JSON metadata as the final stdout line. Progress events retain
the `PROCESS_FLOW_PROGRESS ` stderr prefix.

## Optional tools

Launch the 2D quality GUI with:

```bash
mesher-quality-gui
```

Import visualization only after installing the corresponding extra:

```python
from mesher.mesh2d.visualization import view_mesh
from mesher.mesh3d.visualization import MeshViewer
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m build
```
