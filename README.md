# mesher

`mesher` is a Python library for planar mixed Tri3/Quad4 meshes. It provides
rectilinear mesh generation, circular feature imprinting, mesh quality checks,
and optional PyVista visualization.

## Mesh data model

The former `Mesh` container has been renamed to `Mesh2D` to make the library's
planar, two-dimensional scope explicit and to avoid confusion with surface or
volume mesh types. Import it from the package root when constructing a mesh
directly:

```python
from mesher import Mesh2D

mesh = Mesh2D(nodes=nodes, elements=elements)
```

## Installation

Install the core package:

```bash
python -m pip install .
```

Install visualization support:

```bash
python -m pip install '.[visualization]'
```

## Circular feature imprinting

```python
from mesher.generators import generate_rectilinear_mesh
from mesher.circular import imprint_circle

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
```

The operation is transactional and in place: it returns the same `Mesh2D`
instance after success, while a failure leaves the input mesh unchanged.

`topology="auto"` is the default. It accepts either a complete circular band
or one connected open sector whose two domain-boundary sides lie on rays from
`center`. For example, a mesh covering only the upper half-plane can receive a
0-to-180-degree circular imprint:

```python
mesh = generate_rectilinear_mesh(
    target_edge_size=0.5,
    x_coordinates=[-8.0, 0.0, 8.0],
    y_coordinates=[0.0, 8.0],
)

imprint_circle(
    mesh,
    center=(0.0, 0.0),
    radius=5.0,
    band_width=1.0,
)
```

Pass `topology="closed"` or `topology="open"` to require one form explicitly.
Open cuts that do not follow rays from `center`, disconnected angular regions,
and ambiguous boundary topologies are rejected transactionally.

## Circular mesh extension

Use an existing exposed circular boundary to build concentric mesh layers out
to a larger radius:

```python
from mesher.circular import extend_circular_mesh

extend_circular_mesh(
    mesh,
    element_size=1.0,
    center_x=0.0,
    center_y=0.0,
    inner_radius=5.0,
    outer_radius=10.0,
)
```

The existing mesh outside `inner_radius` is discarded. The operation preserves
the inner ring's node count on every generated circle and returns the same
`Mesh2D` instance transactionally. `element_size` limits radial spacing only;
it does not limit circumferential edge length.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
