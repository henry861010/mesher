# mesher

`mesher` is a Python library for planar mixed Tri3/Quad4 meshes. It provides
rectilinear mesh generation, circular feature imprinting, mesh quality checks,
and optional PyVista visualization.

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
from mesher.imprinting import imprint_circle

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

The operation is transactional and in place: it returns the same `Mesh`
instance after success, while a failure leaves the input mesh unchanged.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
