# Package architecture

The dependency direction is intentionally one-way:

```text
mesher.mesh2d ─┐
               ├── mesher.process_flow
mesher.mesh3d ─┘
```

`mesh2d` and `mesh3d` are reusable dimension-specific libraries and never
import `process_flow`. The process-flow package is an orchestration and
translation boundary that may depend on both mesh domains and on the optional
`process_flow_kernel` package.

Within `process_flow`, `translation.standard_v1.StandardV1Translator` converts
Standard V1 payloads into planar faces and ordered layer-assignment
dictionaries, `domain` owns symmetry clipping, and `circle_planning` owns
circular feature decisions. `mesh3d.extrusion.Dragger` rasterizes those
material events and extrudes the selected planar elements; `pipeline`
coordinates these pieces. The CDB writer remains under `process_flow` because
it implements the repository's text contract rather than a general ANSYS CDB
serializer.

PyVista, Matplotlib, PySide6, and `process_flow_kernel` must not be imported by
the core `mesher` package. Optional entry points raise an actionable import
error when their extra is missing.
