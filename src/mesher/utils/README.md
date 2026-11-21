# Local die-corner control areas

```python
from mesher.utils import get_mm2_control_area

areas = get_mm2_control_area(
    [[0, 0, 100, 100], [180, 0, 280, 100]],
    corner_extent=25.0,
)
# array([[-25., -25.,  25.,  25.],
#        [-25.,  75.,  25., 125.],
#        [ 75., -25., 205.,  25.],
#        [ 75.,  75., 205., 125.],
#        [255., -25., 305.,  25.],
#        [255.,  75., 305., 125.]])
```

Input dies and output control areas use `[x1, y1, x2, y2]` coordinates.
Each die corner starts with a square extending 25 in each positive and
negative axis direction. `corner_extent` adjusts that distance in model units.

A gap requires positive overlap between opposing die sides and no intervening
die in the visible portion. If its transverse span contains a corner of either
participating die, that corner's patch extends across the gap and another
`corner_extent` past the opposite side. Set `max_gap` to the largest
edge-to-edge distance to bridge (inclusive, in model units). Its default
of `None` allows any width. Use `max_gap=0` for corner squares only.
Corner endpoints count; point-only overlap between die sides does not.
Touching dies have no positive-width gap. With partial obstruction, a corner
only uses the gap visible next to it, rather than a farther die behind a blocker.

Horizontal and vertical gap extensions belonging to the same corner combine
into one rectangular patch. This fills the central gap intersection in a
four-die layout. Patches only use gaps incident to their original die corner;
they do not recursively extend through other gaps. Distant die edges never
create global intersections. The output is the exact union of these patches,
decomposed into sorted, interior-disjoint rectangles. It preserves notches and
does not promise a minimum rectangle count.

The function uses only NumPy and the standard library. It does not generate a
mesh, snap coordinates, or clip to an exterior domain. Coordinates are compared
exactly; no mesh tolerance is applied. Input arrays are never modified. Empty
input returns a `float64` array with shape `(0, 4)`. Invalid boxes, overlapping
die interiors (including duplicate dies), invalid extent, and unrepresentable
expanded coordinates raise `ValueError`.

## Editable preview example

Edit `BOXES` and `CORNER_EXTENT` near the top of
[`examples/mm2_control_area.py`](../../../examples/mm2_control_area.py), then run:

```sh
PYTHONPATH=src venv/bin/python examples/mm2_control_area.py --show
```

The left panel shows input dies; the right overlays the returned control areas
in orange. D1, D2, ... identify input dies; C1, C2, ... identify output rectangles
and match the coordinates printed in the terminal. Both panels use equal axis
scales. The Matplotlib window supports zoom and pan. This previews control-area
geometry, not mesh elements.

The example also saves `control_areas.png`, `control_areas.json` (input and
output), and `control_areas.npy` (output only) under `build/mm2-control-area`.
Omit `--show` for file-only rendering, use `--output PATH` for another directory,
or use `--corner-extent 10` to override the distance without editing the script.
Commented alternative layouts include a T junction, diagonal dies, and a
partially blocked gap.

## Sampling for imprint lines

Spacing is independent of control-area geometry. To request spacing 5, sample
both axes of each returned rectangle and pass those points to `get_pair_line`:

```python
import numpy as np
from mesher.mesh2d.imprint_line import get_pair_line

def positions(low, high, spacing=5.0):
    # Include the exact endpoint; the last interval may be shorter than 5.
    return np.append(np.arange(low, high, spacing), high)

controls = [
    {
        "box": [x1, y1, x2, y2],
        "x_points": [[x1, y] for y in positions(y1, y2)],
        "y_points": [[x, y1] for x in positions(x1, x2)],
    }
    for x1, y1, x2, y2 in areas
]
pairs = get_pair_line(controls, expand_size=25.0)
```

Here `x_points` samples Y coordinates for the pair of vertical sides;
`y_points` samples X coordinates for the pair of horizontal sides.
`expand_size` controls the subsequent imprint extension independently of
`corner_extent`. If supplying an exterior outline, it must contain all control
areas, including the portions extending beyond the die boundaries. The mesh's
sampling and geometric tolerance requirements still apply downstream.

Run the geometric acceptance tests with:

```sh
PYTHONPATH=src venv/bin/python -m unittest tests.unit.utils.test_mm2_control_area
```
