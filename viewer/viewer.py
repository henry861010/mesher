import numpy as np
import pyvista as pv

from mesh import Mesh


def build_faces(mesh: Mesh):
    """Build PyVista's flat face array from a padded Tri3/Quad4 mesh.

    Triangles use the fixed-width representation ``[n0, n1, n2, n2]``.
    Every other row is emitted as a four-node face.
    """
    elements = np.asarray(mesh.elements)
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("elements must have shape (M, 4)")

    faces = []
    for element in elements:
        if element[2] == element[3]:
            faces.extend((3, *element[:3]))
        else:
            faces.extend((4, *element))
    return np.asarray(faces, dtype=np.int64)


def _validate_highlight_indices(
    indices,
    upper_bound,
    parameter_name,
    index_kind,
):
    """Return validated zero-based indices for a viewer highlight."""
    if indices is None:
        return np.empty(0, dtype=np.int64)

    selected_indices = np.asarray(indices)
    if selected_indices.ndim != 1:
        raise ValueError(f"{parameter_name} must be a one-dimensional sequence")
    if selected_indices.size:
        if not np.issubdtype(selected_indices.dtype, np.integer):
            raise TypeError(f"{parameter_name} must contain integers")
        selected_indices = selected_indices.astype(np.int64, copy=False)
        if np.any(selected_indices < 0) or np.any(selected_indices >= upper_bound):
            raise IndexError(
                f"{parameter_name} contain a {index_kind} index that is out of range"
            )
    else:
        selected_indices = np.empty(0, dtype=np.int64)

    return selected_indices


def _normalize_reference_geometry(values, item_shape, parameter_name):
    """Return reference geometry as a finite float array of individual items."""
    if values is None:
        return np.empty((0, *item_shape), dtype=np.float64)

    try:
        geometry = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{parameter_name} must contain numeric coordinates") from error

    if geometry.size == 0:
        return np.empty((0, *item_shape), dtype=np.float64)

    if geometry.shape == item_shape:
        geometry = geometry.reshape((1, *item_shape))
    elif geometry.ndim != len(item_shape) + 1 or geometry.shape[1:] != item_shape:
        expected_shape = "(" + ", ".join(map(str, item_shape)) + ")"
        raise ValueError(
            f"each item in {parameter_name} must have shape {expected_shape}"
        )

    if not np.all(np.isfinite(geometry)):
        raise ValueError(f"{parameter_name} must contain only finite values")

    return geometry


def _build_reference_segments(
    reference_circles,
    reference_boxes,
    reference_lines,
):
    """Build paired 3D endpoints for the reference geometry."""
    circles = _normalize_reference_geometry(
        reference_circles,
        (3,),
        "reference_circles",
    )
    boxes = _normalize_reference_geometry(
        reference_boxes,
        (2, 2),
        "reference_boxes",
    )
    lines = _normalize_reference_geometry(
        reference_lines,
        (2, 2),
        "reference_lines",
    )

    if circles.size and np.any(circles[:, 2] <= 0.0):
        raise ValueError("circle radii in reference_circles must be positive")
    if boxes.size and np.any(boxes[:, 0] > boxes[:, 1]):
        raise ValueError(
            "each reference_boxes bottom-left coordinate must not exceed "
            "its top-right coordinate"
        )

    segments = []
    circle_angles = np.linspace(0.0, 2.0 * np.pi, 129)
    for x, y, radius in circles:
        circle_points = np.column_stack(
            (
                x + radius * np.cos(circle_angles),
                y + radius * np.sin(circle_angles),
                np.zeros_like(circle_angles),
            )
        )
        segments.append(
            np.column_stack((circle_points[:-1], circle_points[1:])).reshape(-1, 3)
        )

    for bottom_left, top_right in boxes:
        x_min, y_min = bottom_left
        x_max, y_max = top_right
        corners = np.array(
            [
                [x_min, y_min, 0.0],
                [x_max, y_min, 0.0],
                [x_max, y_max, 0.0],
                [x_min, y_max, 0.0],
            ]
        )
        next_corners = np.roll(corners, -1, axis=0)
        segments.append(np.column_stack((corners, next_corners)).reshape(-1, 3))

    if lines.size:
        line_points = np.pad(lines, ((0, 0), (0, 0), (0, 1)))
        segments.append(line_points.reshape(-1, 3))

    if not segments:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(segments)


def view_mesh(
    mesh: Mesh,
    element_indices=None,
    node_indices=None,
    reference_circles=None,
    reference_boxes=None,
    reference_lines=None,
):
    """Display a mesh with highlighted elements and nodes.

    Parameters
    ----------
    mesh:
        Mesh to display.
    element_indices:
        Zero-based element indices to highlight in red. All other elements
        are shown in light blue.
    node_indices:
        Zero-based node indices to highlight in yellow.
    reference_circles:
        One circle ``[x, y, radius]`` or a sequence of circles to draw as
        black reference lines.
    reference_boxes:
        One box ``[[bottom_left_x, bottom_left_y], [top_right_x, top_right_y]]``
        or a sequence of boxes to draw as black reference lines.
    reference_lines:
        One line ``[[x1, y1], [x2, y2]]`` or a sequence of lines to draw as
        black reference lines.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh instance")

    # PyVista requires the faces array to be 1D and formatted as:
    # [n_points, id0, id1, id2, id3, n_points, id0, ...]
    faces = build_faces(mesh)
    
    # Construct the PyVista mesh object
    poly_data = pv.PolyData(mesh.nodes, faces)
    
    selected_element_indices = _validate_highlight_indices(
        element_indices,
        poly_data.n_cells,
        "element_indices",
        "element",
    )
    selected_node_indices = _validate_highlight_indices(
        node_indices,
        poly_data.n_points,
        "node_indices",
        "node",
    )
    reference_segments = _build_reference_segments(
        reference_circles,
        reference_boxes,
        reference_lines,
    )

    # Initialize the interactive plotter
    plotter = pv.Plotter()
    if poly_data.n_cells:
        # Use per-cell RGB values so highlighted and normal elements remain
        # part of the same actor and do not overlap or flicker.
        cell_colors = np.tile(
            np.array([173, 216, 230], dtype=np.uint8),
            (poly_data.n_cells, 1),
        )
        cell_colors[selected_element_indices] = np.array(
            [255, 0, 0],
            dtype=np.uint8,
        )
        poly_data.cell_data["face_colors"] = cell_colors
        plotter.add_mesh(
            poly_data,
            scalars="face_colors",
            rgb=True,
            show_edges=True,
            line_width=1.5,
        )

    if selected_node_indices.size:
        plotter.add_points(
            poly_data.points[selected_node_indices],
            color="yellow",
            point_size=12,
            render_points_as_spheres=True,
        )

    if reference_segments.size:
        plotter.add_lines(
            reference_segments,
            color="black",
            width=2,
            connected=False,
        )

    # Keep this 2D viewer in a top-down XY view.  Parallel projection avoids
    # perspective distortion, and mapping every drag gesture to pan/zoom
    # prevents the camera from rotating away from the top view.
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.enable_custom_trackball_style(
        left='pan',
        shift_left='pan',
        control_left='pan',
        middle='pan',
        shift_middle='pan',
        control_middle='pan',
        right='dolly',
        shift_right='dolly',
        control_right='dolly',
    )
    plotter.add_axes()
    plotter.show()
