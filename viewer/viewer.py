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


def view_mesh(mesh: Mesh, element_indices=None, node_indices=None):
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
