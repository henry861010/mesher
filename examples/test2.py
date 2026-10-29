import numpy as np
from mesher.mesh2d.adapter import extend_quad_layer
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.visualization import view_mesh

pattern_lines = np.asarray([
    [[-50, 30], [50, 30]],
    [[-100, 63], [-50, 63]],
    [[15,50], [15, 200]]
])

editor = EditorMesh2D()
mesh1 = generate_rectilinear_mesh(
    target_edge_size=5,
    x_coordinates=[0, 100],
    y_coordinates=[0, 100],
)
mesh2 = generate_rectilinear_mesh(
    target_edge_size=5,
    x_coordinates=[100, 200],
    y_coordinates=[0, 200],
)
mesh = editor.equivalence(editor.merge(mesh1, mesh2))
outline = editor.outlines_by_edge(mesh)[0]
result = extend_quad_layer(mesh, outline, element_size=50, expand_ratio=1.5, pattern_lines=pattern_lines)

view_mesh(result.mesh, reference_lines=pattern_lines)

