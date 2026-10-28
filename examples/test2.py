from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.adapter import QuadLayerResult, extend_quad_layer
from mesher.mesh2d.visualization import view_mesh
from mesher.mesh2d.editor import EditorMesh2D

editor = EditorMesh2D()

mesh1 = generate_rectilinear_mesh(
    target_edge_size = 5,
    x_coordinates = [0, 100],
    y_coordinates = [0, 100],
)
mesh2 = generate_rectilinear_mesh(
    target_edge_size = 5,
    x_coordinates = [100, 200],
    y_coordinates = [0, 200],
)
mesh = editor.merge(mesh1, mesh2)
mesh = editor.equivalence(mesh)

outlines = editor.outlines_by_edge(mesh)
outline = outlines[0]

result: QuadLayerResult = extend_quad_layer(
    mesh, outline, 50,
    pattern_lines=[], expand_ratio=1.5,
)

view_mesh(
    result.mesh
)
