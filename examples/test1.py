from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.visualization import view_mesh
from mesher.mesh2d.circular.imprint_v2.main import _move_element_node

def main():
    target_edge_size = 5
    x_coordinates = [40, 50]
    y_coordinates = [40, 50]
    center_x = 0
    center_y = 0
    radius = 58
    
    mesh = generate_rectilinear_mesh(
        target_edge_size,
        x_coordinates,
        y_coordinates,
    )
    
    _move_element_node(
        mesh = mesh,
        element_index = 0,
        center_x = center_x,
        center_y = center_y,
        radius = radius,
        criteria_area = 5,
    )

    view_mesh(
        mesh,
        reference_circles=[[center_x, center_y, radius]],
    )


if __name__ == "__main__":
    main()
