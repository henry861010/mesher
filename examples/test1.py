import random

from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.visualization import view_mesh
from mesher.mesh2d.circular.imprint_v2.utils import to_circle, imprint_circle, remove_redundant_element

random.seed(1)

def generate_random_float_list(begin, end, a, max_step=None):
    """
    Generates an ascending list of floats from 'begin' to 'end'.
    The distance between consecutive numbers is random, but at least 'a'.
    """
    # If no maximum step is provided, default it to double the minimum step
    if max_step is None:
        max_step = a * 2.0

    if a <= 0:
        raise ValueError("Minimum distance 'a' must be greater than 0.")
    if max_step < a:
        raise ValueError("'max_step' cannot be smaller than the minimum distance 'a'.")

    result = [begin]
    current = begin

    while True:
        # Generate a random distance between 'a' and 'max_step'
        step = random.uniform(a, max_step)
        current += step

        # Stop if adding the random step pushes us past the 'end' value
        if current > end:
            break

        result.append(current)

    # Optional: If you strictly want the very last number to be exactly 'end',
    # uncomment the next two lines:
    # if end - result[-1] >= a:
    #     result.append(float(end))

    return result

def main():
    center_x = 0
    center_y = 0
    radius = 57
    
    target_edge_size = 5

    x_coordinates = generate_random_float_list(-100, 100, 1)
    y_coordinates = generate_random_float_list(-100, 100, 2)
    mesh = generate_rectilinear_mesh(
        target_edge_size,
        x_coordinates,
        y_coordinates,
    )
    
    mesh = to_circle(
        mesh = mesh,
        center_x = center_x,
        center_y = center_y,
        radius = radius,
        tolerance = 1,
        guide_tolerance = 1,
        guide_segments=[]
    )
    
    mesh = remove_redundant_element(
        mesh = mesh,
        tolerance = 0.01,
    )
    
    # mesh = imprint_circle(
    #     mesh = mesh,
    #     center_x = center_x,
    #     center_y = center_y,
    #     radius = radius,
    #     tolerance = 1,
    # )
    
    view_mesh(
        mesh,
        reference_circles=[[center_x, center_y, radius]],
    )


if __name__ == "__main__":
    main()
