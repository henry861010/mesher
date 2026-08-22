import random

from mesher.generators import generate_rectilinear_mesh
from mesher.imprinting import imprint_circle
from mesher.quality import MeshQualityChecker
from mesher.visualization import view_mesh


JACOBIAN_THRESHOLD = 0.3

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
    target_edge_size = 5

    x_coordinates = generate_random_float_list(-100, 100, 1)
    y_coordinates = generate_random_float_list(-100, 100, 2)
    mesh = generate_rectilinear_mesh(
        target_edge_size,
        x_coordinates,
        y_coordinates,
    )

    # view_mesh(
    #     mesh,
    #     reference_circles=[
    #         [0, 0, 50],
    #         [0, 0, 50 - element_size],
    #         [0, 0, 50 + element_size]
    #     ],
    # )

    mesh = imprint_circle(
        mesh,
        center=(0, 0),
        radius=50,
        band_width=target_edge_size,
        target_edge_size=2,
        guide_segments=[
            [
                [x_coordinates[50], y_coordinates[-1]],
                [x_coordinates[50], y_coordinates[0]],
            ]
        ],
    )

    checker = MeshQualityChecker(mesh)
    report = checker.check_scaled_jacobian(minimum=0.4)
    low_quality_indices = report.failed_indices

    view_mesh(
        mesh,
        element_indices = low_quality_indices,
        reference_circles=[[0, 0, 50]],
        reference_lines=[
            [[x_coordinates[50], -200], [x_coordinates[50], 200]]
        ],
    )


if __name__ == "__main__":
    main()
