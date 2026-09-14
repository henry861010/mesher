import random

from mesher.mesh2d.generators import generate_rectilinear_mesh
from mesher.mesh2d.visualization import view_mesh
from mesher.mesh2d.circular.imprint_v2.utils import (
    _imprint_circle,
    _remove_redundant_element,
    _to_circle,
)

random.seed(1)

def generate_random_float_list(
    begin,
    end,
    a,
    max_step=None,
    refs: list[float] | None = None,
):
    """
    Generates an ascending list of floats from 'begin' to 'end'.
    The distance between consecutive numbers is random, but at least 'a'.

    Values in 'refs' are guaranteed to be included. Random values closer than
    'a' to a reference value are removed, while reference values take priority.
    """
    # If no maximum step is provided, default it to double the minimum step
    if max_step is None:
        max_step = a * 2.0

    if a <= 0:
        raise ValueError("Minimum distance 'a' must be greater than 0.")
    if max_step < a:
        raise ValueError("'max_step' cannot be smaller than the minimum distance 'a'.")

    refs = [] if refs is None else [float(ref) for ref in refs]
    invalid_refs = [ref for ref in refs if not begin <= ref <= end]
    if invalid_refs:
        raise ValueError(f"All values in 'refs' must be between 'begin' and 'end': {invalid_refs}")

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

    if refs:
        result = [
            value
            for value in result
            if value == begin or all(abs(value - ref) >= a for ref in refs)
        ]
        result.extend(refs)

    return sorted(set(result))

def main():
    center_x = 0
    center_y = 0
    radius = 57

    target_edge_size = 5

    reference_lines = [
        [[10, 30], [10, 40]],
        [[10, 40], [40, 40]],
        [[40, 40], [40, 30]],
        [[40, 30], [10, 30]]
    ]

    x_coordinates = generate_random_float_list(-100, 100, 1, refs=[10,40])
    y_coordinates = generate_random_float_list(-100, 100, 2, refs=[30,40])
    mesh = generate_rectilinear_mesh(
        target_edge_size,
        x_coordinates,
        y_coordinates,
    )
    
    mesh = _to_circle(
        mesh = mesh,
        center_x = center_x,
        center_y = center_y,
        radius = radius,
        tolerance = 1,
        guide_tolerance = 1,
        guide_segments=reference_lines
    )
    
    mesh = _remove_redundant_element(
        mesh = mesh,
        tolerance = 0.01,
    )
    
    mesh = _imprint_circle(
        mesh = mesh,
        center_x = center_x,
        center_y = center_y,
        radius = radius,
        tolerance = 0.01,
    )
    
    view_mesh(
        mesh,
        reference_lines=reference_lines
    )


if __name__ == "__main__":
    main()
