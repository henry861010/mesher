import random
import sys
from pathlib import Path

# When this file is run directly, Python only adds script/ to sys.path.
# Add the project root so its sibling packages can be imported reliably.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viewer import view_mesh
from checkerboard import checkerboard_box
from circle import circle 
from mesh_quality import MeshQualityChecker


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
    element_size = 5

    x_list = generate_random_float_list(-100, 100, 1)
    y_list = generate_random_float_list(-100, 100, 2)
    mesh = checkerboard_box(element_size, x_list, y_list)
    
    # view_mesh(
    #     mesh, 
    #     reference_circles=[
    #         [0, 0, 50],
    #         [0, 0, 50 - element_size],
    #         [0, 0, 50 + element_size]
    #     ],    
    # )

    mesh = circle(
        mesh = mesh,
        x = 0,
        y = 0,
        radius = 50,
        buffer = element_size,
        lines=[[[x_list[50], y_list[49]],[x_list[50], y_list[5]]]]
    )

    checker = MeshQualityChecker(mesh)
    report = checker.check_scaled_jacobian(minimum=0.3)
    low_quality_indices = report.failed_indices

    print(
        f"Elements with Jacobian < {JACOBIAN_THRESHOLD}: "
        f"{low_quality_indices.tolist()}"
    )
    view_mesh(
        mesh, 
        element_indices = low_quality_indices,
        reference_circles=[[0, 0, 50]],    
    )


if __name__ == "__main__":
    main()
