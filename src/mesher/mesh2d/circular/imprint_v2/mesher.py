.utils import (
    _imprint_circle,
    _remove_redundant_element,
    _to_circle,
)

def imprint_circle(
    mesh: Mesh2D,
    *,
    center: ArrayLike,
    radius: float,
    tolerance: float,
    guide_tolerance: float,
    guide_segments: ArrayLike | None = None,
) -> Mesh2D:
    mesh = _to_circle(
        mesh = mesh,
        center = center
        radius = radius,
        tolerance = guide_tolerance,
        guide_tolerance = guide_tolerance,
        guide_segments=guide_segments
    )
    
    mesh = _remove_redundant_element(
        mesh = mesh,
        tolerance = tolerance,
    )
    
    mesh = _imprint_circle(
        mesh = mesh,
        center = center
        radius = radius,
        tolerance = tolerance,
    )
    
    return mesh