import numpy as np
from numpy.typing import ArrayLike

from ...model import Mesh2D

def _triangle_area(node1, node2, node3, tolerance=0.1):
    n1, n2, n3 = np.asarray(node1), np.asarray(node2), np.asarray(node3)
    
    # Return 0 for degenerate/invalid triangles (points too close together)
    if (np.allclose(n1, n2, atol=tolerance) or 
        np.allclose(n2, n3, atol=tolerance) or 
        np.allclose(n3, n1, atol=tolerance)):
        return 0.0
    
    # Calculate area using 2D determinant: 0.5 * |(x2-x1)(y3-y1) - (y2-y1)(x3-x1)|
    return 0.5 * abs((n2[0] - n1[0]) * (n3[1] - n1[1]) - (n2[1] - n1[1]) * (n3[0] - n1[0]))


def _element_area(
    nodes: ArrayLike,
):
    area1 = _triangle_area(nodes[0], nodes[1], nodes[2])
    area2 = _triangle_area(nodes[2], nodes[3], nodes[0])
    return area1 + area2


def _point_angle(
    center_x: float, 
    center_y: float,
    point: ArrayLike,
):
    """
    Returns the angle of a point on a circle in both radians and degrees.
    """
    px, py = point[:2]
    
    # Calculate the distances from the center
    dx = px - center_x
    dy = py - center_y
    
    # arctan2 returns the angle in radians from -pi to pi
    angle_rad = np.arctan2(dy, dx)
    
    # Convert to standard 0-360 degrees format
    angle_deg = np.degrees(angle_rad) % 360
    
    return angle_deg


def _point_from_angle(
    center_x: float, 
    center_y: float, 
    radius: float, 
    angle_degrees: float
):
    """
    Returns the (x, y) coordinates of a point on a circle given its angle in degrees.
    """
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle_degrees)
    
    # Calculate the x and y coordinates
    px = center_x + radius * np.cos(angle_rad)
    py = center_y + radius * np.sin(angle_rad)
    
    return np.array([px, py, 0])


def _find_circle_line_intersection(
        node1: ArrayLike, 
        node2: ArrayLike, 
        center_x: float,
        center_y: float,
        radius: float,
        tolerance: float = 0.1,
        is_segment=False
    ):
        """
            Finds the intersection(s) of a line/segment and a circle.
            
            Parameters:
            - node1: np.array of shape (2,).
            - node2: np.array of shape (2,).
            - center_x, center_y: coordinates of the circle's center.
            - radius: radius of the circle.
            - tolerance: tolerance for floating point comparisons.
            - is_segment: if True, only returns intersections strictly between node1 and node2.
            
            Returns:
            - 1D np.array of the intersection coordinate, or None.
        """
        p1, p2 = node1, node2
        center = np.array([center_x, center_y, 0])

        # Direction vector of the line
        d = p2 - p1
        a = np.dot(d, d)

        if np.isclose(a, 0):
            raise ValueError("node1 and node2 cannot be the exact same point.")

        # Vector from circle center to node1
        f = p1 - center

        # Quadratic equation coefficients
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2

        discriminant = b**2 - 4 * a * c

        # No intersection
        if discriminant < -tolerance:
            return None

        # Clamp discriminant to 0 to handle floating point errors for exact tangents
        discriminant = max(0.0, discriminant)

        # Solve for t
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)

        # Deduplicate points if they are essentially a single tangent point
        t_values = [t1] if np.isclose(t1, t2, atol=tolerance) else [t1, t2]

        # Filter out intersections outside the segment bounds if is_segment is True
        if is_segment:
            t_values = [t for t in t_values if -tolerance <= t <= 1 + tolerance]

        if not t_values:
            return None

        # The intersection closest to node1 (p1) is the one with the smallest absolute t
        closest_t = min(t_values, key=abs)
        intersect_node = p1 + closest_t * d
        
        return intersect_node


def _move_element_node(
    mesh: Mesh2D,
    element_index: int,
    center_x: float,
    center_y: float,
    radius: float,
    criteria_area: float = 25,
    tolerance: float = 0.1
):
    if mesh.elements[element_index][2] == mesh.elements[element_index][3]:
        node_indices = mesh.elements[element_index][:3]
        is_quad = False
    else:
        node_indices = mesh.elements[element_index]
        is_quad = True
    
    # move the node of target element
    for index_local in range(node_indices.size):
        index_pre = node_indices[(index_local - 1 + node_indices.size) % node_indices.size]
        node_pre = mesh.nodes[index_pre]
        
        index = node_indices[index_local]
        node = mesh.nodes[index]
        
        index_post = node_indices[(index_local + 1) % node_indices.size]
        node_post = mesh.nodes[index_post]
        
        intersect1 = _find_circle_line_intersection(
            node1 = node, 
            node2 = node_pre, 
            center_x = center_x,
            center_y = center_y,
            radius = radius,
            tolerance = tolerance 
        )
        
        intersect2 = _find_circle_line_intersection(
            node1 = node, 
            node2 = node_post, 
            center_x = center_x,
            center_y = center_y,
            radius = radius,
            tolerance = tolerance 
        )

        if intersect1 is None and intersect2 is None:
            continue
        
        if intersect1 is not None and intersect2 is not None:
            area = _triangle_area(intersect1, node, intersect2)
            if area <= criteria_area:
                angle1 = _point_angle(center_x, center_y, intersect1)
                angle2 = _point_angle(center_x, center_y, intersect2)
                angle_degrees = (angle1 + angle2) / 2 % 360
                new_node = _point_from_angle(center_x, center_y, radius, angle_degrees)
                
                mesh.nodes[index] = new_node
            continue
        
        if intersect1 is None and intersect2 is not None:
            if is_quad:
                index_prepre = node_indices[(index_local - 2 + node_indices.size) % node_indices.size]
                node_prepre = mesh.nodes[index_prepre]
                
                intersect11 = _find_circle_line_intersection(
                    node1 = node_pre, 
                    node2 = node_prepre, 
                    center_x = center_x,
                    center_y = center_y,
                    radius = radius,
                    tolerance = tolerance 
                )
            
                if intersect11 is not None:
                    area = _triangle_area(intersect11, intersect1, node) + _triangle_area(intersect11, node, intersect2)
                    if area <= criteria_area:
                        mesh.nodes[index_pre] = intersect11
                        mesh.nodes[index] = intersect2

        if intersect1 is not None and intersect2 is None:
            if is_quad:
                index_postpost = node_indices[(index_local + 2) % node_indices.size]
                node_postpost = mesh.nodes[index_postpost]
                
                intersect22 = _find_circle_line_intersection(
                    node1 = node_post, 
                    node2 = node_postpost, 
                    center_x = center_x,
                    center_y = center_y,
                    radius = radius,
                    tolerance = tolerance 
                )
                
                if intersect22 is not None:
                    area = _triangle_area(intersect1, node, intersect22) + _triangle_area(node, intersect2, intersect22)
                    if area <= criteria_area:
                        mesh.nodes[index] = intersect1
                        mesh.nodes[index_post] = intersect22          
            
    # merge four node of the element if element area smaller than 0
    area = _element_area(mesh.nodes[node_indices])
    if area <= criteria_area:
        mesh.elements[element_index] = np.array([node_indices[0],node_indices[0],node_indices[0],node_indices[0]], dtype=np.int32)