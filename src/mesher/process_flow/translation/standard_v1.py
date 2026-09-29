import math


DEFAULT_TOLERANCE = 1e-6
CONTAINER_ITEM_FIELDS = ("bodies", "vias", "circuits", "bumps")

START_NORMAL = 3
START_DENSITY = 2
START_CONVERT = 1
END = 0

class StandardV1Translator:
    """Translates standard geometry containers into 2D print faces.

    This version does not support ``ConeGeometry``. ``CylinderGeometry`` is
    converted to a ``CIRCLE`` face.
    """

    def get_2D_pattern(self, container, mesh_control, tolerance=DEFAULT_TOLERANCE):
        """Extracts the 2D print pattern from a standard container tree.

        Args:
            container (dict): The root or subtree container payload from a
                standard geometry structure.
            tolerance (float): Numeric tolerance used when deduplicating
                completely overlapping faces. Defaults to ``1e-6``.

        Returns:
            tuple: A pair of ``(base_face, faces)``. ``base_face`` is the
            largest footprint face or ``None`` for an empty tree. ``faces``
            contains deduplicated non-base faces.

        Raises:
            ValueError: If the container contains unsupported geometry, invalid
                payload shape, or invalid tolerance.
        """
        normalized_tolerance = _normalize_tolerance(tolerance)
        all_faces = _collect_faces(container)
        unique_faces = _dedupe_faces(all_faces, normalized_tolerance)
        base_face = _select_base_face(unique_faces)
        faces = _remove_base_face(unique_faces, base_face, normalized_tolerance)

        return base_face, faces

    def get_3D_pattern(self, container, mesh_control):
        global_element_size = mesh_control["globalElementSize"]
        
        # assign priority
        _assign_priority(container)
        
        # get assignment
        assignments = _get_assignments(container)
        
        # order by priority and start/end
        assignments = sorted(assignments, key=lambda item: (item['z'], item['type']))
        
        # group by z
        layer_infos = []
        for assignment in assignments:
            if not layer_infos or layer_infos[-1]["z"] < assignment["z"]:
                layer_infos.append({
                    "z": assignment["z"],
                    "element_size": global_element_size,
                    "assignments": [assignment]
                })
            else:
                layer_infos[-1]["assignments"].append(assignment)
                
        # asign the z mesh control
        z_controls = _convert_to_objectless_mesh_controls(container, mesh_control)
        z_controls = sorted(z_controls, key=lambda item: (item['start'], item['element_size'], item['end']))
        
        unhandle_index = 0
        focus_stack = []
        
        layer_index = 0
        while layer_index < len(layer_infos):
            print(f'layer_z: {layer_infos[layer_index]["z"]}')
                        
            # remove the old mesh control
            stack_index = len(focus_stack)-1
            while stack_index >=0:
                layer_start = layer_infos[layer_index]["z"]
                control_end = focus_stack[stack_index]["end"]
                if _le(control_end, layer_start):
                    focus_stack.pop()
                    stack_index -= 1
                else:
                    break
                    
            # set the mesh control to focus if in interval
            while unhandle_index < len(z_controls):
                layer_start = layer_infos[layer_index]["z"]
                layer_end = layer_infos[layer_index+1]["z"]
                
                control_start = z_controls[unhandle_index]["start"]
                control_end = z_controls[unhandle_index]["end"]

                if _eq(control_start, layer_start):
                    # add to focus mesh control
                    focus_stack.append(z_controls[unhandle_index])
                    
                    # add end point
                    temp_index = layer_index
                    while temp_index < len(layer_infos[:-1]):
                        temp_layer_start = layer_infos[temp_index]["z"]
                        temp_layer_end = layer_infos[temp_index+1]["z"]
                        if _lt(temp_layer_start, control_end) and _lt(control_end, temp_layer_end):
                            layer_infos.insert(temp_index+1, {
                                "z": control_end,
                                "element_size": global_element_size,
                                "assignments": []
                            })
                        elif _lt(control_end, temp_layer_start):
                            break
                        temp_index += 1
                        
                    unhandle_index += 1
                                    
                elif _lt(control_start, layer_end):
                    # add to focus mesh control
                    focus_stack.append(z_controls[unhandle_index])
                    
                    # add new point for mesh control start
                    layer_infos.insert(layer_index+1, {
                        "z": control_start,
                        "element_size": global_element_size,
                        "assignments": []
                    })
                    
                    # add end point
                    temp_index = layer_index
                    while temp_index < len(layer_infos[:-1]):
                        temp_layer_start = layer_infos[temp_index]["z"]
                        temp_layer_end = layer_infos[temp_index+1]["z"]
                        if _lt(temp_layer_start, control_end) and _lt(control_end, temp_layer_end):
                            layer_infos.insert(temp_index+1, {
                                "z": control_end,
                                "element_size": global_element_size,
                                "assignments": []
                            })
                        elif _lt(control_end, temp_layer_start):
                            break
                        temp_index += 1
                        
                    unhandle_index += 1
                    
                else:
                    break
                
            # set focus element size
            element_size = global_element_size
            for control in focus_stack:
                if _le(control["start"], layer_infos[layer_index]["z"]):
                    element_size = min(element_size, control["element_size"])
            layer_infos[layer_index]["element_size"] = element_size
             
            layer_index += 1
            
        return layer_infos


def _lt(a, b, tol=0.01):
    if abs(a - b) < tol:
        return False
    return a < b

def _le(a, b, tol=0.01):
    if abs(a - b) < tol:
        return True
    return a < b

def _eq(a, b, tol=0.01):
    return abs(a - b) < tol


def _get_assignments(container, ancestors=None, path="root"):
    '''
        assignment {
            element_size: float
            z: float
            type:  0 / 1 / 2 / 3 (END / START_CONVERT / START_DENSITY / START_NORMAL)
            face: face
            areas: area[]
        }
        
        # area of START_NORMAL
        area {
            element_size: float
            face: face,
            priority: float
            material: str
        }
        
        # area of START_DENSITY
        area {
            element_size: float
            face: face,
            priority: float
            density: float
            material: str
        }
        
        # area of START_CONVERT
        area {
            element_size: float
            face: face,
            priority: float
            priority_o: float
            density: float
            material: str
        }
        
        # area of END
        area {
            element_size: float
            face: face,
            priority_o: float
            priority: float
            material: str
        }
        
        face {
            type: BOX / CIRCLE / POLYGON
            dim: []
        }
    '''
    assignments = []
    ancestors = [] if ancestors is None else ancestors
    
    for key in ["bodies", "bumps", "vias", "circuits"]:
        for term_index, term in enumerate(container[key]):
            source_ref = term.get("id") or f"{path}.{key}[{term_index}]"
            container_ref = container.get("id") or container.get("key") or path
            geometry_type = term.get("geometry", {}).get("type")
            feature_type = key[:-1] if key.endswith("s") else key
            
            if key == "bodies":
                geometry = term["geometry"]
                material = term["material"]
                priority = container["priority"]
                face = _geometry_to_face(geometry)
                
                # START
                assignments.append({
                    "z": _geometry_to_z(geometry, isStart=True),
                    "type": START_NORMAL,
                    "face": face,
                    "areas": [{
                        "face": None,
                        "priority": priority,
                        "material": material
                    }],
                    "diagnostic": {
                        "sourceRef": source_ref,
                        "containerRef": container_ref,
                        "featureType": feature_type,
                        "geometryType": geometry_type,
                        "operation": "start",
                    },
                })
                
                # END
                z = _geometry_to_z(geometry, isStart=False)
                areas = [{
                    "face": None,
                    "priority": 0,
                    "priority_o": priority,
                    "material": "EMPTY",
                }]
                for ancestor in ancestors:
                    for ancestor_body in ancestor["bodies"]:
                        ancestor_geometry = ancestor_body["geometry"]
                        z_start = _geometry_to_z(ancestor_geometry, isStart=True)
                        z_end = _geometry_to_z(ancestor_geometry, isStart=False)
                        if z_start < z and z < z_end:
                            ancestor_face = _geometry_to_face(ancestor_geometry)
                            areas.append({
                                "face": ancestor_face,
                                "priority": ancestor["priority"],
                                "priority_o": priority,
                                "material": ancestor_body["material"],
                            })
                
                assignments.append({
                    "z": z,
                    "type": END,
                    "face": face,
                    "areas": areas,
                    "diagnostic": {
                        "sourceRef": source_ref,
                        "containerRef": container_ref,
                        "featureType": feature_type,
                        "geometryType": geometry_type,
                        "operation": "end",
                    },
                }) 
            
            elif key in ["bumps", "vias", "circuits"]:
                geometry = term["geometry"]
                material = term["material"]
                koz = term["koz"]
                priority = container["priority"] + 0.5
                face = _geometry_to_face(geometry)
            
                # START
                assignments.append({
                    "z": _geometry_to_z(geometry, isStart=True),
                    "type": START_DENSITY,
                    "face": face,
                    "areas": [{
                        "face": None,
                        "priority": priority,
                        "material": material,
                        "density": term["density"],
                        "koz": koz
                    }],
                    "diagnostic": {
                        "sourceRef": source_ref,
                        "containerRef": container_ref,
                        "featureType": feature_type,
                        "geometryType": geometry_type,
                        "operation": "start_density",
                    },
                })
            
                # END
                z = _geometry_to_z(geometry, isStart=False)
                areas = [{
                    "face": None,
                    "priority": 0,
                    "priority_o": priority,
                    "material": "EMPTY",
                }]
                for ancestor in ancestors + [container]:
                    for ancestor_body in ancestor["bodies"]:
                        ancestor_geometry = ancestor_body["geometry"]
                        z_start = _geometry_to_z(ancestor_geometry, isStart=True)
                        z_end = _geometry_to_z(ancestor_geometry, isStart=False)
                        if z_start < z and z < z_end:
                            ancestor_face = _geometry_to_face(ancestor_geometry)
                            areas.append({
                                "face": ancestor_face,
                                "priority": ancestor["priority"],
                                "priority_o": priority,
                                "material": ancestor_body["material"],
                            })
                
                assignments.append({
                    "z": z,
                    "type": END,
                    "face": face,
                    "areas": areas,
                    "diagnostic": {
                        "sourceRef": source_ref,
                        "containerRef": container_ref,
                        "featureType": feature_type,
                        "geometryType": geometry_type,
                        "operation": "end",
                    },
                })     
        
    # child
    for child_index, child in enumerate(container["children"]):
        assignment_child = _get_assignments(
            child,
            ancestors=ancestors+[container],
            path=f"{path}.children[{child_index}]",
        )
        assignments = assignments + assignment_child
        
    return assignments


def _get_control_z_abs(control, obj_min, obj_max, global_element_size, min_dis=0.1):
    '''
        Z_SECTION_AVG
        Z_SECTION_TOP
        Z_SECTION_BOT
        Z_SECTION_CENTER
        Z_POINT
        
        return [{
            element_size: float
            start: float
            end: float
        }]
    '''
    def _parse_z(control_z, z_min, z_max):
        '''            
            {
                "mode": "relative",
                "anchor": "z_min",
                "offset": 10
            }
        '''
        # start z
        if control_z["mode"] == "absolute":
            start_z = float(control_z["value"]) 
        elif control_z["mode"] == "relative":
            if control_z["anchor"] == "z_min":
                start_z = z_min + float(control_z["offset"])
            elif control_z["anchor"] == "z_max":
                start_z = z_max + float(control_z["offset"])
            else:
                raise ValueError(f'Unknown anchor - {control_z["anchor"]}')
        return start_z
    
    if control["method"] == "Z_POINT":
        z = _parse_z(control["z"], obj_min, obj_max)
        
        return [{
            "element_size": global_element_size,
            "method": control["method"],
            "start": z,
            "end": z
        }]
        
    if control["method"] == "Z_SECTION_AVG":
        element_size = control["elementSize"]
        start_z = _parse_z(control["startZ"], obj_min, obj_max)
        end_z = _parse_z(control["endZ"], obj_min, obj_max)
        
        return [{
            "element_size": element_size,
            "method": control["method"],
            "start": start_z,
            "end": end_z
        }]
        
    if control["method"] == "Z_SECTION_TOP":
        element_size = control["elementSize"]
        start_z = _parse_z(control["startZ"], obj_min, obj_max)
        end_z = _parse_z(control["endZ"], obj_min, obj_max)
        
        if (end_z - start_z) % element_size < min_dis or (end_z - start_z) // element_size == 0:
            return [{
                "element_size": element_size,
                "method": control["method"],
                "start": start_z,
                "end": end_z
            }]
        else:
            center_z = end_z - (end_z - start_z) % element_size
            
            return [{
                "element_size": element_size,
                "method": control["method"],
                "start": start_z,
                "end": center_z
            }, {
                "element_size": element_size,
                "method": control["method"],
                "start": center_z,
                "end": end_z
            }]
        
    if control["method"] == "Z_SECTION_BOT":
        element_size = control["elementSize"]
        start_z = _parse_z(control["startZ"], obj_min, obj_max)
        end_z = _parse_z(control["endZ"], obj_min, obj_max)
        
        if (end_z - start_z) % element_size < min_dis or (end_z - start_z) // element_size == 0:
            return [{
                "element_size": element_size,
                "method": control["method"],
                "start": start_z,
                "end": end_z
            }]
        else:
            center_z = start_z + (end_z - start_z) % element_size
            
            return [{
                "element_size": element_size,
                "method": control["method"],
                "start": start_z,
                "end": center_z
            }, {
                "element_size": element_size,
                "method": control["method"],
                "start": center_z,
                "end": end_z
            }]
        
    if control["method"] == "Z_SECTION_CENTER":
        element_size = control["elementSize"]
        start_z = _parse_z(control["startZ"], obj_min, obj_max)
        end_z = _parse_z(control["endZ"], obj_min, obj_max)
        
        if (end_z - start_z) % element_size < min_dis or (end_z - start_z) // element_size <= 1:
            return [{
                "element_size": element_size,
                "method": control["method"],
                "start": start_z,
                "end": end_z
            }]
        else:
            center1_z = start_z + element_size * ((end_z - start_z) // element_size) / 2
            center2_z = center1_z + ((end_z - start_z) % element_size)
            
            return [{
                "element_size": element_size,
                "method": control["method"],
                "start": start_z,
                "end": center1_z
            }, {
                "element_size": element_size,
                "method": control["method"],
                "start": center1_z,
                "end": center2_z
            }, {
                "element_size": element_size,
                "method": control["method"],
                "start": center2_z,
                "end": end_z
            }]
        
    raise ValueError(f'Unknow tymethodpe - {control["method"]}')


def _convert_to_objectless_mesh_controls(container, mesh_control):
    z_controls = []
    
    controls = mesh_control["controls"]
    global_element_size = mesh_control["globalElementSize"]

    for control in controls:
        z_controls_sub = _convert_to_objectless_mesh_control(
            container, 
            control, 
            global_element_size
        )
       
        z_controls += z_controls_sub
        
    return z_controls


def _convert_to_objectless_mesh_control(container, control, global_element_size):
    '''        
        {
        "schemaVersion": "1.0.0",
        "unitSystem": "um",
        "mesher": "process_flow_2_5d",
        "globalElementSize": 200,
        "symmetry": "full",
        "controls": [
            {
                "method": "Z_SECTION_AVG",
                "reference": {
                    "kind": "container",
                    "key": "hbm",
                    "id": "container-id-optional"
                },
                "elementSize": 10,
                "startZ": {
                    "mode": "relative",
                    "anchor": "z_min",
                    "offset": 10
                },
                "endZ": {
                    "mode": "relative",
                    "anchor": "z_min",
                    "offset": 100
                }
            },
            {
                "method": "Z_POINT",
                "reference": {
                    "kind": "root"
                },
                "z": {
                    "mode": "relative",
                    "anchor": "z_max",
                    "offset": -10
                }
            }
        ]
        }
        
        
        return [{
            element_size: float
            start: float
            end: float
        }]
    '''
    def _is_target(object:dict, control_info:dict, kind:str):
        reference_info = control_info["reference"]
        
        if reference_info["kind"] != kind:
            return False
        if "id" in reference_info:
            if "id" not in object:
                return Fasle
            if reference_info["id"] != object["id"]:
                return False
        if "key" in reference_info:
            if "key" not in object:
                return False
            if reference_info["key"] != object["key"]:
                return False
        return True
        
    z_controls = []

    isUsed = False
    
    # container
    if _is_target(container, control, "container"):
        container_min, container_max = _bottom_top_z(container)
            
        z_controls_sub = _get_control_z_abs(
            control, 
            container_min, 
            container_max, 
            global_element_size,
            min_dis=0.1
        )
        
        z_controls += z_controls_sub
        isUsed = True
        
    # body
    if not isUsed:
        for body in container["bodies"]:
            if _is_target(body, control, "body"):
                body_min = _geometry_to_z(body["geometry"])
                body_max = body_min + body["thk"]
                
                z_controls_sub = _get_control_z_abs(
                    control, 
                    body_min, 
                    body_max, 
                    global_element_size
                )
        
                z_controls += z_controls_sub
                isUsed = True
                break
            
    # via
    if not isUsed:
        for via in container["vias"]:
            if _is_target(via, control, "via"):
                via_min = _geometry_to_z(via["geometry"])
                via_max = via_min + via["thk"]
                
                z_controls_sub = _get_control_z_abs(
                    control, 
                    via_min, 
                    via_max, 
                    global_element_size
                )
                
                z_controls += z_controls_sub
                isUsed = True
                break
                            
    # bump
    if not isUsed:
        for bump in container["bumps"]:
            if _is_target(bump, control, "bump"):
                bump_min = _geometry_to_z(bump["geometry"])
                bump_max = bump_min + bump["thk"]
                
                z_controls_sub = _get_control_z_abs(
                    control, 
                    bump_min, 
                    bump_max, 
                    global_element_size
                )
                
                z_controls += z_controls_sub
                isUsed = True
                break
            
    # circuit
    if not isUsed:
        for circuit in container["circuits"]:
            if _is_target(circuit, control, "circuit"):
                circuit_min = _geometry_to_z(circuit["geometry"])
                circuit_max = circuit_min + circuit["thk"]
                
                z_controls_sub = _get_control_z_abs(
                    control, 
                    circuit_min, 
                    circuit_max, 
                    global_element_size
                )
                
                z_controls += z_controls_sub
                isUsed = True
                break
        
    # child
    if not isUsed:
        for child in container["children"]:
            z_controls_sub = _convert_to_objectless_mesh_control(
                child, 
                control, 
                global_element_size
            )
            
            z_controls += z_controls_sub
            
    return z_controls
   
            
def _bottom_top_z(container):
    """Returns the Z bounds of all geometry in a container subtree.

    Bodies, bumps, circuits, vias, and geometries in descendant containers are
    included.  Both geometry endpoints are considered so the result remains
    ordered even if a geometry has a negative thickness.

    Raises:
        ValueError: If ``container`` is malformed or contains no geometry.
    """
    if not isinstance(container, dict):
        raise ValueError("container must be a dictionary")

    min_z = None
    max_z = None

    def include_container(current, path):
        nonlocal min_z, max_z

        if not isinstance(current, dict):
            raise ValueError(f"{path} must be a dictionary")

        for item_type in CONTAINER_ITEM_FIELDS:
            for item_index, item in enumerate(_collect_items(current, item_type)):
                geometry = _required_field(
                    item,
                    "geometry",
                    f"{path}.{item_type}[{item_index}]",
                )
                start_z = _finite_number(
                    _geometry_to_z(geometry, isStart=True),
                    f"{path}.{item_type}[{item_index}].geometry start Z",
                )
                end_z = _finite_number(
                    _geometry_to_z(geometry, isStart=False),
                    f"{path}.{item_type}[{item_index}].geometry end Z",
                )

                item_min_z = min(start_z, end_z)
                item_max_z = max(start_z, end_z)
                min_z = item_min_z if min_z is None else min(min_z, item_min_z)
                max_z = item_max_z if max_z is None else max(max_z, item_max_z)

        children = current.get("children", [])
        if children is None:
            children = []
        if not isinstance(children, list):
            raise ValueError(f"{path}.children must be a list")
        for child_index, child in enumerate(children):
            include_container(child, f"{path}.children[{child_index}]")

    include_container(container, "container")

    if min_z is None:
        raise ValueError("container does not contain any geometry")
    return min_z, max_z


def _assign_priority(container, priority=1):
    container["priority"] = priority
    for child in container["children"]:
        _assign_priority(child, priority=priority+1)


def _collect_faces(container):
    """Collects all 2D faces from a standard container subtree.

    Args:
        container (dict): The root or subtree container payload.

    Returns:
        list: All 2D face payloads found in the container subtree.

    Raises:
        ValueError: If the container or one of its geometry items is malformed.
    """
    if not isinstance(container, dict):
        raise ValueError("container must be a dictionary")

    faces = []
    for item_type in CONTAINER_ITEM_FIELDS:
        for item_index, item in enumerate(_collect_items(container, item_type)):
            geometry = _required_field(
                item,
                "geometry",
                f"container.{item_type}[{item_index}]",
            )
            faces.append(_geometry_to_face(geometry))

    children = container.get("children", [])
    if children is None:
        children = []
    if not isinstance(children, list):
        raise ValueError("container.children must be a list")

    for child in children:
        faces.extend(_collect_faces(child))

    return faces


def _collect_items(container, item_type):
    """Reads an item list from a container payload.

    Args:
        container (dict): The container payload to read from.
        item_type (str): One of ``bodies``, ``vias``, ``circuits``, or
            ``bumps``.

    Returns:
        list: The item payload list.

    Raises:
        ValueError: If the item field is present but is not a list.
    """
    items = container.get(item_type, [])
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError(f"container.{item_type} must be a list")
    return items


def _geometry_to_face(geometry):
    """Converts a standard geometry primitive into a 2D face.

    Args:
        geometry (dict): A standard geometry primitive payload.

    Returns:
        dict: A 2D face payload. Supported face types are ``BOX``,
        ``POLYGON``, and ``CIRCLE``.

    Raises:
        ValueError: If the geometry type is unsupported or malformed.
    """
    geometry_type = _required_field(geometry, "type", "geometry")

    if geometry_type == "BoxGeometry":
        x1, y1 = _point_xy(
            _required_field(geometry, "bottom_left", "BoxGeometry"),
            "BoxGeometry.bottom_left",
        )
        x2, y2 = _point_xy(
            _required_field(geometry, "top_right", "BoxGeometry"),
            "BoxGeometry.top_right",
        )
        return {"type": "BOX", "dim": [x1, y1, x2, y2]}

    if geometry_type == "PolygonGeometry":
        return {"type": "POLYGON", "dim": _polygon_dim(geometry)}

    if geometry_type == "CylinderGeometry":
        x, y = _point_xy(
            _required_field(geometry, "center", "CylinderGeometry"),
            "CylinderGeometry.center",
        )
        radius = _positive_number(
            _required_field(geometry, "bottom_radius", "CylinderGeometry"),
            "CylinderGeometry.bottom_radius",
        )
        return {"type": "CIRCLE", "dim": [x, y, radius]}

    if geometry_type == "ConeGeometry":
        raise ValueError("ConeGeometry is not supported by StandardV1Translator")

    raise ValueError(f"Geometry type {geometry_type} is not supported")


def _geometry_to_z(geometry, isStart=True):
    if geometry["type"] == "BoxGeometry":
        z = geometry["bottom_left"][2] 
        if not isStart:
            z += geometry["thk"]
        return z

    if geometry["type"] == "PolygonGeometry":
        z = geometry["polys"][0][0][2]
        if not isStart:
            z += geometry["thk"]
        return z
    
    if geometry["type"] == "CylinderGeometry":
        z = geometry["center"][2]
        if not isStart:
            z += geometry["thk"]
        return z
    
    if geometry["type"] == "ConeGeometry":
        raise ValueError("ConeGeometry is not supported by StandardV1Translator")


def _polygon_dim(geometry):
    """Builds a 2D polygon dimension payload from ``PolygonGeometry``.

    Args:
        geometry (dict): A ``PolygonGeometry`` payload.

    Returns:
        list: Polygon loops represented as ``[[[x, y], ...], ...]``.

    Raises:
        ValueError: If ``polys`` is missing or malformed.
    """
    polygons = _required_field(geometry, "polys", "PolygonGeometry")
    if not isinstance(polygons, list) or len(polygons) == 0:
        raise ValueError("PolygonGeometry.polys must be a non-empty list")

    polygons_2d = []
    for polygon_index, polygon in enumerate(polygons):
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(
                f"PolygonGeometry.polys[{polygon_index}] must contain at least 3 points"
            )
        polygons_2d.append(
            [
                _point_xy(
                    point,
                    f"PolygonGeometry.polys[{polygon_index}][{point_index}]",
                )
                for point_index, point in enumerate(polygon)
            ]
        )

    return polygons_2d


def _select_base_face(faces):
    """Selects the largest face as the base face.

    Args:
        faces (list): Candidate 2D face payloads.

    Returns:
        dict | None: The largest face, or ``None`` when no face exists.
    """
    base_face = None
    for face in faces:
        if base_face is None or _face_area(base_face) < _face_area(face):
            base_face = face
    return base_face


def _dedupe_faces(faces, tolerance):
    """Removes completely overlapping faces with tolerance.

    Args:
        faces (list): Candidate 2D face payloads.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        list: Deduplicated faces, preserving the first occurrence.
    """
    unique_faces = []
    for face in faces:
        if any(_same_face(face, unique_face, tolerance) for unique_face in unique_faces):
            continue
        unique_faces.append(face)
    return unique_faces


def _remove_base_face(faces, base_face, tolerance):
    """Removes the selected base face from a face list.

    Args:
        faces (list): Deduplicated 2D face payloads.
        base_face (dict | None): The selected base face.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        list: Faces that are not equivalent to ``base_face``.
    """
    if base_face is None:
        return faces
    return [face for face in faces if not _same_face(face, base_face, tolerance)]


def _same_face(left, right, tolerance):
    """Checks whether two faces have the same footprint.

    Args:
        left (dict): The first 2D face payload.
        right (dict): The second 2D face payload.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when both faces have the same type and equivalent
        footprint within tolerance.
    """
    if left["type"] != right["type"]:
        return False

    if left["type"] == "BOX":
        return _same_number_list(
            _normalized_box_dim(left["dim"]),
            _normalized_box_dim(right["dim"]),
            tolerance,
        )

    if left["type"] == "CIRCLE":
        return _same_number_list(left["dim"], right["dim"], tolerance)

    if left["type"] == "POLYGON":
        return _same_polygon_dim(left["dim"], right["dim"], tolerance)

    raise ValueError(f'Face type {left["type"]} is not supported')


def _same_polygon_dim(left_polygons, right_polygons, tolerance):
    """Checks whether two polygon dimensions contain the same loops.

    Args:
        left_polygons (list): Polygon loops from the first face.
        right_polygons (list): Polygon loops from the second face.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when all loops match, ignoring loop order, loop starting
        point, and clockwise/counterclockwise direction.
    """
    if len(left_polygons) != len(right_polygons):
        return False

    unmatched_right_indices = set(range(len(right_polygons)))
    for left_loop in left_polygons:
        matched_index = None
        for right_index in unmatched_right_indices:
            if _same_polygon_loop(left_loop, right_polygons[right_index], tolerance):
                matched_index = right_index
                break
        if matched_index is None:
            return False
        unmatched_right_indices.remove(matched_index)

    return True


def _same_polygon_loop(left_loop, right_loop, tolerance):
    """Checks whether two polygon loops are equivalent.

    Args:
        left_loop (list): The first polygon loop.
        right_loop (list): The second polygon loop.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when the loops match after allowing rotation and
        reversed direction.
    """
    left_open = _open_polygon_loop(left_loop, tolerance)
    right_open = _open_polygon_loop(right_loop, tolerance)
    if len(left_open) != len(right_open):
        return False

    return _same_loop_with_rotation(
        left_open,
        right_open,
        tolerance,
    ) or _same_loop_with_rotation(
        left_open,
        list(reversed(right_open)),
        tolerance,
    )


def _same_loop_with_rotation(left_loop, right_loop, tolerance):
    """Checks whether two polygon loops match under any start index.

    Args:
        left_loop (list): The first open polygon loop.
        right_loop (list): The second open polygon loop.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when a rotation of ``right_loop`` matches ``left_loop``.
    """
    if len(left_loop) != len(right_loop):
        return False
    if not left_loop:
        return True

    for start_index in range(len(right_loop)):
        matches = True
        for left_index, left_point in enumerate(left_loop):
            right_index = (start_index + left_index) % len(right_loop)
            if not _same_point(left_point, right_loop[right_index], tolerance):
                matches = False
                break
        if matches:
            return True

    return False


def _open_polygon_loop(loop, tolerance):
    """Removes a duplicate closing point from a polygon loop.

    Args:
        loop (list): A polygon loop represented as ``[[x, y], ...]``.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        list: The loop without a final point equal to the first point.
    """
    if len(loop) > 1 and _same_point(loop[0], loop[-1], tolerance):
        return loop[:-1]
    return loop


def _same_point(left, right, tolerance):
    """Checks whether two 2D points are equivalent within tolerance.

    Args:
        left (list): The first point represented as ``[x, y]``.
        right (list): The second point represented as ``[x, y]``.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when both coordinates match within tolerance.
    """
    return _same_number_list(left, right, tolerance)


def _same_number_list(left, right, tolerance):
    """Checks whether two numeric lists match within tolerance.

    Args:
        left (list): The first numeric list.
        right (list): The second numeric list.
        tolerance (float): Maximum allowed coordinate difference for equality.

    Returns:
        bool: ``True`` when both lists have the same length and each pair of
        numbers differs by no more than ``tolerance``.
    """
    if len(left) != len(right):
        return False
    return all(
        abs(left_value - right_value) <= tolerance
        for left_value, right_value in zip(left, right)
    )


def _normalized_box_dim(dim):
    """Normalizes a box dimension into ordered bounds.

    Args:
        dim (list): A box dimension represented as ``[x1, y1, x2, y2]``.

    Returns:
        list: Ordered box bounds as ``[x_min, y_min, x_max, y_max]``.
    """
    x1, y1, x2, y2 = dim
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _face_area(face):
    """Calculates the XY area of a 2D face.

    Args:
        face (dict): A 2D face payload.

    Returns:
        float: The absolute XY area of the face.

    Raises:
        ValueError: If the face type is unsupported.
    """
    if face["type"] == "BOX":
        x1, y1, x2, y2 = _normalized_box_dim(face["dim"])
        return abs((x2 - x1) * (y2 - y1))

    if face["type"] == "CIRCLE":
        _, _, radius = face["dim"]
        return math.pi * (radius ** 2)

    if face["type"] == "POLYGON":
        total_area = 0.0
        for polygon in face["dim"]:
            total_area += _signed_polygon_area(polygon)
        return abs(total_area)

    raise ValueError(f'Face type {face["type"]} is not supported')


def _signed_polygon_area(points):
    """Calculates the signed XY area for one polygon loop.

    Args:
        points (list): A polygon loop represented as ``[[x, y], ...]``.

    Returns:
        float: The signed polygon area.
    """
    area_value = 0.0
    for index in range(len(points)):
        next_index = (index + 1) % len(points)
        area_value += points[index][0] * points[next_index][1]
        area_value -= points[next_index][0] * points[index][1]
    return area_value / 2.0


def _required_field(payload, field_name, context):
    """Reads a required field from a dictionary payload.

    Args:
        payload (dict): The source payload to read from.
        field_name (str): The required field name.
        context (str): Human-readable payload context for error messages.

    Returns:
        object: The value stored under ``field_name``.

    Raises:
        ValueError: If ``payload`` is not a dictionary or the field is missing.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a dictionary")
    if field_name not in payload:
        raise ValueError(f"{context} missing field {field_name}")
    return payload[field_name]


def _point_xy(point, context):
    """Extracts the XY coordinates from a point payload.

    Args:
        point (list | tuple): A point payload with at least ``[x, y]``.
        context (str): Human-readable point context for error messages.

    Returns:
        list: The ``[x, y]`` coordinates.

    Raises:
        ValueError: If the point does not contain finite XY values.
    """
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        raise ValueError(f"{context} must be a point with at least [x, y]")
    return [
        _finite_number(point[0], f"{context}[0]"),
        _finite_number(point[1], f"{context}[1]"),
    ]


def _normalize_tolerance(tolerance):
    """Converts and validates the face deduplication tolerance.

    Args:
        tolerance (object): The candidate tolerance value.

    Returns:
        float: The validated positive tolerance.

    Raises:
        ValueError: If tolerance is not a positive finite number.
    """
    return _positive_number(tolerance, "tolerance")


def _positive_number(value, context):
    """Converts a value into a positive finite number.

    Args:
        value (object): The candidate numeric value.
        context (str): Human-readable value context for error messages.

    Returns:
        float: The converted positive finite number.

    Raises:
        ValueError: If the value is not a positive finite number.
    """
    number = _finite_number(value, context)
    if number <= 0:
        raise ValueError(f"{context} must be greater than 0")
    return number


def _finite_number(value, context):
    """Converts a value into a finite number.

    Args:
        value (object): The candidate numeric value.
        context (str): Human-readable value context for error messages.

    Returns:
        float: The converted finite number.

    Raises:
        ValueError: If the value cannot be converted to a finite number.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be a finite number")
    return number
