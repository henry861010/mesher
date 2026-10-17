"""Geometry material events; _get_assignments is preserved verbatim."""

from .primitives import _geometry_to_face, _geometry_to_z
from .koz import inset_face, koz_value

START_NORMAL = 3
START_DENSITY = 2
START_CONVERT = 1
END = 0


def _assign_priority(container, priority=1):
    container["priority"] = priority
    for child in container["children"]:
        _assign_priority(child, priority=priority+1)


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
                if key == "bumps" and term["stage"] != "joined":
                    continue
                
                geometry = term["geometry"]
                material = term["material"]
                koz = koz_value(term)
                priority = container["priority"] + 0.5
                face = _geometry_to_face(geometry)
                # An empty polygon explicitly selects no elements. None keeps
                # the legacy density-area convention for zero KOZ.
                density_face = None
                if koz > 0:
                    density_face = inset_face(face, koz)
                    if density_face is None:
                        density_face = {"type": "POLYGON", "dim": []}
            
                # START
                assignments.append({
                    "z": _geometry_to_z(geometry, isStart=True),
                    "type": START_DENSITY,
                    "face": face,
                    "areas": [{
                        "face": density_face,
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


def extract_assignments(container):
    """Annotate the owned container copy and return its material events."""
    _assign_priority(container)
    return _get_assignments(container)
