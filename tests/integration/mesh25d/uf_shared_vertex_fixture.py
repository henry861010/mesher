"""Synthetic six-die UF case; no saved flows or company geometry."""

from process_flow_kernel import ProcessGeometryState


def uf_structure():
    state = ProcessGeometryState.create()
    state.initialize_box_layer(material="base", bottom_left=[-2, -2, 0],
                               top_right=[30, 30, 0], thickness=4)
    for x, y in ((0, 0), (0, 10), (10, 10), (10, 20), (20, 0), (20, 20)):
        die = ProcessGeometryState.create()
        die.initialize_box_layer(material="Si", bottom_left=[0, 0, 0],
                                 top_right=[6, 6, 0], thickness=6)
        state.place_geometry_state(die, x=x, y=y, bottom_z=4, anchor="bottomLeft")
    state.apply_under_fill(material="UF", thk=6, gap=14, key="uf")
    return state.to_geometry_structure()


def mesh_settings(*, symmetry="full", adapter=False):
    return dict(schemaVersion="1.1.0", unitSystem="um", mesher="process_flow_2_5d",
                globalElementSize=4, symmetry=symmetry, geometryTolerance=.001,
                regionMeshing="adapter" if adapter else "grid",
                controls=[{
                    "method": "3D_REGION", "coordinateMode": "absolute", "label": "UF focus",
                    "elementSize": .4, "region": {"type": "box",
                        "min": {"x": 8, "y": 14, "z": 0},
                        "max": {"x": 12, "y": 18, "z": 10}},
                }] if adapter else [])
