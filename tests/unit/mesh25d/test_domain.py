import unittest

from mesher.mesh25d.domain import (
    ModelDomain,
    SymmetryMode,
    filter_container_to_domain,
)
from mesher.mesh25d.geometry.patterns import extract_pattern
from mesher.mesh25d.geometry.assignments import extract_assignments
from mesher.mesh25d.plan3d import plan_layers


def _box(x_min, y_min, x_max, y_max):
    return {
        "geometry": {
            "type": "BoxGeometry",
            "bottom_left": [x_min, y_min, 0.0],
            "top_right": [x_max, y_max, 0.0],
            "thk": 1.0,
        },
        "material": "Si",
    }


def _container(key, *, children=(), **items):
    return {
        "key": key,
        **{field: items.get(field, []) for field in ("bodies", "vias", "circuits", "bumps")},
        "children": list(children),
    }


class DomainFilteringTests(unittest.TestCase):
    def test_prunes_outside_containers_for_each_geometry_field_and_reduced_mode(self):
        for mode in (
            SymmetryMode.UPPER_HALF,
            SymmetryMode.RIGHT_HALF,
            SymmetryMode.UPPER_RIGHT_QUARTER,
        ):
            for field in ("bodies", "vias", "circuits", "bumps"):
                with self.subTest(mode=mode, field=field):
                    outside = _container("outside", **{field: [_box(-3, -3, -2, -2)]})
                    inside = _container("inside", **{field: [_box(1, 1, 2, 2)]})
                    root = _container("root", children=[outside, inside])

                    filter_container_to_domain(root, ModelDomain(mode, 0.0, 0.0))

                    self.assertEqual(root["children"], [inside])
                    self.assertEqual(len(inside[field]), 1)

    def test_prunes_nested_empty_branches_without_flattening_retained_children(self):
        first = _container("first", bodies=[_box(1, 1, 2, 2)])
        inside = _container("inside", bodies=[_box(1, 1, 2, 2)])
        parent = _container(
            "parent", bodies=[_box(-3, -3, -2, -2)], children=[inside]
        )
        last = _container("last", bodies=[_box(2, 2, 3, 3)])
        root = _container("root", children=[
            _container("empty-before"),
            first,
            _container("outside-branch", children=[
                _container("outside-leaf", bodies=[_box(-3, -3, -2, -2)])
            ]),
            parent,
            _container("empty-after"),
            last,
        ])

        filter_container_to_domain(
            root, ModelDomain(SymmetryMode.UPPER_RIGHT_QUARTER, 0.0, 0.0)
        )
        layers = plan_layers(extract_assignments(root), [], element_size={"globalElementSize": 1.0, "controls": []}["globalElementSize"]).layer_infos

        self.assertEqual(root["children"], [first, parent, last])
        self.assertIs(parent["children"][0], inside)
        self.assertEqual(parent["bodies"], [])
        self.assertEqual(inside["priority"], 3)
        start = next(
            assignment for layer in layers for assignment in layer["assignments"]
            if assignment["diagnostic"]["containerRef"] == "inside"
            and assignment["diagnostic"]["operation"] == "start"
        )
        self.assertEqual(start["areas"][0]["priority"], 3)

    def test_keeps_crossing_geometry_but_prunes_boundary_only_contacts(self):
        crossing = _container("crossing", bodies=[_box(-1, -1, 1, 1)])
        root = _container("root", children=[
            _container("touches-x", bodies=[_box(-1, 1, 0, 2)]),
            crossing,
            _container("touches-y", bodies=[_box(1, -1, 2, 0)]),
        ])

        filter_container_to_domain(
            root, ModelDomain(SymmetryMode.UPPER_RIGHT_QUARTER, 0.0, 0.0)
        )

        self.assertEqual(root["children"], [crossing])
        self.assertEqual(crossing["bodies"], [_box(-1, -1, 1, 1)])

    def test_keeps_root_when_all_geometry_is_filtered_out(self):
        root = _container("root", children=[
            _container("outside", bodies=[_box(-3, -3, -2, -2)])
        ])

        result = filter_container_to_domain(
            root, ModelDomain(SymmetryMode.UPPER_RIGHT_QUARTER, 0.0, 0.0)
        )

        self.assertIsNone(result)
        self.assertEqual(root, _container("root"))


if __name__ == "__main__":
    unittest.main()
