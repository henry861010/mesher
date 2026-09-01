"""Interactive PyVista viewer for :class:`mesher.Mesh3D`."""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import random
from vtkmodules.vtkRenderingCore import vtkCellPicker, vtkPointPicker

from ..model import Mesh3D

_COLOR_RANDOM = random.Random(1)

class MeshViewer:
    def __init__(self, mesh: Mesh3D, *, component_names=None):
        self.mesh = mesh
        self.component_names = component_names or {}

    def _build_grid(self):
        cells = []
        celltypes = np.empty(self.mesh.element_count, dtype=np.uint8)
        for index, element in enumerate(self.mesh.elements):
            is_wedge = element[2] == element[3] and element[6] == element[7]
            if is_wedge:
                wedge = element[[0, 1, 2, 4, 5, 6]]
                cells.append(np.concatenate(([6], wedge)))
                celltypes[index] = pv.CellType.WEDGE
            else:
                cells.append(np.concatenate(([8], element)))
                celltypes[index] = pv.CellType.HEXAHEDRON
        packed_cells = (
            np.concatenate(cells).astype(np.int32, copy=False)
            if cells
            else np.empty(0, dtype=np.int32)
        )

        grid = pv.UnstructuredGrid(packed_cells, celltypes, self.mesh.nodes)

        ### Attach component ids as cell data for coloring
        grid.cell_data['comp'] = self.mesh.element_comps.astype(np.int32)
        return grid

    def _to_int(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _update_component_name_map(self, comp_names, mapping, overwrite=False):
        if mapping is None:
            return

        if isinstance(mapping, (list, tuple)):
            mapping = dict(enumerate(mapping))

        for key, value in mapping.items():
            key_id = self._to_int(key)
            value_id = self._to_int(value)

            if key_id is not None and value_id is None:
                comp_id = key_id
                name = str(value)
            elif value_id is not None:
                comp_id = value_id
                name = str(key)
            elif (
                key in self.mesh.comps
                and self._to_int(self.mesh.comps[key]) is not None
            ):
                comp_id = int(self.mesh.comps[key])
                name = str(value)
            else:
                continue

            if overwrite or comp_id not in comp_names:
                comp_names[comp_id] = name

    def _component_name_map(self, component_names=None):
        comp_names = {}
        self._update_component_name_map(comp_names, self.mesh.comps)
        self._update_component_name_map(comp_names, self.component_names, overwrite=True)
        self._update_component_name_map(comp_names, component_names, overwrite=True)
        return comp_names

    def _component_rows(self, comp, randomize_colors, component_names=None):
        vals, counts = np.unique(comp, return_counts=True)
        comp_names = self._component_name_map(component_names)

        base = plt.get_cmap('viridis', max(len(vals), 1))
        palette = [to_hex(base(i)) for i in np.linspace(0, 1, max(len(vals), 1))]
        if randomize_colors:
            _COLOR_RANDOM.shuffle(palette)

        rows = []
        for index, (comp_id, count) in enumerate(zip(vals, counts)):
            comp_id = int(comp_id)
            name = comp_names.get(comp_id, f"Component {index + 1}")
            rows.append({
                "id": comp_id,
                "name": name,
                "count": int(count),
                "color": palette[index],
            })
        return rows

    def _add_component_panel(
        self,
        plotter,
        rows,
        actors,
        element_count,
        node_count,
        on_visibility_change=None,
    ):
        x0 = 12
        top_y = plotter.window_size[1] - 28
        button_size = 22
        row_gap = 30
        text_gap = 18
        text_x = x0 + button_size + 10

        plotter.add_text(
            f"Elements: {element_count}",
            position=(x0, top_y),
            font_size=9,
            color="black",
            name="mesh_element_count",
            render=False,
        )
        plotter.add_text(
            f"Nodes: {node_count}",
            position=(x0, top_y - text_gap),
            font_size=9,
            color="black",
            name="mesh_node_count",
            render=False,
        )

        label_actors = {}
        if not rows:
            return label_actors

        plotter.add_text(
            "Components",
            position=(x0, top_y - text_gap * 3),
            font_size=11,
            color="black",
            name="component_panel_title",
            render=False,
        )

        for index, row in enumerate(rows):
            y = top_y - text_gap * 4 - button_size - row_gap * index
            label = f"{row['name']}: {row['count']} elems"
            actor = actors[row["id"]]

            def toggle_component(is_visible, actor=actor, comp_id=row["id"]):
                is_visible = bool(is_visible)
                actor.SetVisibility(is_visible)
                actor.SetPickable(is_visible)
                if on_visibility_change is not None:
                    on_visibility_change(comp_id, is_visible, render=False)
                plotter.render()

            plotter.add_checkbox_button_widget(
                toggle_component,
                value=True,
                position=(x0, y),
                size=button_size,
                border_size=2,
                color_on=row["color"],
                color_off="lightgrey",
                background_color="white",
            )
            label_actors[row["id"]] = plotter.add_text(
                label,
                position=(text_x, y + 3),
                font_size=9,
                color=row["color"],
                name=f"component_panel_label_{row['id']}",
                render=False,
            )

        return label_actors

    def _visible_node_ids(self, actors, component_node_indices):
        visible_node_ids = [
            component_node_indices[comp_id]
            for comp_id, actor in actors.items()
            if actor.GetVisibility()
        ]
        if not visible_node_ids:
            return np.empty((0,), dtype=np.int32)
        return np.unique(np.concatenate(visible_node_ids)).astype(np.int32)

    def _nearest_visible_node(self, point, actors, component_node_indices):
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return None

        node_ids = self._visible_node_ids(actors, component_node_indices)
        if node_ids.size == 0:
            return None

        node_points = self.mesh.nodes[node_ids]
        nearest_index = int(np.argmin(np.sum((node_points - point) ** 2, axis=1)))
        node_id = int(node_ids[nearest_index])
        return {
            "id": node_id,
            "point": self.mesh.nodes[node_id].astype(float),
        }

    def _remove_actor(self, plotter, name, render=False):
        try:
            plotter.remove_actor(name, render=render)
        except TypeError:
            plotter.remove_actor(name)

    def _add_selection_tool(
        self,
        plotter,
        actors,
        component_label_actors,
        component_node_indices,
    ):
        point_actor_names = (
            "selected_node_marker",
            "selected_node_label",
        )
        selected_component_id = None

        cell_picker = vtkCellPicker()
        point_picker = vtkPointPicker()
        point_picker.SetTolerance(0.03)

        def component_id_for_actor(picked_actor):
            if picked_actor is None:
                return None
            for comp_id, actor in actors.items():
                if picked_actor is actor or picked_actor == actor:
                    return comp_id
            return None

        def clear_point_selection():
            for name in point_actor_names:
                self._remove_actor(plotter, name, render=False)

        def select_component(comp_id):
            nonlocal selected_component_id
            selected_component_id = comp_id
            for current_comp_id, label_actor in component_label_actors.items():
                label_actor.prop.bold = current_comp_id == comp_id

        def clear_selection(render=True):
            clear_point_selection()
            select_component(None)
            if render:
                plotter.render()

        def add_point_annotation(node):
            point = node["point"]
            label = (
                f"X: {point[0]:.6g}\n"
                f"Y: {point[1]:.6g}\n"
                f"Z: {point[2]:.6g}"
            )

            plotter.add_mesh(
                pv.PolyData(np.asarray([point])),
                color="red",
                point_size=14,
                render_points_as_spheres=True,
                pickable=False,
                name="selected_node_marker",
                render=False,
            )
            plotter.add_point_labels(
                np.asarray([point]),
                [label],
                font_size=12,
                text_color="black",
                shape_color="white",
                shape_opacity=0.75,
                show_points=False,
                always_visible=True,
                pickable=False,
                name="selected_node_label",
                render=False,
            )

        def on_click(position):
            if position is None or len(position) != 2:
                clear_selection()
                return

            x, y = position
            cell_picker.Pick(x, y, 0, plotter.renderer)
            comp_id = component_id_for_actor(cell_picker.GetActor())
            if comp_id is None:
                clear_selection()
                return

            select_component(comp_id)
            clear_point_selection()

            point_picker.Pick(x, y, 0, plotter.renderer)
            point_comp_id = component_id_for_actor(point_picker.GetActor())
            if point_comp_id is not None:
                node = self._nearest_visible_node(
                    point_picker.GetPickPosition(),
                    actors,
                    component_node_indices,
                )
                if node is not None:
                    add_point_annotation(node)

            plotter.render()

        def on_visibility_change(comp_id, is_visible, render=True):
            if not is_visible and selected_component_id == comp_id:
                clear_selection(render=render)

        plotter.track_click_position(
            on_click,
            side="left",
            double=False,
            viewport=True,
        )
        plotter.track_click_position(
            on_click,
            side="left",
            double=True,
            viewport=True,
        )
        return on_visibility_change

    def show(self, randomize_colors=False, component_names=None):
        grid = self._build_grid()

        ### colors
        if 'comp' in grid.point_data and 'comp' not in grid.cell_data:
            grid = grid.point_data_to_cell_data(pass_point_data=False)
        comp = grid.cell_data['comp'].astype(int)
        rows = self._component_rows(comp, randomize_colors, component_names)
        
        ### Plot
        plotter = pv.Plotter()
        actors = {}
        component_node_indices = {}
        for row in rows:
            cell_indices = np.where(comp == row["id"])[0]
            component_grid = grid.extract_cells(cell_indices)
            component_node_indices[row["id"]] = np.unique(
                self.mesh.elements[cell_indices].ravel()
            ).astype(np.int32)
            actors[row["id"]] = plotter.add_mesh(
                component_grid,
                color=row["color"],
                show_edges=True,
                smooth_shading=False,
                show_scalar_bar=False,
                pickable=True,
                name=f"component_{row['id']}",
            )
        
        on_visibility_change = None

        def handle_visibility_change(comp_id, is_visible, render=True):
            if on_visibility_change is not None:
                on_visibility_change(comp_id, is_visible, render=render)

        component_label_actors = self._add_component_panel(
            plotter,
            rows,
            actors,
            self.mesh.element_count,
            self.mesh.node_count,
            on_visibility_change=handle_visibility_change,
        )
        on_visibility_change = self._add_selection_tool(
            plotter,
            actors,
            component_label_actors,
            component_node_indices,
        )
        plotter.add_axes()
        plotter.show()
