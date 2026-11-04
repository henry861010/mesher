"""Hole-aware grid mesh 使用範例。

在專案根目錄執行：
    PYTHONPATH=src venv/bin/python examples/test3.py

藍線為輸入 pattern，黃色點為 holes 的完整邊界節點。
顯示網格需要 visualization extra（PyVista）。
"""

import numpy as np

from mesher.mesh2d.generators import generate_rectilinear_mesh_optimal
from mesher.mesh2d.visualization import view_mesh


def main():
    # 修改這些輸入即可試不同案例；輪廓不必重複第一個點。
    global_size = 20.0
    outer_outline = np.array([
        [0, 0], [200, 0], [200, 80],
        [150, 80], [150, 100], [0, 100],
    ], dtype=np.float64)

    holes = [
        np.array([[50, 30], [80, 30], [80, 70], [50, 70]], dtype=np.float64),
        np.array([[120, 20], [140, 20], [140, 60], [120, 60]], dtype=np.float64),
    ]

    # 左側的短線延伸到第一個 hole 就停止，不會傳播到 hole 右側。
    pattern_lines = np.array([
        [[10, 42], [30, 42]],
        [[10, 43], [30, 43]],
        [[10, 44], [30, 44]],
    ], dtype=np.float64)

    result = generate_rectilinear_mesh_optimal(
        target_edge_size=global_size,
        outer_outline=outer_outline,
        holes=holes,
        pattern_lines=pattern_lines,
    )

    mesh = result.mesh
    print(f"Mesh: {mesh.node_count} nodes, {mesh.element_count} Quad4 elements")
    for index, node_ids in enumerate(result.hole_outlines, start=1):
        # 取得 hole 的實際邊界座標可使用 mesh.nodes[node_ids, :2]。
        print(f"Hole {index}: {len(node_ids)} boundary nodes (even={len(node_ids) % 2 == 0})")
    print(f"Parity repair cuts: {len(result.diagnostics['parity_repairs'])}")

    view_mesh(
        mesh,
        node_indices=np.concatenate(result.hole_outlines) if result.hole_outlines else None,
        reference_lines=pattern_lines,
    )


if __name__ == "__main__":
    main()
