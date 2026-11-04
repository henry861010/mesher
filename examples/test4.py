"""generate_full_adaptive_mesh 使用範例。

在專案根目錄執行：
    PYTHONPATH=src venv/bin/python examples/test4.py
    PYTHONPATH=src venv/bin/python examples/test4.py --no-show

黃色點標示各層轉接的邊界節點；顯示網格需要 visualization extra。
"""

import argparse

import numpy as np

from mesher. .adapter import generate_full_adaptive_mesh
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="只產生網格，不開啟視窗。")
    args = parser.parse_args()

    # 先建立兩個分離的 focus mesh，單元尺寸都是 5。
    focus_a = generate_rectilinear_mesh(5, [0, 40], [0, 40])
    focus_b = generate_rectilinear_mesh(5, [120, 160], [0, 40])
    mesh = EditorMesh2D.merge(focus_a, focus_b)

    # 每個 focus 區域依序轉接到 10、20；global mesh 也使用最後的尺寸 20。
    # 轉接後的區域必須彼此分離，且嚴格位於 outer_outline 內。
    element_sizes = [10.0, 20.0]
    outer_outline = np.array([
        [-80, -80], [240, -80], [240, 120], [-80, 120],
    ], dtype=np.float64)
    pattern_lines = np.empty((0, 2, 2), dtype=np.float64)

    print(f"Focus mesh: {mesh.node_count} nodes, {mesh.element_count} Quad4 elements")
    result = generate_full_adaptive_mesh(
        mesh=mesh,
        element_sizes=element_sizes,
        outer_outline=outer_outline,
        expand_ratio=1.5,
        pattern_lines=pattern_lines,
    )
    # 成功後 result.mesh 就是已更新的原 mesh；失敗時原 mesh 不會改變。
    print(f"Full mesh: {result.mesh.node_count} nodes, {result.mesh.element_count} Quad4 elements")
    print(f"Global outer boundary: {len(result.outer_outline)} nodes")
    for index, outlines in enumerate(result.layer_outlines):
        print(f"Layer {index + 1}: size={element_sizes[index]:g}, {len(outlines)} focus boundaries")

    if not args.no_show:
        from mesher.mesh2d.visualization import view_mesh

        # layer_outlines 的索引順序：層 → 原始邊界 → 邏輯邊 → node IDs。
        boundary_nodes = np.unique(np.concatenate([
            np.asarray(edge[:-1], dtype=np.int64)
            for layer in result.layer_outlines for outline in layer for edge in outline
        ]))
        view_mesh(result.mesh, node_indices=boundary_nodes, reference_lines=pattern_lines)


if __name__ == "__main__":
    main()
