"""Compare the old cap, relaxed cap, and opt-in dense LINE sections.

PYTHONPATH=src python examples/dense_adaptive_sections.py --no-show --plot /tmp/dense.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import generate_layer_adaptive_mesh
from mesher.mesh2d.editor import EditorMesh2D


def build_fixture():
    """A 300-wide dense run with spacing 5, surrounded by spacing about 1000."""
    xs = np.r_[0., 1000., 2000., 3000., np.arange(4000., 4301., 5.),
               5000., 6000., 7000., 8000.]
    n = len(xs)
    nodes = np.array([[x, y] for y in (0., 2000.) for x in xs])
    elements = np.array([[i, i + 1, i + n + 1, i + n] for i in range(n - 1)])
    mesh = Mesh2D(nodes, elements)
    return mesh, EditorMesh2D.outlines_by_edge(mesh)[0]


def build_comparison():
    cases = [('Previous: cap 10', dict(max_transition_count=10)),
             ('Relaxed cap: 1000', dict(max_transition_count=1000)),
             ('Dense sections: threshold 10, cap 1000',
              dict(small_element_size=10, max_transition_count=1000))]
    results = []
    for label, settings in cases:
        mesh, outline = build_fixture()
        result = generate_layer_adaptive_mesh(mesh, outline, 1000., **settings)
        results.append((label, result))
    return results


def plot_comparison(results):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    figure, axes = plt.subplots(3, 2, figsize=(16, 7.8), layout='constrained')
    for row, (label, result) in zip(axes, results):
        mesh, diagnostics = result.mesh, result.diagnostics
        dense_ids = {r['section_id'] for r in diagnostics['dense_sections']}
        sections = diagnostics['section_details']
        polygons = [np.asarray(d['polygon']) for d in sections]
        colors = ['#94d8ad' if d['section_id'] in dense_ids else '#d8e8f5' for d in sections]
        row[0].add_collection(PolyCollection(polygons, facecolors=colors,
                                            edgecolors='#24618a', linewidths=1.4))
        edges = np.concatenate([mesh.elements[:, [i, (i + 1) % 4]] for i in range(4)])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        row[1].add_collection(LineCollection(mesh.nodes[edges, :2], colors='#526477', linewidths=.65))
        for record in diagnostics['dense_sections']:
            detail = sections[record['section_id']]
            for ids in (detail['left_ids'], detail['right_ids']):
                row[1].plot(*mesh.nodes[ids, :2].T, color='#137c43', linewidth=1.8)
        for axis in row:
            axis.set_xlim(-120, 8120)
            axis.set_ylim(1850, 3200)
            axis.set_aspect('equal')
            axis.set_yticks([2000, 3000], ['Source', 'Outer'])
            axis.set_xticks([0, 2000, 4000, 6000, 8000])
            axis.spines[['top', 'right']].set_visible(False)
            axis.axvspan(4000, 4300, color='#36a368', alpha=.08, zorder=-1)
        row[0].set_title(f'{label} | sections', loc='left', fontsize=11)
        row[1].set_title(f'{label} | Quad4 mesh', loc='left', fontsize=11)
    for axis in axes[-1]:
        axis.set_xlabel('x (model units); dense interval: 4000–4300')
    figure.suptitle('Same boundary nodes: dense runs become independent LINE sections', fontsize=14)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    results = build_comparison()
    for label, result in results:
        print(f"{label}: {result.diagnostics['section_count']} sections, "
              f"{len(result.diagnostics['dense_sections'])} dense sections, "
              f"{result.diagnostics['added_elements']} added quads")
    if args.plot or not args.no_show:
        figure = plot_comparison(results)
        if args.plot:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.plot, dpi=180)
        import matplotlib.pyplot as plt
        if not args.no_show:
            plt.show()
        plt.close(figure)


if __name__ == '__main__':
    main()
