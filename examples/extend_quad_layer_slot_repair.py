"""Show staged C-slot filling and expansion with a misaligned mouth.

Run from the repository root::

    venv/bin/python examples/extend_quad_layer_slot_repair.py
    venv/bin/python examples/extend_quad_layer_slot_repair.py --case misaligned --no-show
    venv/bin/python examples/extend_quad_layer_slot_repair.py --no-show --plot output.png --diagnostics output.json

The example creates the old mesh, calls the public adapter once per case, and
plots that committed result. It does not implement repair or remesh the result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mesher import Mesh2D
from mesher.mesh2d.adapter import extend_quad_layer
from mesher.mesh2d.adapter.visualization import (
    diagnostic_report, plot_quad_layer, save_diagnostics,
)
from mesher.mesh2d.editor import EditorMesh2D


def make_example(case, element_size=.5):
    """A unit-wide C slot; the lower lip need not align with an upper node."""
    bottom_x = [1., 1.5, 2., 2.5, 3.] if case != 'misaligned' else [1., 1.5, 2.25]
    blocks = [
        ([0., .5, 1.], np.arange(0., 3.01, .5)),
        (bottom_x, [0., .5, 1.]),
        ([1., 1.5, 2., 2.5, 3.], [2., 2.5, 3.]),
    ]
    points, cells, ids = [], [], {}
    for xs, ys in blocks:
        for left, right in zip(xs[:-1], xs[1:]):
            for bottom, top in zip(ys[:-1], ys[1:]):
                cell = []
                for point in ((left, bottom), (right, bottom), (right, top), (left, top)):
                    if point not in ids:
                        ids[point] = len(points)
                        points.append(point)
                    cell.append(ids[point])
                cells.append(cell)
    mesh = Mesh2D(points, np.asarray(cells, dtype=np.int32))
    patterns = [[[1., 1.5], [3. + element_size, 1.5]]] if case == 'pattern' else ()
    return mesh, EditorMesh2D.outlines_by_edge(mesh)[0], patterns


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', choices=('all', 'aligned', 'misaligned', 'pattern'), default='all')
    parser.add_argument('--element-size', type=float, default=.5)
    parser.add_argument('--plot', type=Path,
                        default=Path(__file__).resolve().parent / 'output' / 'extend_quad_layer_slot_repair.png')
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cases = ('aligned', 'misaligned', 'pattern') if args.case == 'all' else (args.case,)
    figure, axes = plt.subplots(len(cases), 3, squeeze=False, figsize=(16, 5.1 * len(cases)),
                                layout='constrained')
    figure.suptitle('Slot repair → adaptive Quad4 expansion\n'
                   f'element_size = {args.element_size:g}; original nodes and cells stay fixed',
                   fontsize=16, weight='bold')
    reports = []
    for case, row in zip(cases, axes):
        mesh, outline, patterns = make_example(case, args.element_size)
        result = extend_quad_layer(mesh, outline, args.element_size, pattern_lines=patterns)
        name = {'aligned': 'Aligned C slot', 'misaligned': 'Mouth offset 0.25 / unequal wall counts',
                'pattern': 'Pattern through repair and expansion'}[case]
        plot_quad_layer(result, outline, pattern_lines=patterns, name=name, axes=row)
        reports.append(diagnostic_report(result, name=name, element_size=args.element_size,
                                         expand_ratio=1.))
        diagnostics = result.diagnostics
        print(f"{case}: {diagnostics['repair_added_elements']} repair + "
              f"{diagnostics['expand_added_elements']} expansion = "
              f"{diagnostics['added_elements']} new Quad4 cells; "
              f"{diagnostics['elapsed_seconds']:.3f} s")
        for repair in diagnostics['slot_repairs']:
            print(f"  mouth nodes: {repair['mouth_node_ids']}; "
                  f"repair elements [{repair['element_start']}, {repair['element_end']})")
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot, dpi=170)
    print(f'Saved {args.plot.resolve()}')
    if args.diagnostics is not None:
        save_diagnostics(reports, args.diagnostics)
        print(f'Saved {args.diagnostics.resolve()}')
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == '__main__':
    main()
