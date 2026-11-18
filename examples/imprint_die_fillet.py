"""Synthetic die/fillet assembly; render actual meshes and selected end caps."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from imprint_lines import pair, plot_case
from mesher.mesh2d.imprint_line import imprint_lines


def model():
    # Coordinates are illustrative micrometres; die is a material region, not a hole.
    offsets = np.r_[0, np.cumsum([5, 10, 15, 20] * 4)]
    pairs = [pair(0, (4000, 3800, 10000, 4000), 3800 + offsets, 800),
             pair(0, (4000, 10000, 10000, 10200), 10000 + offsets, 800),
             pair(1, (3800, 4000, 4000, 10000), 3800 + offsets, 800),
             pair(1, (10000, 4000, 10200, 10000), 10000 + offsets, 800)]
    focuses = [(1800, 10600, 2500, 11300), (2700, 10900, 3400, 11600),
               (10800, 7200, 11500, 7900), (11700, 7500, 12400, 8200),
               (6500, 1800, 7200, 2500)]
    for box in focuses:
        for axis in (0, 1):
            lo, hi = (box[1], box[3]) if axis == 0 else (box[0], box[2])
            pairs.append(pair(axis, box, np.linspace(lo, hi, 9), (400, 600)))
    die = [[4000,4000],[10000,4000],[10000,10000],[4000,10000]]
    patterns = [[die[i], die[(i+1)%4]] for i in range(4)]
    patterns += [[[7000, 800], [7000, 13200]], [[800, 7600], [13200, 7600]],
                 [[1200, 11000], [5200, 11000]], [[3000, 9400], [3000, 12500]],
                 [[11200, 6200], [11200, 9000]], [[5800, 2200], [8000, 2200]]]
    return pairs, patterns, focuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs, patterns, focuses = model()
    outer = [[0,0],[14000,0],[14000,14000],[0,14000]]
    start = perf_counter()
    mesh = imprint_lines(1000, pairs, patterns, outer_outline=outer,
                         min_transition_size=100, pattern_snap_distance=100)
    seconds = perf_counter() - start
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    case = ('die','Die with four fillet bands and five offset focuses',1000,pairs,patterns,outer,100)
    fig, ax = plt.subplots(figsize=(12,11))
    plot_case(ax,case,mesh)
    ax.text(7000,8500,'DIE\n6000 x 6000',ha='center',fontsize=16,color='#1765b5',bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    for i,b in enumerate(focuses,1):
        ax.text(b[0],b[3]+180,f'F{i}',fontsize=10,color='#1765b5')
    handles=[Patch(facecolor=c,label=l) for c,l in [('#cce3ff','Input control'),('#ffe4c5','Slanted quad'),('#b38bda','Triangle')]]
    handles.append(plt.Line2D([],[],color='#c53543',label='Pattern / die boundary'))
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False)
    ax.set_xlabel('X (illustrative micrometres)');ax.set_ylabel('Y (illustrative micrometres)')
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(args.output/'full.png',dpi=160);plt.close(fig)
    # Fixed viewing windows only; every displayed element comes from imprint_lines.
    # The public API does not expose transition bounds or source-band provenance.
    windows = [
        ('Horizontal fillet end', (-300, 3500, 1300, 4300)),
        ('Vertical fillet end', (9700, 12800, 10500, 14300)),
        ('Merged focus end', (-300, 10300, 1300, 11900)),
        ('Focus / pattern intersection', (8700, 6900, 10500, 8500)),
    ]
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for ax,(title,(a,b,c,d)) in zip(axes.flat,windows):
        plot_case(ax,case,mesh)
        ax.set_xlim(a,c);ax.set_ylim(b,d)
        ax.set_title(title,fontsize=11)
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(args.output/'details.png',dpi=180);plt.close(fig)
    np.savez_compressed(args.output/'mesh.npz',nodes=mesh.nodes,elements=mesh.elements)
    report=dict(seconds=seconds,nodes=mesh.node_count,elements=mesh.element_count,triangles=int(np.count_nonzero(mesh.element_types==3)),line_pairs=len(pairs),patterns=len(patterns))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__': main()
