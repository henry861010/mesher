"""Inspect boxes on their boundary grid using Matplotlib.

Run from the repository root:
    PYTHONPATH=src venv/bin/python examples/pattern2d.py

Edit BOXES below, then use the window toolbar to pan, zoom, or save a picture.
"""

from mesher.mesh2d.visualization import plot_pattern2d


BOXES = [
    [0, 0, 3, 2],
    [2, 1, 5, 4],
    [6, 0, 7, 1],
]


def main():
    import matplotlib.pyplot as plt

    plot_pattern2d(BOXES)
    plt.show()


if __name__ == "__main__":
    main()
