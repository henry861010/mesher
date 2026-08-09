import numpy as np

from mesh import Mesh


def _sort(float_list, tolerance=1e-3):
    # Convert to a numpy array and sort it
    arr = np.sort(np.asarray(float_list, dtype=np.float64))
    
    if arr.size == 0:
        return arr
    
    # Calculate the difference between consecutive elements
    # np.diff(arr) returns an array of size N-1
    diffs = np.diff(arr)
    
    # Create a boolean mask. We ALWAYS keep the first element (True),
    # and keep subsequent elements only if their difference from the previous is > tolerance
    mask = np.append([True], diffs > tolerance)
    
    return arr[mask]


def _densify(element_size, arr):
    out = []
    for a, b in zip(arr[:-1], arr[1:]):
        length = float(b - a)
        if length == 0.0:
            # duplicate line: keep only one point when joining segments
            if not out:
                out.append(a)
            continue
        nseg = max(1, int(np.ceil(length / element_size)))
        seg = np.linspace(a, b, nseg + 1, endpoint=True, dtype=np.float64)
        if out:
            seg = seg[1:]  # avoid boundary duplicate
        out.extend(seg.tolist())
    return np.asarray(out, dtype=np.float64)


def checkerboard_box(element_size, x_list, y_list) -> Mesh:
    x_list = _sort(x_list)
    y_list = _sort(y_list)

    x = _densify(element_size, x_list)
    y = _densify(element_size, y_list)
    
    if x.size < 2 or y.size < 2:
        raise ValueError("After _densify, need at least 2 x-lines and 2 y-lines.")

    Nx, Ny = int(x.size), int(y.size)

    ### nodes (x varies fastest)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.zeros_like(X)
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
        np.float64, copy=False
    )  # (Ny*Nx, 3)

    ### element node ids
    ix = np.arange(Nx - 1, dtype=np.int32)
    iy = np.arange(Ny - 1, dtype=np.int32)
    GX, GY = np.meshgrid(ix, iy, indexing="xy")

    n00 = (GY    ) * Nx + (GX    )  # BL
    n10 = (GY    ) * Nx + (GX + 1)  # BR
    n11 = (GY + 1) * Nx + (GX + 1)  # TR
    n01 = (GY + 1) * Nx + (GX    )  # TL

    # COUNTER-CLOCKWISE: BL, BR, TR, TL
    elements = np.stack([n00, n10, n11, n01], axis=-1).reshape(-1, 4).astype(np.int32)
    
    return Mesh(nodes=nodes, elements=elements)
