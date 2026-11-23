import argparse
from pathlib import Path

import matplotlib
import numpy as np


def read_pbbs_triangles(path):
    tokens = path.read_text(encoding="utf-8").split()
    if len(tokens) < 3 or tokens[0] != "pbbs_triangles":
        raise ValueError("Expected a pbbs_triangles file with point/triangle counts")
    n, m = int(tokens[1]), int(tokens[2])
    if n < 0 or m < 0 or len(tokens) != 3 + 2 * n + 3 * m:
        raise ValueError("PBBS point/triangle counts disagree with the file contents")
    points = np.array(tokens[3:3 + 2 * n], dtype=float).reshape(n, 2)
    triangles = np.array(tokens[3 + 2 * n:], dtype=int).reshape(m, 3)
    return points, triangles


def main():
    parser = argparse.ArgumentParser(description="Plot exported CUDA Delaunay triangles")
    parser.add_argument("--pbbs", type=Path,
                        help="Read a PBBS triangles output instead of the simple text files")
    parser.add_argument("--output", type=Path, default=Path("delaunay_gpu.png"))
    parser.add_argument("--no-show", action="store_true",
                        help="Save the image without opening a plotting window")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.pbbs:
        points, triangles = read_pbbs_triangles(args.pbbs)
    else:
        # ndmin preserves the row dimension for a single point or triangle.
        points = np.loadtxt("points.txt", ndmin=2)
        triangle_file = Path("triangles.txt")
        triangles = (np.loadtxt(triangle_file, dtype=int, ndmin=2)
                     if triangle_file.read_text(encoding="utf-8").strip()
                     else np.empty((0, 3), dtype=int))

    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Points must contain two finite coordinates per row")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Triangles must contain three vertex indices per row")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(points)):
        raise ValueError("Triangle vertex index is outside the points array")

    fig, ax = plt.subplots(figsize=(8, 8))
    # Do not let Matplotlib invent a triangulation when the program emitted none.
    if len(triangles):
        ax.triplot(points[:, 0], points[:, 1], triangles, color="crimson", lw=0.8)
    ax.scatter(points[:, 0], points[:, 1], color="blue", s=15)
    ax.set_title(f"GPU Delaunay Triangulation ({len(points)} Points, {len(triangles)} Triangles)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {args.output.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
