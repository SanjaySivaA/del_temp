# Checking the CUDA output in Google Colab

`colab_check.py` builds `src/main.cu`, `src/delaunay.cu`, and `src/io.cpp`,
generates PBBS samples and expected outputs, runs the program, and checks the
result. Correctness checking uses only the Python standard library. Optional
plotting calls the repository's `plot.py` and requires NumPy and Matplotlib in
the Colab runtime. No package installation is performed by the checker.

Select a GPU runtime in Colab, clone or upload this repository **including
`tests/colab_check.py`**, then run this Python cell. Adjust the checkout path if
needed:

```python
import subprocess

repo = "/content/del_temp"
subprocess.run(
    ["python3", f"{repo}/tests/colab_check.py", "--repo", repo],
    check=True,
)
```

The script uses `nvcc -arch=native` to target the visible GPU. This avoids the
Makefile's fixed `sm_80` target when Colab assigns different hardware. To
explicitly target an A30, append `"--arch", "sm_80"` to the arguments. Building
and running require a working CUDA toolkit and GPU in the Colab runtime.

The default is three runs of each of ten cases. An entirely passing run ends
with `30/30 runs passed`. This is the **expected success summary**, not a result
observed in the development environment. Exit status is 0 for success, 1 for
output/runtime/plotting failures, and 2 for setup/reference/build errors. A failure in
one case does not skip the remaining cases; each program run has a 60-second
timeout. Build failures are recorded in `build.log`.

## Viewing the triangulation in Colab

Use the updated `plot.py` to plot an existing checker output without rerunning
CUDA. Run this cell after the checker has produced the selected `.tri` file:

```python
from IPython.display import Image, display
import subprocess

repo = "/content/del_temp"
image_path = f"{repo}/colab-results/random_32_seed_42.png"
subprocess.run([
    "python3", f"{repo}/plot.py",
    "--pbbs", f"{repo}/colab-results/random_32_seed_42.actual.1.tri",
    "--output", image_path, "--no-show",
], check=True)
display(Image(filename=image_path, width=700))
```

Replace `random_32_seed_42` with another case name to view it. The explicit
`display` call shows the saved PNG inside the notebook; running a plotting
script as a subprocess alone does not embed its figure in the cell output.

To generate a PNG for every case while running the checks:

```python
%cd /content/del_temp
!python3 tests/colab_check.py --plot

from pathlib import Path
from IPython.display import Image, display

for image_path in sorted(Path("/content/del_temp/colab-results").glob("*.png")):
    print(image_path.stem)
    display(Image(filename=str(image_path), width=600))
```

`--plot` saves the first run of each case as `<case>.png`, with plotting logs
in `<case>.plot.log`. It also tries to draw incorrect outputs to help diagnose
failures. A saved image does not imply that the correctness check passed.
`--generate-only` writes fixtures and exits without plotting.

The original simple-file workflow still works: run the CUDA program with
`--simple` to create `points.txt` and `triangles.txt`, then run `plot.py` from
that same working directory. With no arguments it saves `delaunay_gpu.png`.
The plotter also handles a single triangle and outputs containing no triangles.

## Sample inputs and expected triangles

Point IDs are zero based, in the listed order. Every row below is embedded in
the script and written as actual PBBS files. Triangle order and vertex winding
do not affect acceptance.

| Case | Input points in order | Expected triangle vertex sets |
| --- | --- | --- |
| `triangle` | `(0,0), (4,0), (0,3)` | `012` |
| `quad_unique` | `(0,0), (4,0), (4,3), (0,2)` | `013, 123` |
| `interior_point` | `(0,0), (6,0), (0,6), (1,2)` | `013, 123, 023` |
| `square_tie` | `(0,0), (2,0), (2,2), (0,2)` | `012, 023` **or** `013, 123` |
| `square_center` | `(0,0), (4,0), (4,4), (0,4), (2,2)` | `014, 124, 234, 034` |
| `hull_collinear` | `(0,0), (2,0), (4,0), (0,3)` | `013, 123` |
| `quad_permuted` | `(4,3), (0,0), (0,2), (4,0)` | `123, 023` |
| `quad_transformed` | `(-100,200), (-72,200), (-72,221), (-100,214)` | `013, 123` |

Here `013` means the triangle with vertices `0 1 3`, not the integer 13.
The last two cases use 16 and 32 seeded integer points with no collinear
triples or cocircular quadruples. An independent exhaustive empty-circumcircle
reference computes their unique expected triangle sets using exact Python
integer arithmetic. It also cross-checks the manually specified unique cases.
The reference is intentionally small and slow, not a benchmark implementation.

For example, `quad_unique.2dpoints` is:

```text
pbbs_sequencePoint2d
0 0
4 0
4 3
0 2
```

Its complete expected `quad_unique.expected.tri` is:

```text
pbbs_triangles
4
2
0 0
4 0
4 3
0 2
0 1 3
1 2 3
```

The point input has **no count line**. The triangle output has point and
triangle counts, followed by exactly that many points and triangles. This
matches `src/io.cpp`; the repeated `<x0> <y0>` line in `geometry.md` is a
documentation typo. Whitespace, row order of triangles, and reversed triangle
winding are accepted. Point order must match the input because the program
maps sorted GPU indices back to original indices.

The square has four cocircular vertices, so either diagonal is valid. The
checker accepts either complete triangulation, not a mixture of their faces.
It rejects malformed output, changed/reordered points, invalid indices,
repeated vertices, zero-area faces, duplicate faces, and missing/extra faces.
Exact triangle-set comparison against the reference checks the complete
triangulation, rather than only its triangle count or local circle tests.

## Inspecting files and rerunning

The default artifact folder is `<repo>/colab-results/`:

- `<case>.2dpoints`: input.
- `<case>.expected.tri`: reference output. The square instead has
  `.expected.1.tri` and `.expected.2.tri` for its two valid choices.
- `<case>.actual.<run>.tri`: program output, if produced.
- `<case>.run.<run>.log`: program stdout/stderr, including its timings.
- `build.log` and `delaunay_check`: compiler log and executable.

Run these commands in Colab from the repository root as needed:

```python
# Generate inputs and reference outputs without building or running CUDA.
!python3 tests/colab_check.py --generate-only

# Check an already built binary, with ten repetitions per case.
!python3 tests/colab_check.py --binary ./delaunay --repeat 10

# Increase the per-invocation timeout and change the artifacts folder.
!python3 tests/colab_check.py --timeout 120 --output-dir /content/delaunay-checks
```

These are correctness samples, not a full PBBS benchmark or proof of
correctness. They include a point on a hull edge and cocircular points, which
may expose degeneracy handling problems in the current implementation. They
do not define duplicate-point behavior, malformed-input handling, or behavior
for entirely collinear inputs. The current host driver rejects fewer than
three points. GPU execution and the Python checker have not been run in the
development environment, in accordance with `AGENTS.md`.
