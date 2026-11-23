#!/usr/bin/env python3
"""Build and check this CUDA implementation on a GPU runtime, e.g. Colab.

Uses only Python's standard library. Inputs and expected PBBS outputs are
written before building/running CUDA; --generate-only needs no CUDA at all.
"""

import argparse
from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys


@dataclass(frozen=True)
class Case:
    name: str
    points: tuple
    # Each entry is a complete acceptable triangulation, not individual choices.
    expected: tuple


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def incircle(a, b, c, d):
    """Positive inside the circle of CCW abc; exact for Python integer inputs."""
    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]
    return ((ax * ax + ay * ay) * (bx * cy - by * cx)
            - (bx * bx + by * by) * (ax * cy - ay * cx)
            + (cx * cx + cy * cy) * (ax * by - ay * bx))


def reference_triangles(points):
    """Enumerate empty circumcircles: a small, independent O(n^4) oracle.

    An empty circle through four points makes the triangulation nonunique.
    Such cases need explicit alternatives, as supplied for the square below.
    This is deliberately for small correctness fixtures, not PBBS benchmarks.
    """
    triangles = []
    for triangle in combinations(range(len(points)), 3):
        a, b, c = (points[i] for i in triangle)
        winding = orient(a, b, c)
        if winding == 0:
            continue
        signs = [incircle(a, b, c, p) * (1 if winding > 0 else -1)
                 for i, p in enumerate(points) if i not in triangle]
        if any(sign > 0 for sign in signs):
            continue
        if any(sign == 0 for sign in signs):
            raise ValueError("Nonunique reference: supply explicit triangulations")
        triangles.append(triangle)
    return tuple(triangles)


def general_position_points(count, seed):
    """Distinct integer points, no collinear triples or cocircular quadruples."""
    rng = random.Random(seed)
    points = []
    for _ in range(100000):
        candidate = (rng.randrange(-100, 101), rng.randrange(-100, 101))
        if candidate in points:
            continue
        if any(orient(a, b, candidate) == 0 for a, b in combinations(points, 2)):
            continue
        if any(incircle(a, b, c, candidate) == 0
               for a, b, c in combinations(points, 3)):
            continue
        points.append(candidate)
        if len(points) == count:
            return tuple(points)
    raise ValueError("Could not generate the requested general-position sample")


def canonical(triangles):
    # Triangle order, rotation and winding are not part of PBBS correctness.
    return frozenset(tuple(sorted(t)) for t in triangles)


def make_cases():
    quad = ((0, 0), (4, 0), (4, 3), (0, 2))
    quad_triangles = ((0, 1, 3), (1, 2, 3))
    cases = [
        Case("triangle", ((0, 0), (4, 0), (0, 3)), (((0, 1, 2),),)),
        Case("quad_unique", quad, (quad_triangles,)),
        Case("interior_point", ((0, 0), (6, 0), (0, 6), (1, 2)),
             (((0, 1, 3), (1, 2, 3), (0, 2, 3)),)),
        Case("square_tie", ((0, 0), (2, 0), (2, 2), (0, 2)),
             (((0, 1, 2), (0, 2, 3)), ((0, 1, 3), (1, 2, 3)))),
        Case("square_center", ((0, 0), (4, 0), (4, 4), (0, 4), (2, 2)),
             (((0, 1, 4), (1, 2, 4), (2, 3, 4), (0, 3, 4)),)),
        Case("hull_collinear", ((0, 0), (2, 0), (4, 0), (0, 3)),
             (((0, 1, 3), (1, 2, 3)),)),
        Case("quad_permuted", tuple(quad[i] for i in (2, 0, 3, 1)),
             (((1, 2, 3), (0, 2, 3)),)),
        Case("quad_transformed", tuple((7 * x - 100, 7 * y + 200) for x, y in quad),
             (quad_triangles,)),
    ]
    # Cross-check the hand-written unique fixtures against the independent oracle.
    for case in cases:
        if len(case.expected) == 1:
            if canonical(reference_triangles(case.points)) != canonical(case.expected[0]):
                raise ValueError(f"Incorrect expected fixture: {case.name}")
    for count, seed in ((16, 7), (32, 42)):
        points = general_position_points(count, seed)
        cases.append(Case(f"random_{count}_seed_{seed}", points,
                          (reference_triangles(points),)))
    return cases


def write_fixtures(case, directory):
    coordinates = "".join(f"{x} {y}\n" for x, y in case.points)
    input_path = directory / f"{case.name}.2dpoints"
    input_path.write_text("pbbs_sequencePoint2d\n" + coordinates, encoding="utf-8")
    for index, triangles in enumerate(case.expected, 1):
        suffix = "" if len(case.expected) == 1 else f".{index}"
        output_path = directory / f"{case.name}.expected{suffix}.tri"
        output_path.write_text(
            f"pbbs_triangles\n{len(case.points)}\n{len(triangles)}\n" + coordinates
            + "".join(f"{a} {b} {c}\n" for a, b, c in triangles), encoding="utf-8")
    return input_path


def read_triangles(path):
    tokens = path.read_text(encoding="utf-8").split()
    if len(tokens) < 3 or tokens[0] != "pbbs_triangles":
        raise ValueError("Missing pbbs_triangles header or point/triangle counts")
    n, m = int(tokens[1]), int(tokens[2])
    if n < 0 or m < 0:
        raise ValueError("Negative point/triangle count")
    if len(tokens) != 3 + 2 * n + 3 * m:
        raise ValueError(f"Token count disagrees with header: n={n}, m={m}")
    points = [tuple(map(float, tokens[i:i + 2])) for i in range(3, 3 + 2 * n, 2)]
    if not all(math.isfinite(value) for point in points for value in point):
        raise ValueError("Non-finite coordinate in output")
    triangles = [tuple(map(int, tokens[i:i + 3]))
                 for i in range(3 + 2 * n, len(tokens), 3)]
    return points, triangles


def check_output(case, path):
    points, triangles = read_triangles(path)
    # These fixtures contain exactly representable small integers. No coordinate
    # tolerance is needed, even for the program's 16-significant-digit writer.
    if tuple(points) != case.points:
        raise ValueError("Output points differ from the input coordinates/order")
    for triangle in triangles:
        if any(i < 0 or i >= len(points) for i in triangle):
            raise ValueError(f"Out-of-range vertex index: {triangle}")
        if len(set(triangle)) != 3:
            raise ValueError(f"Repeated vertex in triangle: {triangle}")
        if orient(*(case.points[i] for i in triangle)) == 0:
            raise ValueError(f"Zero-area triangle: {triangle}")
    actual = canonical(triangles)
    if len(actual) != len(triangles):
        raise ValueError("Duplicate triangle(s) in output")
    alternatives = [canonical(expected) for expected in case.expected]
    if actual not in alternatives:
        closest = min(alternatives, key=lambda expected: len(actual ^ expected))
        missing, extra = sorted(closest - actual), sorted(actual - closest)
        raise ValueError(
            f"Expected {len(closest)} triangles, got {len(triangles)}; "
            f"missing (up to 8): {missing[:8]}; unexpected (up to 8): {extra[:8]}")
    return len(triangles)


def run_logged(command, cwd, log_path, timeout):
    # Direct argv invocation also supports repository paths containing spaces.
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(command, cwd=cwd, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Timed out after {timeout:g}s; see {log_path}") from error
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"Exit code {result.returncode}; see {log_path}\n{tail}")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root (default: parent of tests/)")
    parser.add_argument("--output-dir", type=Path,
                        help="Artifacts directory (default: <repo>/colab-results)")
    parser.add_argument("--binary", type=Path, help="Check an existing executable; skip build")
    parser.add_argument("--arch", default="native",
                        help="nvcc architecture (default: native, the visible Colab GPU)")
    parser.add_argument("--repeat", type=positive_int, default=3,
                        help="Runs per case to help expose nondeterminism (default: 3)")
    parser.add_argument("--timeout", type=positive_int, default=60,
                        help="Timeout in seconds per CUDA invocation (default: 60)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Write inputs/expected outputs without compiling or running CUDA")
    args = parser.parse_args()
    repo = args.repo.resolve()
    directory = (args.output_dir or repo / "colab-results").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cases = make_cases()
    inputs = {case.name: write_fixtures(case, directory) for case in cases}
    print(f"Generated {len(cases)} cases in {directory}", flush=True)
    for case in cases:
        print(f"  {case.name}: {len(case.points)} points, "
              f"{len(case.expected[0])} expected triangles, "
              f"{len(case.expected)} accepted triangulation(s)")
    if args.generate_only:
        return 0

    if args.binary:
        binary = args.binary.resolve()
        if not binary.is_file():
            raise ValueError(f"Executable not found: {binary}")
    else:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise RuntimeError("nvcc not found; select a Colab GPU runtime with CUDA available")
        sources = [repo / name for name in ("src/main.cu", "src/delaunay.cu", "src/io.cpp")]
        for source in sources:
            if not source.is_file():
                raise ValueError(f"Source not found: {source}; set --repo to the checkout")
        binary = directory / "delaunay_check"
        command = [nvcc, "-O3", "-std=c++17", f"-arch={args.arch}",
                   "-I", str(repo / "include"), *map(str, sources), "-o", str(binary)]
        print(f"Building for {args.arch} (log: {directory / 'build.log'})", flush=True)
        run_logged(command, repo, directory / "build.log", timeout=180)

    passed = 0
    for case in cases:
        successes = 0
        for repetition in range(1, args.repeat + 1):
            actual = directory / f"{case.name}.actual.{repetition}.tri"
            log = directory / f"{case.name}.run.{repetition}.log"
            # Never let a previous invocation's output mask a missing write.
            if actual.exists():
                actual.unlink()
            try:
                run_logged([str(binary), str(inputs[case.name]), str(actual)],
                           directory, log, args.timeout)
                check_output(case, actual)
                successes += 1
            except (OSError, ValueError, RuntimeError) as error:
                print(f"FAIL {case.name} run {repetition}: {error}", flush=True)
                print(f"  Output: {actual}\n  Program log: {log}", flush=True)
        passed += successes
        if successes == args.repeat:
            print(f"PASS {case.name}: {len(case.expected[0])} triangles "
                  f"({successes}/{args.repeat} runs)", flush=True)
    total = len(cases) * args.repeat
    print(f"\n{passed}/{total} runs passed. Artifacts: {directory}", flush=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
