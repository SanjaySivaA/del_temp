"""CPU-only checker regressions to run in Colab, without CUDA or plotting."""

import tempfile
from pathlib import Path
import unittest

from colab_check import check_output, make_cases


class CheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case.name: case for case in make_cases()}

    def check_mesh(self, case, triangles):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actual.tri"
            path.write_text(
                f"pbbs_triangles\n{len(case.points)}\n{len(triangles)}\n"
                + "".join(f"{x} {y}\n" for x, y in case.points)
                + "".join(f"{a} {b} {c}\n" for a, b, c in triangles), encoding="utf-8")
            return check_output(case, path)

    def test_expected_faces_accept_reordering_and_reversed_winding(self):
        for case in self.cases.values():
            for expected in case.expected:
                with self.subTest(case=case.name, expected=expected):
                    reversed_faces = [tuple(reversed(t)) for t in reversed(expected)]
                    self.assertEqual(self.check_mesh(case, reversed_faces), len(expected))

    def test_containing_face_fails_even_with_correct_triangle_count(self):
        case = self.cases["interior_point"]
        with self.assertRaisesRegex(ValueError, "Point 3 .* strictly inside face"):
            self.check_mesh(case, ((0, 1, 2), (0, 1, 3), (1, 2, 3)))

    def test_crossing_edges_fail_even_with_correct_triangle_count(self):
        case = self.cases["quad_unique"]
        with self.assertRaisesRegex(ValueError, "Edges .* cross"):
            self.check_mesh(case, ((0, 1, 2), (1, 2, 3)))

    def test_planar_but_wrong_diagonal_fails_delaunay_check(self):
        case = self.cases["quad_unique"]
        with self.assertRaisesRegex(ValueError, "circumcircle.*not Delaunay"):
            self.check_mesh(case, ((0, 1, 2), (0, 2, 3)))


if __name__ == "__main__":
    unittest.main()
