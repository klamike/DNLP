"""
Copyright 2026 Michael Klamkin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import warnings

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.fenchel_dual import FenchelDual, fenchel_dual
from cvxpy.reductions.solution import Solution
from cvxpy.tests.base_test import BaseTest


class TestAtomConjugates(BaseTest):
    """Test conjugate() methods on individual atoms."""

    def test_exp_conjugate_value(self) -> None:
        """exp*(y) = y*log(y) - y for y > 0."""
        y = cp.Variable(3)
        x_atom = cp.exp(cp.Variable(3))
        conj_expr, conj_constraints = x_atom.conjugate(y)

        # Verify with a specific y value: exp*(2) = 2*ln(2) - 2
        y.value = np.array([2.0, 1.0, np.e])
        expected = np.array([
            2.0 * np.log(2.0) - 2.0,
            1.0 * np.log(1.0) - 1.0,
            np.e * np.log(np.e) - np.e,
        ])
        # Sum of conjugate values.
        actual = conj_expr.value
        self.assertItemsAlmostEqual(actual, expected, places=5)

    def test_norm1_conjugate(self) -> None:
        """||x||_1* is the indicator of ||y||_inf <= 1."""
        y = cp.Variable(3)
        x_atom = cp.norm1(cp.Variable(3))
        conj_expr, conj_constraints = x_atom.conjugate(y)

        # The conjugate expression should be 0.
        self.assertEqual(conj_expr, 0)
        # There should be one constraint: norm_inf(y) <= 1.
        self.assertEqual(len(conj_constraints), 1)

    def test_norm1_conjugate_axis(self) -> None:
        """Axis-aware ||x||_1* is per-slice ||y||_inf <= 1."""
        y = cp.Variable((2, 3))
        x_atom = cp.norm1(cp.Variable((2, 3)), axis=0)
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

    def test_norm_inf_conjugate(self) -> None:
        """||x||_inf* is the indicator of ||y||_1 <= 1."""
        y = cp.Variable(3)
        x_atom = cp.norm_inf(cp.Variable(3))
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

    def test_norm_inf_conjugate_axis(self) -> None:
        """Axis-aware ||x||_inf* is per-slice ||y||_1 <= 1."""
        y = cp.Variable((2, 3))
        x_atom = cp.norm_inf(cp.Variable((2, 3)), axis=0)
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

    def test_abs_conjugate(self) -> None:
        """|x|* is indicator of -1 <= y <= 1."""
        y = cp.Variable(3)
        x_atom = cp.abs(cp.Variable(3))
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 2)

    def test_abs_conjugate_complex(self) -> None:
        """For complex y, |x|* domain is |y| <= 1."""
        y = cp.Variable(3, complex=True)
        x_atom = cp.abs(cp.Variable(3, complex=True))
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

    def test_max_conjugate(self) -> None:
        """max(x)* is indicator of the simplex."""
        y = cp.Variable(4)
        x_atom = cp.max(cp.Variable(4))
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        # y >= 0, sum(y) == 1
        self.assertEqual(len(conj_constraints), 2)

    def test_max_conjugate_axis(self) -> None:
        """Axis-aware max conjugate is product of simplices."""
        y = cp.Variable((2, 4))
        x_atom = cp.max(cp.Variable((2, 4)), axis=0)
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 2)

    def test_sum_largest_conjugate(self) -> None:
        """sum_largest(x, k)* is indicator of capped simplex."""
        y = cp.Variable(5)
        x_atom = cp.sum_largest(cp.Variable(5), 2)
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        # y >= 0, y <= 1, sum(y) == 2
        self.assertEqual(len(conj_constraints), 3)

    def test_pnorm_conjugate(self) -> None:
        """||x||_p* is indicator of ||y||_q <= 1 with 1/p + 1/q = 1."""
        y = cp.Variable(3)
        x_atom = cp.pnorm(cp.Variable(3), 2, approx=False)
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

    def test_pnorm_conjugate_axis(self) -> None:
        """Axis-aware pnorm conjugate uses per-slice dual norm bound."""
        y = cp.Variable((2, 3))
        x_atom = cp.pnorm(cp.Variable((2, 3)), 2, axis=0, approx=False)
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

    def test_pnorm_conjugate_axis_non_euclidean(self) -> None:
        """Axis-aware pnorm conjugate supports p != 2."""
        y = cp.Variable((2, 3))
        x_atom = cp.pnorm(cp.Variable((2, 3)), 1.5, axis=0, approx=False)
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

    def test_power_conjugate_general_p_one_sided(self) -> None:
        """For p>1 non-power-of-two, conjugate uses positive part."""
        p = 1.5
        q = p / (p - 1.0)
        coeff = (p - 1.0) / (p ** q)
        y = cp.Variable(3)
        x_atom = cp.power(cp.Variable(3), p)
        conj_expr, conj_constraints = x_atom.conjugate(y)

        y_val = np.array([-1.0, 0.0, 2.0])
        prob = cp.Problem(cp.Minimize(cp.sum(conj_expr)), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)

        expected = coeff * np.sum(np.power(np.maximum(y_val, 0.0), q))
        self.assertAlmostEqual(prob.value, expected, places=5)

    def test_power_conjugate_general_p_even_power(self) -> None:
        """For p in {2,4,8,...}, conjugate is even in y."""
        p = 4
        q = p / (p - 1.0)
        coeff = (p - 1.0) / (p ** q)
        y = cp.Variable(2)
        x_atom = cp.power(cp.Variable(2), p)
        conj_expr, conj_constraints = x_atom.conjugate(y)

        y_val = np.array([-2.0, 3.0])
        prob = cp.Problem(cp.Minimize(cp.sum(conj_expr)), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)

        expected = coeff * np.sum(np.power(np.abs(y_val), q))
        self.assertAlmostEqual(prob.value, expected, places=5)

    def test_log_sum_exp_conjugate(self) -> None:
        y = cp.Variable(3)
        x_atom = cp.log_sum_exp(cp.Variable(3))
        conj_expr, conj_constraints = x_atom.conjugate(y)

        y_val = np.array([0.2, 0.3, 0.5])
        prob = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = np.sum(y_val * np.log(y_val))
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_log_sum_exp_conjugate_axis_scaled(self) -> None:
        y = cp.Variable((2, 3))
        x_atom = cp.log_sum_exp(cp.Variable((2, 3)), axis=0)
        scale = np.array([1.0, 2.0, 0.5])
        conj_expr, conj_constraints = x_atom.conjugate(y, perspective_scale=scale)

        y_val = np.array([[0.4, 0.8, 0.1], [0.6, 1.2, 0.4]])
        prob = cp.Problem(cp.Minimize(cp.sum(conj_expr)), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = np.sum(y_val * np.log(y_val / scale))
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_log_sum_exp_conjugate_axis1_scaled(self) -> None:
        y = cp.Variable((2, 3))
        x_atom = cp.log_sum_exp(cp.Variable((2, 3)), axis=1)
        scale = np.array([1.0, 0.5])
        conj_expr, conj_constraints = x_atom.conjugate(y, perspective_scale=scale)

        y_val = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.3]])
        prob = cp.Problem(cp.Minimize(cp.sum(conj_expr)), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = np.sum(y_val * np.log(y_val / scale[:, None]))
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_logistic_conjugate(self) -> None:
        y = cp.Variable(3)
        x_atom = cp.logistic(cp.Variable(3))
        conj_expr, conj_constraints = x_atom.conjugate(y)

        y_val = np.array([0.2, 0.3, 0.8])
        prob = cp.Problem(cp.Minimize(cp.sum(conj_expr)), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = np.sum(y_val * np.log(y_val) + (1.0 - y_val) * np.log(1.0 - y_val))
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_logistic_conjugate_scaled(self) -> None:
        y = cp.Variable(2)
        x_atom = cp.logistic(cp.Variable(2))
        scale = np.array([2.0, 0.5])
        conj_expr, conj_constraints = x_atom.conjugate(y, perspective_scale=scale)

        y_val = np.array([0.4, 0.1])
        prob = cp.Problem(cp.Minimize(cp.sum(conj_expr)), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = np.sum(y_val * np.log(y_val / scale) + (scale - y_val) * np.log((scale - y_val) / scale))
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_rel_entr_conjugate_constant_second_arg(self) -> None:
        y = cp.Variable(2)
        c = np.array([2.0, 0.5])
        x_atom = cp.rel_entr(cp.Variable(2), c)
        conj_expr, conj_constraints = x_atom.conjugate(y)

        y_val = np.array([0.2, -0.3])
        prob = cp.Problem(cp.Minimize(cp.sum(conj_expr)), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = np.sum(c * np.exp(y_val - 1.0))
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_kl_div_conjugate_constant_second_arg(self) -> None:
        y = cp.Variable(2)
        c = np.array([1.2, 0.7])
        x_atom = cp.kl_div(cp.Variable(2), c)
        conj_expr, conj_constraints = x_atom.conjugate(y)

        y_val = np.array([0.1, -0.2])
        prob = cp.Problem(cp.Minimize(cp.sum(conj_expr)), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = np.sum(c * (np.exp(y_val) - 1.0))
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_rel_entr_full_term_conjugate_indicator(self) -> None:
        atom = cp.rel_entr(cp.Variable(2), cp.Variable(2))
        u = cp.Variable(2)
        v = cp.Variable(2)
        conj_expr, conj_constraints = atom.conjugate_term((0, 1), (u, v))
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

    def test_kl_div_full_term_conjugate_indicator(self) -> None:
        atom = cp.kl_div(cp.Variable(2), cp.Variable(2))
        u = cp.Variable(2)
        v = cp.Variable(2)
        conj_expr, conj_constraints = atom.conjugate_term((0, 1), (u, v))
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

    def test_huber_conjugate(self) -> None:
        y = cp.Variable(3)
        M = 1.5
        x_atom = cp.huber(cp.Variable(3), M=M)
        conj_expr, conj_constraints = x_atom.conjugate(y)

        y_val = np.array([0.5, -1.0, 2.0])
        prob = cp.Problem(cp.Minimize(cp.sum(conj_expr)), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = np.sum(np.square(y_val) / 4.0)
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_huber_conjugate_scaled(self) -> None:
        y = cp.Variable(2)
        M = 2.0
        scale = np.array([2.0, 0.5])
        x_atom = cp.huber(cp.Variable(2), M=M)
        conj_expr, conj_constraints = x_atom.conjugate(y, perspective_scale=scale)

        y_val = np.array([1.0, 0.25])
        prob = cp.Problem(cp.Minimize(cp.sum(conj_expr)), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = np.sum(np.square(y_val) / (4.0 * scale))
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_log_det_negative_conjugate(self) -> None:
        y = cp.Variable((2, 2))
        x_atom = cp.log_det(cp.Variable((2, 2)))
        conj_expr, conj_constraints = x_atom.negative_conjugate(y)

        y_val = -np.eye(2)
        prob = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = -2.0 - np.log(np.linalg.det(-y_val))
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_log_det_negative_conjugate_scaled(self) -> None:
        y = cp.Variable((2, 2))
        x_atom = cp.log_det(cp.Variable((2, 2)))
        conj_expr, conj_constraints = x_atom.negative_conjugate(y, perspective_scale=2.0)

        y_val = -2.0 * np.eye(2)
        prob = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = -4.0 - 2.0 * np.log(np.linalg.det(-y_val / 2.0))
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_lambda_max_conjugate(self) -> None:
        y = cp.Variable((2, 2))
        x_atom = cp.lambda_max(cp.Variable((2, 2)))
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 3)

        y_val = np.array([[0.5, 0.0], [0.0, 0.5]])
        prob = cp.Problem(cp.Minimize(0), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)

    def test_sigma_max_conjugate_scaled(self) -> None:
        y = cp.Variable((2, 2))
        x_atom = cp.sigma_max(cp.Variable((2, 2)))
        conj_expr, conj_constraints = x_atom.conjugate(y, perspective_scale=2.0)
        self.assertEqual(conj_expr, 0)
        self.assertGreaterEqual(len(conj_constraints), 1)

        y_val = np.array([[0.3, 0.0], [0.0, 0.4]])
        prob = cp.Problem(cp.Minimize(0), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)

    def test_norm_nuc_conjugate_scaled(self) -> None:
        y = cp.Variable((2, 2))
        x_atom = cp.normNuc(cp.Variable((2, 2)))
        conj_expr, conj_constraints = x_atom.conjugate(y, perspective_scale=1.5)
        self.assertEqual(conj_expr, 0)
        self.assertGreaterEqual(len(conj_constraints), 1)

        y_val = np.array([[0.6, 0.0], [0.0, -0.4]])
        prob = cp.Problem(cp.Minimize(0), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)

    def test_lambda_sum_largest_conjugate(self) -> None:
        y = cp.Variable((3, 3))
        x_atom = cp.lambda_sum_largest(cp.Variable((3, 3)), 2)
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 4)

        y_val = np.diag([1.0, 1.0, 0.0])
        prob = cp.Problem(cp.Minimize(0), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)

    def test_matrix_frac_conjugate_vector(self) -> None:
        y = cp.Variable(3)
        P = np.diag([2.0, 1.0, 3.0])
        x_atom = cp.matrix_frac(cp.Variable(3), cp.Constant(P))
        conj_expr, conj_constraints = x_atom.conjugate(y)

        y_val = np.array([1.0, -2.0, 0.5])
        prob = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = 0.25 * y_val @ P @ y_val
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_matrix_frac_conjugate_matrix_scaled(self) -> None:
        y = cp.Variable((3, 2))
        P = np.diag([2.0, 1.0, 3.0])
        x_atom = cp.matrix_frac(cp.Variable((3, 2)), cp.Constant(P))
        conj_expr, conj_constraints = x_atom.conjugate(y, perspective_scale=2.0)

        y_val = np.array([[1.0, 0.3], [-2.0, 0.2], [0.5, -1.0]])
        prob = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = 0.125 * np.trace(y_val.T @ P @ y_val)
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_quad_over_lin_conjugate_complex(self) -> None:
        """quad_over_lin conjugate uses squared magnitude for complex inputs."""
        y = cp.Variable(2, complex=True)
        x_atom = cp.quad_over_lin(cp.Variable(2, complex=True), 2.0)
        conj_expr, conj_constraints = x_atom.conjugate(y)

        y.value = np.array([1.0 + 1.0j, 2.0 - 1.0j])
        expected = 0.5 * (2.0 + 5.0)
        self.assertAlmostEqual(conj_expr.value, expected, places=5)
        self.assertEqual(len(conj_constraints), 0)

    def test_quad_form_conjugate_singular_domain(self) -> None:
        """Singular PSD quad_form conjugate includes the range(P) constraint."""
        y = cp.Variable(2)
        x_atom = cp.quad_form(cp.Variable(2), np.array([[1.0, 0.0], [0.0, 0.0]]))
        conj_expr, conj_constraints = x_atom.conjugate(y)

        y.value = np.array([2.0, 0.0])
        self.assertAlmostEqual(conj_expr.value, 1.0, places=5)
        self.assertEqual(len(conj_constraints), 1)

    def test_quad_form_conjugate_sparse_psd(self) -> None:
        """quad_form conjugate supports sparse PSD P values."""
        y = cp.Variable(3)
        P = sp.diags([1.0, 2.0, 3.0], format="csc")
        x_atom = cp.quad_form(cp.Variable(3), P)
        conj_expr, conj_constraints = x_atom.conjugate(y)

        y.value = np.array([1.0, 2.0, 3.0])
        expected = 0.25 * (1.0**2 / 1.0 + 2.0**2 / 2.0 + 3.0**2 / 3.0)
        self.assertAlmostEqual(conj_expr.value, expected, places=6)
        self.assertEqual(len(conj_constraints), 0)

    def test_quad_form_parameter_matrix_supported(self) -> None:
        """quad_form conjugate supports PSD Parameter matrices."""
        y = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True, value=np.eye(2))
        x_atom = cp.quad_form(cp.Variable(2), P)
        conj_expr, conj_constraints = x_atom.conjugate(y)
        self.assertTrue(conj_expr.is_scalar())
        self.assertGreaterEqual(len(conj_constraints), 1)

    def test_quad_form_conjugate_scaled_complex_psd(self) -> None:
        """Scaled quad_form conjugate preserves complex Hermitian PSD structure."""
        y = cp.Variable(2, complex=True)
        P = np.array([[2.0, 1.0j], [-1.0j, 2.0]], dtype=complex)
        x_atom = cp.quad_form(cp.Variable(2, complex=True), P)
        conj_expr, conj_constraints = x_atom.conjugate(y, perspective_scale=2.0)

        self.assertEqual(len(conj_constraints), 1)
        self.assertTrue(conj_expr.is_real())

    def test_quad_form_negative_conjugate_nsd(self) -> None:
        y = cp.Variable(2)
        P = -np.diag([2.0, 1.0])
        x_atom = cp.quad_form(cp.Variable(2), P)
        conj_expr, conj_constraints = x_atom.negative_conjugate(y)

        y.value = np.array([2.0, -1.0])
        expected = 0.25 * (2.0**2 / 2.0 + (-1.0)**2 / 1.0)
        self.assertAlmostEqual(conj_expr.value, expected, places=6)
        self.assertEqual(len(conj_constraints), 0)

    def test_log_conjugate_not_implemented(self) -> None:
        """conjugate() is only implemented for convex atoms."""
        y = cp.Variable(2)
        x_atom = cp.log(cp.Variable(2))
        with self.assertRaises(NotImplementedError):
            x_atom.conjugate(y)

    def test_entr_conjugate_not_implemented(self) -> None:
        """conjugate() is only implemented for convex atoms."""
        y = cp.Variable(2)
        x_atom = cp.entr(cp.Variable(2))
        with self.assertRaises(NotImplementedError):
            x_atom.conjugate(y)

    def test_exp_conjugate_scaled(self) -> None:
        y = cp.Variable(2)
        x_atom = cp.exp(cp.Variable(2))
        conj_expr, conj_constraints = x_atom.conjugate(y, perspective_scale=2.0)
        y.value = np.array([2.0, 1.0])
        expected = np.array([-2.0, np.log(0.5) - 1.0])
        self.assertItemsAlmostEqual(conj_expr.value, expected, places=5)
        self.assertEqual(len(conj_constraints), 2)

    def test_quad_over_lin_conjugate_scaled(self) -> None:
        y = cp.Variable(2)
        x_atom = cp.quad_over_lin(cp.Variable(2), 2.0)
        conj_expr, conj_constraints = x_atom.conjugate(y, perspective_scale=2.0)
        y.value = np.array([1.0, 2.0])
        expected = 1.25  # t/(4*s) * ||y||^2 = 2/(8) * 5
        self.assertAlmostEqual(conj_expr.value, expected, places=6)
        self.assertEqual(len(conj_constraints), 0)

    def test_quad_over_lin_conjugate_axis_vector_scale(self) -> None:
        y = cp.Variable((2, 3))
        x_atom = cp.quad_over_lin(cp.Variable((2, 3)), 2.0, axis=0)
        scale = np.array([1.0, 2.0, 4.0])
        conj_expr, conj_constraints = x_atom.conjugate(y, perspective_scale=scale)

        y.value = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]])
        expected = (
            2.0 / (4.0 * 1.0) * (1.0**2 + 0.0**2)
            + 2.0 / (4.0 * 2.0) * (2.0**2 + (-1.0)**2)
            + 2.0 / (4.0 * 4.0) * (3.0**2 + 2.0**2)
        )
        self.assertAlmostEqual(conj_expr.value, expected, places=6)
        self.assertEqual(len(conj_constraints), 0)

    def test_log_entr_conjugate_accept_perspective_kwarg(self) -> None:
        y = cp.Variable(2)
        with self.assertRaises(NotImplementedError):
            cp.log(cp.Variable(2)).conjugate(y, perspective_scale=1.0)
        with self.assertRaises(NotImplementedError):
            cp.entr(cp.Variable(2)).conjugate(y, perspective_scale=1.0)

    def test_log_negative_conjugate_scaled_and_zero_closure(self) -> None:
        y = cp.Variable()
        x_atom = cp.log(cp.Variable())

        conj_expr, conj_constraints = x_atom.negative_conjugate(y, perspective_scale=2.0)
        y.value = -4.0
        self.assertAlmostEqual(conj_expr.value, -2.0 - 2.0 * np.log(2.0), places=6)
        self.assertEqual(len(conj_constraints), 2)

        zero_expr, zero_constraints = x_atom.negative_conjugate(y, perspective_scale=0.0)
        y.value = -1.0
        self.assertAlmostEqual(zero_expr.value, 0.0, places=8)
        self.assertEqual(len(zero_constraints), 2)

    def test_entr_negative_conjugate_scaled(self) -> None:
        y = cp.Variable()
        x_atom = cp.entr(cp.Variable())
        conj_expr, conj_constraints = x_atom.negative_conjugate(y, perspective_scale=2.0)

        y_val = 0.4
        prob = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == y_val])
        prob.solve(solver=cp.CLARABEL)
        expected = 2.0 * np.exp(y_val / 2.0 - 1.0)
        self.assertAlmostEqual(prob.value, expected, places=6)

    def test_minimum_negative_conjugate_term(self) -> None:
        atom = cp.minimum(cp.Variable(2), cp.Variable(2))
        y1 = cp.Variable(2)
        y2 = cp.Variable(2)
        conj_expr, conj_constraints = atom.negative_conjugate_term(
            (0, 1),
            (y1, y2),
            perspective_scale=2.0,
        )
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 3)

    def test_minimum_negative_conjugate_term_with_constant_arg(self) -> None:
        atom = cp.minimum(cp.Variable(2), cp.Constant(np.array([2.0, -1.0])))
        y = cp.Variable(2)
        conj_expr, conj_constraints = atom.negative_conjugate_term(
            (0,),
            (y,),
            perspective_scale=2.0,
        )

        y.value = np.array([-1.0, -2.0])
        self.assertItemsAlmostEqual(conj_expr.value, np.array([2.0, 0.0]), places=7)
        self.assertEqual(len(conj_constraints), 2)

    def test_maximum_conjugate_term(self) -> None:
        atom = cp.maximum(cp.Variable(2), cp.Variable(2))
        y1 = cp.Variable(2)
        y2 = cp.Variable(2)
        conj_expr, conj_constraints = atom.conjugate_term(
            (0, 1),
            (y1, y2),
            perspective_scale=2.0,
        )
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 3)

    def test_maximum_conjugate_term_with_constant_arg(self) -> None:
        atom = cp.maximum(cp.Variable(2), cp.Constant(np.array([2.0, -1.0])))
        y = cp.Variable(2)
        conj_expr, conj_constraints = atom.conjugate_term(
            (0,),
            (y,),
            perspective_scale=2.0,
        )

        y.value = np.array([1.0, 1.0])
        self.assertItemsAlmostEqual(conj_expr.value, np.array([-2.0, 1.0]), places=7)
        self.assertEqual(len(conj_constraints), 2)


class TestFenchelDualLP(BaseTest):
    """Test Fenchel dualization on linear programs."""

    def test_simple_lp(self) -> None:
        """Simple LP: min c'x s.t. x >= 0.

        Primal: min c'x s.t. x >= 0
        Dual:   max 0 s.t. lambda = c, lambda >= 0
        i.e.:   max 0 s.t. c >= 0 (feasible iff c >= 0, opt val = 0 if bounded)

        Better test: min c'x s.t. Ax <= b, x >= 0
        """
        n = 3
        x = cp.Variable(n)
        c = np.array([1.0, 2.0, 3.0])
        A = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [1.0, 1.0, 1.0]])
        b = np.array([4.0, 5.0, 6.0, 10.0])

        primal = cp.Problem(
            cp.Minimize(c @ x),
            [A @ x <= b, x >= 0]
        )
        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        # The optimal value should be 0 (x = 0 is feasible).
        self.assertAlmostEqual(primal_val, 0.0, places=4)

    def test_lp_strong_duality(self) -> None:
        """LP with nontrivial optimal: verify strong duality.

        min  c'x  s.t. Ax <= b, x >= 0
        """
        c = np.array([-1.0, -2.0])
        A = np.array([[1.0, 1.0],
                       [1.0, 0.0],
                       [0.0, 1.0]])
        b = np.array([4.0, 3.0, 3.0])

        x = cp.Variable(2)
        primal = cp.Problem(
            cp.Minimize(c @ x),
            [A @ x <= b, x >= 0]
        )
        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        dual_val = dual.value

        self.assertAlmostEqual(primal_val, -7.0, places=3)
        self.assertAlmostEqual(dual_val, primal_val, places=3)

    def test_scalar_variable_lp(self) -> None:
        """Scalar-variable LP dualizes without shape errors."""
        x = cp.Variable()
        primal = cp.Problem(cp.Minimize(x), [x >= 0])
        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        dual_val = dual.value

        self.assertAlmostEqual(primal_val, dual_val, places=6)

    def test_zero_coefficient_atom_is_ignored(self) -> None:
        """0 * atom(x) should not alter the dual problem."""
        x = cp.Variable()
        primal = cp.Problem(cp.Minimize(0 * cp.exp(x) + x), [x >= 0])
        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        dual_val = dual.value

        self.assertAlmostEqual(primal_val, dual_val, places=6)

    def test_maximization_problem_supported(self) -> None:
        x = cp.Variable()
        primal = cp.Problem(cp.Maximize(x), [x <= 1, x >= 0])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(primal.value, dual.value, places=6)

    def test_zero_variable_problem_supported(self) -> None:
        primal = cp.Problem(cp.Minimize(cp.Constant(3.0)))
        dual = fenchel_dual(primal)
        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(primal.value, 3.0, places=6)
        self.assertAlmostEqual(dual.value, primal.value, places=6)


class TestFenchelDualNorm(BaseTest):
    """Test Fenchel dualization on norm-based problems."""

    def test_norm1_minimization(self) -> None:
        """min ||x||_1 s.t. Ax = b.

        Dual: max b'nu s.t. ||A'nu||_inf <= 1.
        """
        np.random.seed(42)
        n, m = 5, 3
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cp.Variable(n)
        primal = cp.Problem(
            cp.Minimize(cp.norm1(x)),
            [A @ x == b]
        )
        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        dual_val = dual.value

        # Strong duality should hold.
        self.assertAlmostEqual(primal_val, dual_val, places=3)

    def test_norm_inf_minimization(self) -> None:
        """min ||x||_inf s.t. Ax = b.

        Dual: max b'nu s.t. ||A'nu||_1 <= 1.
        """
        np.random.seed(123)
        n, m = 4, 2
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cp.Variable(n)
        primal = cp.Problem(
            cp.Minimize(cp.norm_inf(x)),
            [A @ x == b]
        )
        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        dual_val = dual.value

        self.assertAlmostEqual(primal_val, dual_val, places=3)

    def test_axis_norm1_sum_minimization(self) -> None:
        """min sum(norm1(X, axis=0)) with affine constraints."""
        np.random.seed(99)
        X = cp.Variable((2, 3))
        B = np.random.randn(2, 3)
        primal = cp.Problem(
            cp.Minimize(cp.sum(cp.norm1(X, axis=0))),
            [X == B]
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=4)

    def test_axis_sum_with_explicit_axis_scalarized(self) -> None:
        """Explicit axis reductions that scalarize should decompose correctly."""
        np.random.seed(98)
        X = cp.Variable((2, 3))
        B = np.random.randn(2, 3)
        primal = cp.Problem(
            cp.Minimize(cp.sum(cp.norm1(X, axis=0), axis=0)),
            [X == B],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=4)


class TestFenchelDualQP(BaseTest):
    """Test Fenchel dualization on quadratic programs."""

    def test_sum_squares(self) -> None:
        """min ||x||^2 s.t. Ax = b.

        Dual: max b'nu - (1/4)||A'nu||^2.
        """
        np.random.seed(7)
        n, m = 4, 2
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cp.Variable(n)
        primal = cp.Problem(
            cp.Minimize(cp.sum_squares(x)),
            [A @ x == b]
        )
        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        dual_val = dual.value

        self.assertAlmostEqual(primal_val, dual_val, places=3)

    def test_quad_form_singular_psd(self) -> None:
        """Singular PSD quad_form objective dualizes correctly."""
        P = np.diag([1.0, 0.0, 3.0])
        c = np.array([2.0, 0.0, 6.0])  # in range(P)
        x = cp.Variable(3)
        primal = cp.Problem(cp.Minimize(cp.quad_form(x, P) + c @ x))
        primal.solve(solver=cp.CLARABEL)

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(primal.value, -4.0, places=5)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_lambda_max_objective(self) -> None:
        X = cp.Variable((2, 2))
        C = np.array([[1.0, 0.3], [0.3, -0.5]])
        primal = cp.Problem(cp.Minimize(cp.lambda_max(X) + 0.2 * cp.sum_squares(X - C)))
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_sigma_max_objective(self) -> None:
        X = cp.Variable((2, 3))
        C = np.array([[0.8, -0.3, 0.1], [0.2, 0.4, -0.7]])
        primal = cp.Problem(cp.Minimize(cp.sigma_max(X) + 0.1 * cp.sum_squares(X - C)))
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_norm_nuc_objective(self) -> None:
        X = cp.Variable((2, 3))
        C = np.array([[0.7, -0.2, 0.2], [0.1, 0.5, -0.6]])
        primal = cp.Problem(cp.Minimize(cp.normNuc(X) + 0.15 * cp.sum_squares(X - C)))
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_lambda_sum_largest_objective(self) -> None:
        X = cp.Variable((3, 3), symmetric=True)
        C = np.array([[0.5, 0.1, -0.2], [0.1, -0.3, 0.2], [-0.2, 0.2, 0.7]])
        primal = cp.Problem(cp.Minimize(cp.lambda_sum_largest(X, 2) + 0.1 * cp.sum_squares(X - C)))
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_matrix_frac_objective_with_constant_expression(self) -> None:
        x = cp.Variable(3)
        P = cp.Constant(np.diag([2.0, 1.0, 3.0]))
        c = np.array([0.8, -0.4, 0.2])
        primal = cp.Problem(cp.Minimize(cp.matrix_frac(x, P) + c @ x))
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)


class TestFenchelDualExpCone(BaseTest):
    """Test Fenchel dualization on problems with exp/log atoms."""

    def test_sum_exp(self) -> None:
        """min sum(exp(x)) s.t. sum(x) = 1.

        By symmetry and convexity, optimum at x = (1/n, ..., 1/n),
        value = n * exp(1/n).
        Dual should achieve the same value.
        """
        n = 3
        x = cp.Variable(n)
        primal = cp.Problem(
            cp.Minimize(cp.sum(cp.exp(x))),
            [cp.sum(x) == 1]
        )
        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        dual_val = dual.value

        expected = n * np.exp(1.0 / n)
        self.assertAlmostEqual(primal_val, expected, places=3)
        self.assertAlmostEqual(dual_val, primal_val, places=3)

    def test_sum_exp_div_scalar(self) -> None:
        """min sum(exp(x) / c) with scalar c should parse like scaling."""
        n = 3
        c = 2.0
        x = cp.Variable(n)
        primal = cp.Problem(
            cp.Minimize(cp.sum(cp.exp(x) / c)),
            [cp.sum(x) == 1]
        )
        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        dual_val = dual.value

        expected = (n * np.exp(1.0 / n)) / c
        self.assertAlmostEqual(primal_val, expected, places=3)
        self.assertAlmostEqual(dual_val, primal_val, places=3)

    def test_log_sum_exp(self) -> None:
        """min log_sum_exp(x) s.t. sum(x)=1."""
        n = 4
        x = cp.Variable(n)
        primal = cp.Problem(cp.Minimize(cp.log_sum_exp(x)), [cp.sum(x) == 1.0])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        expected = np.log(n) + 1.0 / n
        self.assertAlmostEqual(primal.value, expected, places=4)
        self.assertAlmostEqual(dual.value, primal.value, places=4)

    def test_sum_power_general_p(self) -> None:
        """General p>1 power atom in objective dualizes correctly."""
        x = cp.Variable(3)
        c = np.array([0.2, 0.1, 0.3])
        primal = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 1.5)) + c @ x), [x >= 0])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_sum_logistic(self) -> None:
        """logistic objective dualizes correctly."""
        x = cp.Variable(4)
        primal = cp.Problem(cp.Minimize(cp.sum(cp.logistic(x))), [cp.sum(x) == 0.5])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_sum_kl_div_constant_second_arg(self) -> None:
        x = cp.Variable(3)
        c = np.array([1.0, 2.0, 0.8])
        primal = cp.Problem(cp.Minimize(cp.sum(cp.kl_div(x, c)) + 0.1 * cp.sum_squares(x)))
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_sum_kl_div_two_arg(self) -> None:
        x = cp.Variable(3)
        y = cp.Variable(3)
        primal = cp.Problem(
            cp.Minimize(cp.sum(cp.kl_div(x, y)) + 0.1 * cp.sum_squares(x - 1.0) + 0.1 * cp.sum_squares(y - 2.0)),
            [x >= 0.1, y >= 0.1],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_sum_rel_entr_two_arg(self) -> None:
        x = cp.Variable(3)
        y = cp.Variable(3)
        primal = cp.Problem(
            cp.Minimize(cp.sum(cp.rel_entr(x, y)) + 0.1 * cp.sum_squares(x - 1.0) + 0.1 * cp.sum_squares(y - 2.0)),
            [x >= 0.1, y >= 0.1],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_sum_huber(self) -> None:
        x = cp.Variable(4)
        primal = cp.Problem(cp.Minimize(cp.sum(cp.huber(x, M=1.2)) + 0.3 * cp.sum(x)), [cp.sum(x) == 0.4])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_neg_log_det(self) -> None:
        X = cp.Variable((2, 2), symmetric=True)
        C = np.eye(2)
        primal = cp.Problem(cp.Minimize(-cp.log_det(X) + cp.trace(C @ X)), [X >> 1e-3 * np.eye(2)])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_neg_sum_log(self) -> None:
        """min -sum(log(x)) over simplex; dual should match primal value."""
        n = 4
        x = cp.Variable(n)
        primal = cp.Problem(
            cp.Minimize(-cp.sum(cp.log(x))),
            [cp.sum(x) == 1, x >= 1e-3]
        )
        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        dual_val = dual.value

        expected = n * np.log(n)
        self.assertAlmostEqual(primal_val, expected, places=3)
        self.assertAlmostEqual(dual_val, primal_val, places=3)

    def test_neg_sum_entr(self) -> None:
        """min -sum(entr(x)) over simplex; dual should match primal value."""
        n = 5
        x = cp.Variable(n)
        primal = cp.Problem(
            cp.Minimize(-cp.sum(cp.entr(x))),
            [cp.sum(x) == 1, x >= 0]
        )
        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        dual_val = dual.value

        expected = -np.log(n)
        self.assertAlmostEqual(primal_val, expected, places=3)
        self.assertAlmostEqual(dual_val, primal_val, places=3)


class TestFenchelDualConicConstraints(BaseTest):
    """Test conic primal constraints in Fenchel dualization."""

    def test_soc_constraint(self) -> None:
        x = cp.Variable(3)
        t = cp.Variable()
        b = np.array([1.0, -2.0, 2.0])
        primal = cp.Problem(cp.Minimize(t), [cp.SOC(t, x), x == b])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(primal.value, np.linalg.norm(b), places=5)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_psd_constraint(self) -> None:
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        X = cp.Variable((2, 2), symmetric=True)
        primal = cp.Problem(cp.Minimize(cp.trace(X)), [X - A >> 0])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(primal.value, np.trace(A), places=5)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_expcone_constraint(self) -> None:
        z = cp.Variable()
        primal = cp.Problem(cp.Minimize(z), [cp.ExpCone(0.0, 1.0, z)])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(primal.value, 1.0, places=5)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_non_affine_norm_inequality(self) -> None:
        np.random.seed(10)
        x = cp.Variable(5)
        c = np.random.randn(5)
        primal = cp.Problem(cp.Minimize(cp.sum_squares(x - c)), [cp.norm(x) <= 1.2])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_non_affine_pnorm_inequality_perspective(self) -> None:
        """pnorm inequality dualizes via perspective rules (no cone pre-lowering)."""
        np.random.seed(11)
        x = cp.Variable(4)
        c = np.random.randn(4)
        primal = cp.Problem(
            cp.Minimize(cp.sum_squares(x - c)),
            [cp.pnorm(x, 1.5, approx=False) <= 1.0],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_non_affine_exp_inequality(self) -> None:
        """exp inequality dualization remains correct."""
        x = cp.Variable()
        primal = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.exp(x) <= 2.0])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_non_affine_axis_sum_exp_inequality_supported(self) -> None:
        """Axis-sum exp inequality dualizes directly via broadcasted perspective scales."""
        X = cp.Variable((2, 3))
        rhs = np.array([2.5, 2.7, 2.9])
        constraint = cp.sum(cp.exp(X), axis=0) <= rhs

        primal = cp.Problem(cp.Minimize(cp.sum_squares(X)), [constraint])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_non_affine_neg_log_inequality_perspective(self) -> None:
        """-log inequality is recognized by direct perspective dualization."""
        x = cp.Variable(pos=True)
        constraint = -cp.log(x) <= 1.0

        primal = cp.Problem(cp.Minimize(x), [constraint])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_non_affine_square_inequality_falls_back_to_canonicalization(self) -> None:
        """square inequality should not be misclassified as perspective-supported."""
        x = cp.Variable()
        primal = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.square(x) <= 1.0])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_non_affine_axis_quad_over_lin_inequality_supported(self) -> None:
        """Axis quad_over_lin inequality dualizes directly."""
        X = cp.Variable((2, 3))
        primal = cp.Problem(
            cp.Minimize(cp.sum_squares(X)),
            [cp.sum_squares(X, axis=0) <= 1.0],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_non_affine_axis_quad_over_lin_variable_denom_supported(self) -> None:
        """Axis quad_over_lin with variable denominator dualizes directly."""
        X = cp.Variable((2, 3))
        t = cp.Variable(nonneg=True)
        primal = cp.Problem(
            cp.Minimize(cp.sum_squares(X) + 0.5 * t),
            [cp.quad_over_lin(X, t, axis=0) <= 1.0, t >= 0.5],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_non_affine_quad_over_lin_variable_denom_supported(self) -> None:
        """quad_over_lin with variable denominator dualizes directly."""
        x = cp.Variable(2)
        t = cp.Variable(nonneg=True)
        primal = cp.Problem(
            cp.Minimize(cp.sum_squares(x) + t),
            [cp.quad_over_lin(x, t) <= 1.0, t >= 0.5],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_non_affine_quad_over_lin_constant_numerator_axis_supported(self) -> None:
        """quad_over_lin(constant, t, axis) inequality dualizes directly."""
        C = np.array([[1.0, -2.0, 0.5], [2.0, 1.0, -1.5]])
        t = cp.Variable(nonneg=True)
        primal = cp.Problem(cp.Minimize(t), [cp.quad_over_lin(C, t, axis=0) <= 1.0])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)


class TestFenchelDualMultiVariable(BaseTest):
    """Multiple primal variables are dualized via stacked stationarity."""

    def test_two_var_lp(self) -> None:
        x = cp.Variable(2)
        z = cp.Variable(2)
        primal = cp.Problem(
            cp.Minimize(cp.sum(x) + cp.sum(z)),
            [x + z >= 1, x >= 0, z >= 0]
        )

        primal.solve(solver=cp.CLARABEL)
        primal_val = primal.value

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        dual_val = dual.value

        self.assertAlmostEqual(primal_val, 2.0, places=3)
        self.assertAlmostEqual(dual_val, primal_val, places=3)


class TestFenchelDualUnsupportedDomains(BaseTest):
    """Unsupported domain/type checks."""

    def test_nonneg_variable_attribute_is_respected(self) -> None:
        x = cp.Variable(nonneg=True)
        primal = cp.Problem(cp.Minimize(x))
        dual = fenchel_dual(primal)

        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(primal.value, 0.0, places=6)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_bounds_variable_attribute_is_respected(self) -> None:
        x = cp.Variable(bounds=(0.0, 1.0))
        primal = cp.Problem(cp.Minimize(-x))
        dual = fenchel_dual(primal)

        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(primal.value, -1.0, places=6)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_diag_variable_attribute_is_respected(self) -> None:
        X = cp.Variable((2, 2), diag=True)
        primal = cp.Problem(
            cp.Minimize(cp.sum(X)),
            [X[0, 0] >= 1.0, X[1, 1] >= 2.0],
        )
        dual = fenchel_dual(primal)

        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(primal.value, 3.0, places=6)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_sparsity_variable_attribute_is_respected(self) -> None:
        sparsity = (np.array([0, 1]), np.array([0, 1]))
        X = cp.Variable((2, 2), sparsity=sparsity)
        primal = cp.Problem(
            cp.Minimize(X[0, 0] + X[1, 1]),
            [X[0, 0] >= 1.0, X[1, 1] >= 2.0],
        )
        dual = fenchel_dual(primal)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Reading from a sparse CVXPY expression via `.value` is discouraged",
                category=RuntimeWarning,
            )
            primal.solve(solver=cp.CLARABEL)
            dual.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(primal.value, 3.0, places=6)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_complex_primal_variable_supported(self) -> None:
        x = cp.Variable(2, complex=True)
        primal = cp.Problem(cp.Minimize(cp.norm1(x)), [cp.real(x[0]) == 1.0, cp.imag(x[1]) == -2.0])
        dual = fenchel_dual(primal)

        primal.solve(solver=cp.SCS, eps=1e-6, max_iters=10000)
        dual.solve(solver=cp.SCS, eps=1e-6, max_iters=10000)

        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_integer_variable_not_supported(self) -> None:
        x = cp.Variable(integer=True)
        primal = cp.Problem(cp.Minimize(x), [x >= 0])

        with self.assertRaises(ValueError):
            fenchel_dual(primal)

    def test_parameterized_problem_supported(self) -> None:
        x = cp.Variable()
        c = cp.Parameter(nonneg=True, value=2.0)
        a = cp.Parameter(pos=True, value=4.0)
        primal = cp.Problem(cp.Minimize(c * x), [a * x >= 1.0])

        dual = fenchel_dual(primal)

        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

        c.value = 3.0
        a.value = 6.0
        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_parameterized_scalar_nonlinear_multiplier_supported(self) -> None:
        alpha = cp.Parameter(nonneg=True, value=2.0)
        x = cp.Variable()
        primal = cp.Problem(cp.Minimize(alpha * cp.exp(x) + x), [x >= 0])
        dual = fenchel_dual(primal)

        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

        alpha.value = 5.0
        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_parameterized_scalar_division_not_dpp(self) -> None:
        alpha = cp.Parameter(pos=True, value=2.0)
        x = cp.Variable()
        primal = cp.Problem(cp.Minimize(cp.exp(x) / alpha + x), [x >= 0])
        with self.assertRaises(NotImplementedError):
            fenchel_dual(primal)

    def test_parameterized_negative_scalar_on_concave_atom_supported(self) -> None:
        beta = cp.Parameter(nonneg=True, value=2.0)
        x = cp.Variable()
        primal = cp.Problem(cp.Minimize(-beta * cp.log(x)), [x >= 1.0, x <= 3.0])
        dual = fenchel_dual(primal)

        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

        beta.value = 4.0
        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_negative_concave_quad_form_term_supported(self) -> None:
        x = cp.Variable(2)
        b = np.array([1.0, -2.0])
        P = -np.diag([2.0, 1.0])  # NSD, so quad_form(x, P) is concave.
        primal = cp.Problem(cp.Minimize(-cp.quad_form(x, P)), [x == b])
        dual = fenchel_dual(primal)

        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_negative_concave_minimum_multiarg_term_supported(self) -> None:
        x = cp.Variable(3)
        z = cp.Variable(3)
        a = np.array([1.0, -2.0, 0.5])
        b = np.array([-1.0, -1.0, 2.0])
        primal = cp.Problem(cp.Minimize(-cp.sum(cp.minimum(x, z))), [x == a, z == b])
        dual = fenchel_dual(primal)

        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_negative_concave_minimum_with_constant_arg_supported(self) -> None:
        x = cp.Variable(3)
        c = np.array([0.25, -1.5, 1.0])
        d = np.array([1.0, -0.5, 2.0])
        primal = cp.Problem(
            cp.Minimize(cp.sum_squares(x - d) - cp.sum(cp.minimum(x, c))),
        )
        dual = fenchel_dual(primal)

        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_parameterized_scalar_nonlinear_multiplier_unknown_sign_not_supported(self) -> None:
        alpha = cp.Parameter(value=2.0)
        x = cp.Variable()
        primal = cp.Problem(cp.Minimize(alpha * cp.exp(x) + x), [x >= 0])
        with self.assertRaises(ValueError):
            fenchel_dual(primal)

    def test_parameterized_quad_over_lin_not_dpp(self) -> None:
        x = cp.Variable(2)
        t = cp.Parameter(pos=True, value=2.0)
        primal = cp.Problem(
            cp.Minimize(cp.quad_over_lin(x, t) + 0.1 * cp.norm1(x)),
            [cp.sum(x) == 1, x >= 0],
        )
        with self.assertRaises(NotImplementedError):
            fenchel_dual(primal)

    def test_parameterized_scalar_quad_form_multiplier_supported(self) -> None:
        alpha = cp.Parameter(nonneg=True, value=3.0)
        P = np.diag([2.0, 1.0])
        c = np.array([1.0, 2.0])
        x = cp.Variable(2)
        primal = cp.Problem(cp.Minimize(alpha * cp.quad_form(x, P) + c @ x))
        dual = fenchel_dual(primal)

        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

        alpha.value = 5.0
        primal.solve(solver=cp.CLARABEL)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_parameterized_psd_quad_form_constraint_not_dpp(self) -> None:
        x = cp.Variable(2)
        P = cp.Parameter((2, 2), PSD=True, value=np.eye(2))
        primal = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.quad_form(x, P) <= 1.0])
        with self.assertRaises(NotImplementedError):
            fenchel_dual(primal)

    def test_nested_non_affine_objective_supported(self) -> None:
        x = cp.Variable(2)
        primal = cp.Problem(cp.Minimize(cp.exp(cp.norm(x))), [cp.sum(x) == 1.0])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(dual.value, primal.value, places=5)

    def test_objective_multiarg_atom_nonconstant_arg_supported(self) -> None:
        x = cp.Variable(2)
        t = cp.Variable(nonneg=True)
        primal = cp.Problem(cp.Minimize(cp.quad_over_lin(x, t)), [t >= 1.0])

        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_objective_quad_over_lin_constant_numerator_supported(self) -> None:
        c = np.array([1.0, -2.0, 3.0])
        t = cp.Variable(nonneg=True)
        primal = cp.Problem(cp.Minimize(cp.quad_over_lin(c, t) + 0.5 * t))

        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(dual.value, primal.value, places=6)

    def test_objective_elementwise_maximum_constant_arg_supported(self) -> None:
        x = cp.Variable(4)
        c = np.array([0.5, -1.0, 1.5, 0.0])
        d = np.array([1.0, -0.2, 2.0, -1.0])
        primal = cp.Problem(cp.Minimize(0.3 * cp.sum_squares(x - d) + cp.sum(cp.maximum(x, c))))

        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(dual.value, primal.value, places=5)


class TestFenchelDualReduction(BaseTest):
    def test_apply_and_invert_solution_value(self) -> None:
        x = cp.Variable(2)
        d = np.array([1.0, -2.0])
        primal = cp.Problem(cp.Minimize(cp.sum_squares(x - d)))
        primal.solve(solver=cp.CLARABEL)

        reduction = FenchelDual()
        dual, inv = reduction.apply(primal)
        dual.solve(solver=cp.CLARABEL)
        recovered = reduction.invert(dual._solution, inv)

        self.assertAlmostEqual(recovered.opt_val, primal.value, places=6)
        self.assertIn(x.id, recovered.primal_vars)
        self.assertItemsAlmostEqual(recovered.primal_vars[x.id], x.value, places=5)
        self.assertEqual(recovered.dual_vars, {})

    def test_apply_and_invert_solution_value_complex(self) -> None:
        x = cp.Variable(2, complex=True)
        d = np.array([1.0 + 2.0j, -3.0 + 0.5j])
        primal = cp.Problem(cp.Minimize(cp.sum_squares(cp.abs(x - d))))
        primal.solve(solver=cp.SCS, eps=1e-6, max_iters=10000)

        reduction = FenchelDual()
        dual, inv = reduction.apply(primal)
        dual.solve(solver=cp.SCS, eps=1e-6, max_iters=10000)
        recovered = reduction.invert(dual._solution, inv)

        self.assertIn(x.id, recovered.primal_vars)
        self.assertItemsAlmostEqual(recovered.primal_vars[x.id], x.value, places=4)

    def test_accepts_supported_and_rejects_integer(self) -> None:
        reduction = FenchelDual()
        self.assertTrue(reduction.accepts(cp.Problem(cp.Minimize(cp.norm1(cp.Variable(2))))))
        self.assertTrue(reduction.accepts(cp.Problem(cp.Minimize(cp.norm1(cp.Variable(2, complex=True))))))
        self.assertFalse(reduction.accepts(cp.Problem(cp.Minimize(cp.Variable(integer=True)))))

    def test_invert_status_mapping(self) -> None:
        reduction = FenchelDual()
        sol_infeasible = Solution(s.INFEASIBLE, np.inf, {}, {}, {})
        mapped_infeasible = reduction.invert(sol_infeasible, {})
        self.assertEqual(mapped_infeasible.status, s.UNBOUNDED)

        sol_unbounded = Solution(s.UNBOUNDED, -np.inf, {}, {}, {})
        mapped_unbounded = reduction.invert(sol_unbounded, {})
        self.assertEqual(mapped_unbounded.status, s.INFEASIBLE)
