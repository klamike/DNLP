"""Tests for composition-based Fenchel dualization."""

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.reductions.fenchel_rules as fenchel_rules
from cvxpy.reductions.fenchel_dual import FenchelDual, fenchel_dual
from cvxpy.reductions.solution import Solution
from cvxpy.tests.base_test import BaseTest


class TestFenchelDual(BaseTest):

    def test_l1_linear_constraints_value_match(self) -> None:
        rng = np.random.default_rng(0)
        x = cp.Variable(4)
        A = rng.normal(size=(6, 4))
        b = np.ones(6)
        c = rng.normal(size=4)

        primal = cp.Problem(cp.Minimize(cp.norm1(x) + c @ x), [A @ x <= b])
        primal.solve(solver=cp.CLARABEL)

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_nested_non_affine_objective(self) -> None:
        x = cp.Variable(3)
        primal = cp.Problem(cp.Minimize(cp.exp(cp.norm1(x))))
        primal.solve(solver=cp.CLARABEL)

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, 1.0, places=6)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_nested_non_affine_without_lifting_fallback(self) -> None:
        x = cp.Variable(2)
        primal = cp.Problem(cp.Minimize(cp.exp(cp.norm1(x))))

        # Ensure dualization succeeds while no lifting helper exists in the rules module.
        self.assertFalse(hasattr(fenchel_rules, "_lift_nonaffine_atom"))
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(dual.status, cp.OPTIMAL)

    def test_non_affine_inequality(self) -> None:
        x = cp.Variable(2)
        primal = cp.Problem(cp.Minimize(cp.max(x)), [cp.norm1(x) <= 1])
        primal.solve(solver=cp.CLARABEL)

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_log_sum_exp_supported(self) -> None:
        x = cp.Variable(3)
        A = np.array([[1.0, -1.0, 0.5], [0.2, 0.3, -0.4], [0.1, -0.2, 0.7]])
        b = np.array([0.3, -0.5, 0.2])
        primal = cp.Problem(cp.Minimize(cp.log_sum_exp(A @ x + b) + 0.2 * cp.norm1(x)))
        primal.solve(solver=cp.CLARABEL)

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_logistic_supported(self) -> None:
        x = cp.Variable(3)
        A = np.array([[0.6, -0.4, 0.3], [-0.2, 0.7, -0.1], [0.4, 0.2, -0.5]])
        b = np.array([0.1, -0.3, 0.2])
        primal = cp.Problem(cp.Minimize(cp.sum(cp.logistic(A @ x + b)) + 0.1 * cp.norm1(x)))
        primal.solve(solver=cp.CLARABEL)

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_rel_entr_supported_constant_second_arg(self) -> None:
        x = cp.Variable(3, nonneg=True)
        c = np.array([1.2, 0.8, 2.0])
        primal = cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(x, c)) + 0.2 * cp.sum(x)))
        primal.solve(solver=cp.CLARABEL)

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_powcone3d_constraint_supported(self) -> None:
        x = cp.Variable(nonneg=True)
        y = cp.Variable(nonneg=True)
        z = cp.Variable()
        primal = cp.Problem(cp.Minimize(x + y), [cp.PowCone3D(x, y, z, 0.5), z == 1.0])
        primal.solve(solver=cp.SCS, eps=1e-6)

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.SCS, eps=1e-6)

        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=4)

    def test_powconend_constraint_supported(self) -> None:
        w = cp.Variable(2, nonneg=True)
        z = cp.Variable()
        alpha = cp.Constant([0.3, 0.7])
        primal = cp.Problem(cp.Minimize(cp.sum(w)), [cp.PowConeND(w, z, alpha), z == 1.0])
        primal.solve(solver=cp.SCS, eps=1e-6)

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.SCS, eps=1e-6)

        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=4)

    def test_axis_sum_with_nonscalar_multiplier(self) -> None:
        X = cp.Variable((2, 3))
        w = np.array([1.0, 2.0, 0.5])
        expr = cp.sum(cp.multiply(w, cp.sum(cp.exp(X), axis=0))) + 0.1 * cp.norm1(X)
        primal = cp.Problem(cp.Minimize(expr))
        primal.solve(solver=cp.CLARABEL)

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_dpp_parameter_affine_adjoint_supported(self) -> None:
        x = cp.Variable(3)
        a = cp.Parameter(3, value=np.array([0.2, -0.1, 0.3]))
        primal = cp.Problem(cp.Minimize(cp.norm1(x) + a @ x))
        primal.solve(solver=cp.CLARABEL)

        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)

        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_maximize_supported(self) -> None:
        x = cp.Variable(2)
        primal = cp.Problem(cp.Maximize(-cp.norm1(x) - 1), [x >= -1])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_maximize_concave_obj(self) -> None:
        x = cp.Variable()
        primal = cp.Problem(cp.Maximize(x), [x <= 1])
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_sum_squares_supported(self) -> None:
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x) + cp.norm1(x)))
        prob.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(prob)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob.value, dual.value, places=5)

    def test_inv_pos_supported_via_generic_fallback(self) -> None:
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.inv_pos(x)) + 0.1 * cp.norm1(x)), [x >= 1])
        prob.solve(solver=cp.SCS, eps=1e-6)
        dual = fenchel_dual(prob)
        dual.solve(solver=cp.SCS, eps=1e-6)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob.value, dual.value, places=4)

    def test_geo_mean_constraint_supported_via_generic_fallback(self) -> None:
        x = cp.Variable(2, nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [cp.geo_mean(x) >= 1.0])
        prob.solve(solver=cp.SCS, eps=1e-6)
        dual = fenchel_dual(prob)
        dual.solve(solver=cp.SCS, eps=1e-6)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob.value, dual.value, places=4)

    def test_unsupported_cone_rejected_without_fallback(self) -> None:
        x = cp.Variable()
        y = cp.Variable()
        z = cp.Variable()
        prob = cp.Problem(cp.Minimize(0), [cp.RelEntrConeQuad(x, y, z, m=3, k=3)])
        with self.assertRaises(NotImplementedError):
            fenchel_dual(prob)

    def test_reduction_interface(self) -> None:
        x = cp.Variable(2)
        primal = cp.Problem(cp.Minimize(cp.norm1(x)), [x == np.array([0.2, -0.1])])
        red = FenchelDual()
        self.assertTrue(red.accepts(primal))
        dual, inv = red.apply(primal)
        dual.solve(solver=cp.CLARABEL)
        sol = red.invert(dual.solution, inv)
        self.assertEqual(sol.status, cp.OPTIMAL)

    # ------------------------------------------------------------------
    # Atom conjugate tests
    # ------------------------------------------------------------------

    def test_abs_conjugate(self) -> None:
        x = cp.Variable(3)
        primal = cp.Problem(
            cp.Minimize(cp.sum(cp.abs(x)) + 0.5 * cp.sum_squares(x))
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_huber_conjugate(self) -> None:
        x = cp.Variable(3)
        primal = cp.Problem(
            cp.Minimize(cp.sum(cp.huber(x, M=2)) + 0.1 * cp.norm1(x)),
            [cp.norm_inf(x) <= 5],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_power_p_gt_1_conjugate(self) -> None:
        x = cp.Variable(3, nonneg=True)
        primal = cp.Problem(
            cp.Minimize(cp.sum(cp.power(x, 3)) + 0.1 * cp.sum(x)),
            [cp.sum(x) >= 1],
        )
        primal.solve(solver=cp.SCS, eps=1e-6)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.SCS, eps=1e-6)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=4)

    def test_power_concave_negative_conjugate(self) -> None:
        """Test -power(x, p) for 0 < p < 1 (convex objective from concave atom)."""
        x = cp.Variable(3, nonneg=True)
        primal = cp.Problem(
            cp.Minimize(-cp.sum(cp.power(x, 0.5)) + cp.sum(x)),
            [cp.sum(x) <= 5],
        )
        primal.solve(solver=cp.SCS, eps=1e-6)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.SCS, eps=1e-6)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=4)

    def test_power_concave_p_third(self) -> None:
        """Test -power(x, 1/3) for p=1/3."""
        x = cp.Variable(2, nonneg=True)
        primal = cp.Problem(
            cp.Minimize(-cp.sum(cp.power(x, 1.0 / 3.0)) + 2 * cp.sum(x)),
            [cp.sum(x) <= 3],
        )
        primal.solve(solver=cp.SCS, eps=1e-6)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.SCS, eps=1e-6)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=4)

    def test_quad_form_conjugate(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.normal(size=(3, 3))
        P = A.T @ A + 0.1 * np.eye(3)
        x = cp.Variable(3)
        primal = cp.Problem(
            cp.Minimize(cp.quad_form(x, P) + 0.2 * cp.norm1(x))
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_quad_over_lin_conjugate(self) -> None:
        x = cp.Variable(3)
        primal = cp.Problem(
            cp.Minimize(cp.quad_over_lin(x, 2.0) + 0.3 * cp.norm1(x))
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_kl_div_conjugate(self) -> None:
        x = cp.Variable(3, nonneg=True)
        c = np.array([1.0, 2.0, 0.5])
        primal = cp.Problem(
            cp.Minimize(cp.sum(cp.kl_div(x, c)) + 0.1 * cp.sum(x)),
            [cp.sum(x) <= 5],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_pnorm_p2_conjugate(self) -> None:
        x = cp.Variable(3)
        primal = cp.Problem(
            cp.Minimize(cp.pnorm(x, 2) + 0.1 * cp.norm1(x))
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_pnorm_p3_conjugate(self) -> None:
        x = cp.Variable(3)
        primal = cp.Problem(
            cp.Minimize(cp.pnorm(x, 3) + 0.2 * cp.sum_squares(x)),
            [cp.sum(x) == 1],
        )
        primal.solve(solver=cp.SCS, eps=1e-6)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.SCS, eps=1e-6)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=4)

    def test_norm_inf_conjugate(self) -> None:
        x = cp.Variable(3)
        primal = cp.Problem(
            cp.Minimize(cp.norm_inf(x) + 0.5 * cp.sum_squares(x))
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_sum_largest_conjugate(self) -> None:
        x = cp.Variable(4)
        primal = cp.Problem(
            cp.Minimize(
                cp.sum_largest(x, 2) + 0.1 * cp.sum_squares(x)
            ),
            [cp.sum(x) == 1],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_entr_negative_conjugate(self) -> None:
        """Test -entr(x) = x*log(x) via concave objective."""
        x = cp.Variable(3, nonneg=True)
        primal = cp.Problem(
            cp.Minimize(-cp.sum(cp.entr(x)) + 0.2 * cp.norm1(x)),
            [cp.sum(x) == 1],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_log_negative_conjugate(self) -> None:
        """Test -log(x) as convex term."""
        x = cp.Variable(3, pos=True)
        primal = cp.Problem(
            cp.Minimize(-cp.sum(cp.log(x)) + 0.5 * cp.sum(x)),
            [cp.sum(x) <= 10],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    # ------------------------------------------------------------------
    # Constraint / cone tests
    # ------------------------------------------------------------------

    def test_soc_constraint(self) -> None:
        x = cp.Variable(3)
        t = cp.Variable()
        primal = cp.Problem(
            cp.Minimize(t + 0.1 * cp.sum(x)),
            [cp.SOC(t, x), cp.sum(x) >= 1],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_psd_constraint(self) -> None:
        X = cp.Variable((2, 2), symmetric=True)
        primal = cp.Problem(
            cp.Minimize(cp.trace(X)),
            [X >> np.eye(2)],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    # ------------------------------------------------------------------
    # Spectral / matrix atom tests
    # ------------------------------------------------------------------

    def test_lambda_max_conjugate(self) -> None:
        X = cp.Variable((3, 3), symmetric=True)
        primal = cp.Problem(
            cp.Minimize(cp.lambda_max(X) + 0.1 * cp.norm(X, "fro")),
            [cp.trace(X) == 1],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=4)

    def test_sigma_max_conjugate(self) -> None:
        X = cp.Variable((2, 3))
        primal = cp.Problem(
            cp.Minimize(cp.sigma_max(X)),
            [X[0, 0] == 1, cp.sum(X) == 2],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=4)

    def test_norm_nuc_conjugate(self) -> None:
        X = cp.Variable((2, 3))
        primal = cp.Problem(
            cp.Minimize(cp.normNuc(X)),
            [X[0, 0] == 1, cp.sum(X) == 2],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=4)

    def test_log_det_negative_conjugate(self) -> None:
        X = cp.Variable((3, 3), symmetric=True)
        primal = cp.Problem(
            cp.Minimize(-cp.log_det(X) + cp.trace(X)),
            [X >> 0.1 * np.eye(3)],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=4)

    def test_matrix_frac_conjugate(self) -> None:
        x = cp.Variable(2)
        P = np.array([[2.0, 0.5], [0.5, 1.0]])
        primal = cp.Problem(
            cp.Minimize(cp.matrix_frac(x, P) + 0.1 * cp.norm1(x)),
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=5)

    def test_lambda_sum_largest_conjugate(self) -> None:
        X = cp.Variable((3, 3), symmetric=True)
        primal = cp.Problem(
            cp.Minimize(
                cp.lambda_sum_largest(X, 2) + 0.05 * cp.norm(X, "fro")
            ),
            [cp.trace(X) == 1],
        )
        primal.solve(solver=cp.CLARABEL)
        dual = fenchel_dual(primal)
        dual.solve(solver=cp.CLARABEL)
        self.assertEqual(primal.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(primal.value, dual.value, places=4)

    # ------------------------------------------------------------------
    # Status inversion tests
    # ------------------------------------------------------------------

    def test_invert_infeasible_to_unbounded(self) -> None:
        """If dual is infeasible, primal should be unbounded."""
        red = FenchelDual()
        sol = Solution("infeasible", None, {}, {}, {})
        inv = {"fenchel": {
            "inverse_data": None,
            "stationarity_id": None,
            "primal_var_ids": (),
            "flipped": False,
        }}
        result = red.invert(sol, inv)
        self.assertEqual(result.status, "unbounded")

    def test_invert_unbounded_to_infeasible(self) -> None:
        """If dual is unbounded, primal should be infeasible."""
        red = FenchelDual()
        sol = Solution("unbounded", None, {}, {}, {})
        inv = {"fenchel": {
            "inverse_data": None,
            "stationarity_id": None,
            "primal_var_ids": (),
            "flipped": False,
        }}
        result = red.invert(sol, inv)
        self.assertEqual(result.status, "infeasible")


class TestFenchelDualSuppFuncOracle(BaseTest):
    """Validate conjugate implementations against support-function epigraph oracle.

    For every atom that provides a closed-form conjugate(), we compare its
    value to the ground-truth obtained by formulating the conjugate via a
    support function over the epigraph.  This catches subtle sign, scaling,
    or domain errors that end-to-end primal-dual tests might miss due to
    symmetry or zero optima.
    """

    SCS_EPS = 1e-7
    SCS_ITERS = 40000

    @staticmethod
    def _as_expr(expr):
        return expr if isinstance(expr, cp.Expression) else cp.Constant(expr)

    @staticmethod
    def _shape_of(value):
        shape = np.asarray(value).shape
        return tuple(shape)

    @staticmethod
    def _make_var(shape):
        return cp.Variable() if shape == tuple() else cp.Variable(shape)

    def _scaled_atom_sum(self, atom_expr, perspective_scale=1):
        atom_expr = self._as_expr(atom_expr)
        scale = self._as_expr(perspective_scale)
        if atom_expr.is_scalar():
            return scale * atom_expr
        if scale.is_scalar():
            return scale * cp.sum(atom_expr)
        return cp.sum(cp.multiply(scale, atom_expr))

    # -- single-argument atom helpers --

    def _atom_conjugate_value(
        self, atom_fn, x_shape, y_val, perspective_scale=1, negative=False,
    ):
        y_shape = self._shape_of(y_val)
        y = self._make_var(y_shape)
        x = self._make_var(x_shape)
        atom = atom_fn(x)
        if negative:
            conj_expr, conj_con = atom.negative_conjugate(
                y, perspective_scale=perspective_scale,
            )
        else:
            conj_expr, conj_con = atom.conjugate(
                y, perspective_scale=perspective_scale,
            )
        conj_expr = self._as_expr(conj_expr)
        objective = conj_expr if conj_expr.is_scalar() else cp.sum(conj_expr)
        prob = cp.Problem(
            cp.Minimize(objective), conj_con + [y == y_val],
        )
        prob.solve(solver=cp.SCS, eps=self.SCS_EPS, max_iters=self.SCS_ITERS)
        self.assertIn(prob.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        return prob.value

    def _suppfunc_epigraph_oracle_value(
        self, atom_fn, x_shape, y_val, perspective_scale=1, negative=False,
    ):
        n = int(np.prod(x_shape))
        z = cp.Variable(n + 1)
        x = (
            cp.reshape(z[:-1], x_shape, order="F")
            if x_shape != tuple()
            else z[0]
        )
        t = z[-1]
        f_expr = atom_fn(x)
        if negative:
            f_expr = -f_expr
        f_scalar = self._scaled_atom_sum(f_expr, perspective_scale)
        sigma = cp.suppfunc(z, [f_scalar <= t])

        y_aug = np.concatenate(
            [np.asarray(y_val).reshape(-1, order="F"), np.array([-1.0])],
        )
        prob = cp.Problem(cp.Minimize(sigma(y_aug)))
        prob.solve(solver=cp.SCS, eps=self.SCS_EPS, max_iters=self.SCS_ITERS)
        self.assertIn(prob.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        return prob.value

    def _assert_atom_matches_suppfunc_oracle(
        self,
        atom_fn,
        x_shape,
        y_val,
        perspective_scale=1,
        negative=False,
        places=5,
    ):
        conj_val = self._atom_conjugate_value(
            atom_fn, x_shape, y_val,
            perspective_scale=perspective_scale, negative=negative,
        )
        oracle_val = self._suppfunc_epigraph_oracle_value(
            atom_fn, x_shape, y_val,
            perspective_scale=perspective_scale, negative=negative,
        )
        self.assertAlmostEqual(conj_val, oracle_val, places=places)

    # -- multi-argument term helpers --

    def _term_conjugate_value(
        self, atom_fn, x_shapes, y_vals,
        perspective_scale=1, negative=False,
    ):
        x_vars = [self._make_var(s) for s in x_shapes]
        y_vars = [self._make_var(self._shape_of(v)) for v in y_vals]
        atom = atom_fn(*x_vars)
        indices = tuple(range(len(x_vars)))
        if negative:
            conj_expr, conj_con = atom.negative_conjugate_term(
                indices, tuple(y_vars),
                perspective_scale=perspective_scale,
            )
        else:
            conj_expr, conj_con = atom.conjugate_term(
                indices, tuple(y_vars),
                perspective_scale=perspective_scale,
            )
        conj_expr = self._as_expr(conj_expr)
        objective = conj_expr if conj_expr.is_scalar() else cp.sum(conj_expr)
        eq_con = [var == val for var, val in zip(y_vars, y_vals)]
        prob = cp.Problem(cp.Minimize(objective), conj_con + eq_con)
        prob.solve(solver=cp.SCS, eps=self.SCS_EPS, max_iters=self.SCS_ITERS)
        self.assertIn(prob.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        return prob.value

    def _suppfunc_term_oracle_value(
        self, atom_fn, x_shapes, y_vals,
        perspective_scale=1, negative=False,
    ):
        total_n = int(sum(np.prod(s) for s in x_shapes))
        z = cp.Variable(total_n + 1)
        t = z[-1]
        x_vars, cursor = [], 0
        for shape in x_shapes:
            size = int(np.prod(shape))
            x_vars.append(
                cp.reshape(z[cursor:cursor + size], shape, order="F"),
            )
            cursor += size
        f_expr = atom_fn(*x_vars)
        if negative:
            f_expr = -f_expr
        f_scalar = self._scaled_atom_sum(f_expr, perspective_scale)
        sigma = cp.suppfunc(z, [f_scalar <= t])

        y_parts = [np.asarray(v).reshape(-1, order="F") for v in y_vals]
        y_aug = np.concatenate(y_parts + [np.array([-1.0])])
        prob = cp.Problem(cp.Minimize(sigma(y_aug)))
        prob.solve(solver=cp.SCS, eps=self.SCS_EPS, max_iters=self.SCS_ITERS)
        self.assertIn(prob.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        return prob.value

    def _assert_term_matches_suppfunc_oracle(
        self,
        atom_fn,
        x_shapes,
        y_vals,
        perspective_scale=1,
        negative=False,
        places=5,
    ):
        conj_val = self._term_conjugate_value(
            atom_fn, x_shapes, y_vals,
            perspective_scale=perspective_scale, negative=negative,
        )
        oracle_val = self._suppfunc_term_oracle_value(
            atom_fn, x_shapes, y_vals,
            perspective_scale=perspective_scale, negative=negative,
        )
        self.assertAlmostEqual(conj_val, oracle_val, places=places)

    # ------------------------------------------------------------------
    # Convex atom conjugates
    # ------------------------------------------------------------------

    def test_convex_atom_conjugates_match_suppfunc_oracle(self) -> None:
        P_dense = np.array([[2.0, 0.3], [0.3, 1.5]])
        P_singular = np.array([[1.0, 0.0], [0.0, 0.0]])
        P_mf = np.diag([2.0, 1.0, 3.0])
        # (name, atom_fn, x_shape, y_val, scale, places)
        cases = [
            # --- norms ---
            ("norm1", lambda x: cp.norm1(x),
             (3,), np.array([0.2, -0.5, 0.7]), 1, 6),
            ("norm1_axis0", lambda x: cp.norm1(x, axis=0),
             (2, 3), np.array([[0.2, -0.6, 0.5], [0.1, 0.3, -0.4]]), 1, 6),
            ("norm_inf", lambda x: cp.norm_inf(x),
             (3,), np.array([0.2, -0.3, 0.1]), 1, 6),
            ("norm_inf_axis0", lambda x: cp.norm_inf(x, axis=0),
             (2, 3), np.array([[0.2, -0.4, 0.3], [0.1, 0.2, -0.2]]), 1, 6),
            ("pnorm_p2", lambda x: cp.pnorm(x, 2, approx=False),
             (3,), np.array([0.2, -0.1, 0.3]), 1, 6),
            ("pnorm_p1p5", lambda x: cp.pnorm(x, 1.5),
             (3,), np.array([0.2, -0.3, 0.1]), 1, 5),
            ("pnorm_axis0", lambda x: cp.pnorm(x, 2, axis=0, approx=False),
             (2, 3), np.array([[0.2, -0.3, 0.1], [0.4, 0.2, -0.1]]), 1, 6),
            # --- max / sum_largest ---
            ("max", lambda x: cp.max(x),
             (3,), np.array([0.2, 0.3, 0.5]), 1, 6),
            ("max_axis0", lambda x: cp.max(x, axis=0),
             (2, 3), np.array([[0.4, 0.7, 0.2], [0.6, 0.3, 0.8]]), 1, 6),
            ("sum_largest_k2", lambda x: cp.sum_largest(x, 2),
             (4,), np.array([0.7, 0.6, 0.5, 0.2]), 1, 6),
            # --- abs ---
            ("abs", lambda x: cp.abs(x),
             (3,), np.array([0.2, -0.5, 0.7]), 1, 6),
            # --- exp, logistic, log_sum_exp ---
            ("exp", lambda x: cp.exp(x),
             (3,), np.array([0.7, 0.8, 0.5]), 1, 6),
            ("exp_scale_scalar", lambda x: cp.exp(x),
             (3,), np.array([0.7, 0.8, 0.5]), 2.0, 6),
            ("exp_scale_vector", lambda x: cp.exp(x),
             (3,), np.array([0.7, 0.2, 0.5]),
             np.array([2.0, 0.5, 1.5]), 6),
            ("logistic", lambda x: cp.logistic(x),
             (3,), np.array([0.2, 0.4, 0.5]), 1, 6),
            ("logistic_scale_vector", lambda x: cp.logistic(x),
             (3,), np.array([0.4, 0.2, 0.3]),
             np.array([2.0, 0.5, 1.5]), 6),
            ("log_sum_exp", lambda x: cp.log_sum_exp(x),
             (3,), np.array([0.2, 0.3, 0.5]), 1, 6),
            ("log_sum_exp_axis0_scale",
             lambda x: cp.log_sum_exp(x, axis=0),
             (2, 3), np.array([[0.4, 0.8, 0.1], [0.6, 1.2, 0.4]]),
             np.array([1.0, 2.0, 0.5]), 5),
            # --- entropy-family ---
            ("huber", lambda x: cp.huber(x, M=1.5),
             (3,), np.array([0.5, -1.0, 2.0]), 1, 6),
            ("huber_scale_vector", lambda x: cp.huber(x, M=1.5),
             (3,), np.array([1.0, 0.4, -2.0]),
             np.array([2.0, 0.5, 1.5]), 5),
            ("rel_entr_c_scalar", lambda x: cp.rel_entr(x, 1.7),
             (3,), np.array([0.2, -0.3, 0.5]), 1, 5),
            ("rel_entr_c_vector_scale",
             lambda x: cp.rel_entr(x, np.array([2.0, 0.5, 1.5])),
             (3,), np.array([0.2, -0.3, 0.5]),
             np.array([1.2, 0.7, 0.9]), 5),
            ("kl_div_c_scalar", lambda x: cp.kl_div(x, 1.3),
             (3,), np.array([0.1, -0.2, 0.4]), 1, 5),
            ("kl_div_c_vector_scale",
             lambda x: cp.kl_div(x, np.array([1.2, 0.7, 1.1])),
             (3,), np.array([0.1, -0.2, 0.4]),
             np.array([1.1, 0.8, 0.9]), 5),
            # --- power ---
            ("power_p2", lambda x: cp.power(x, 2),
             (3,), np.array([-1.2, 0.4, -0.6]), 1, 6),
            ("power_p1p5", lambda x: cp.power(x, 1.5),
             (3,), np.array([0.8, -0.2, 0.5]), 1, 5),
            # --- quadratics ---
            ("quad_over_lin", lambda x: cp.quad_over_lin(x, 2.0),
             (3,), np.array([1.2, -0.7, 0.3]), 1, 6),
            ("quad_over_lin_scale", lambda x: cp.quad_over_lin(x, 2.0),
             (3,), np.array([1.2, -0.7, 0.3]), 2.0, 6),
            ("quad_over_lin_axis0",
             lambda x: cp.quad_over_lin(x, 2.0, axis=0),
             (2, 3),
             np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]]), 1, 6),
            ("quad_over_lin_axis0_scale",
             lambda x: cp.quad_over_lin(x, 2.0, axis=0),
             (2, 3),
             np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]]),
             np.array([1.0, 2.0, 4.0]), 6),
            ("quad_form_psd", lambda x: cp.quad_form(x, P_dense),
             (2,), np.array([0.6, -0.4]), 1, 6),
            ("quad_form_psd_scale", lambda x: cp.quad_form(x, P_dense),
             (2,), np.array([0.6, -0.4]), 2.0, 5),
            ("quad_form_singular",
             lambda x: cp.quad_form(x, P_singular),
             (2,), np.array([0.8, 0.0]), 1, 6),
            ("quad_form_zero",
             lambda x: cp.quad_form(x, np.zeros((2, 2))),
             (2,), np.array([0.0, 0.0]), 1, 6),
            ("matrix_frac_vector",
             lambda x: cp.matrix_frac(x, cp.Constant(P_mf)),
             (3,), np.array([1.0, -2.0, 0.5]), 1, 5),
            ("matrix_frac_vector_scale",
             lambda x: cp.matrix_frac(x, cp.Constant(P_mf)),
             (3,), np.array([1.0, -2.0, 0.5]), 2.0, 5),
            ("matrix_frac_matrix",
             lambda x: cp.matrix_frac(x, cp.Constant(P_mf)),
             (3, 2),
             np.array([[1.0, 0.3], [-2.0, 0.2], [0.5, -1.0]]), 1, 5),
            # --- spectral ---
            ("lambda_max", lambda x: cp.lambda_max(x),
             (2, 2), np.array([[0.5, 0.0], [0.0, 0.5]]), 1, 6),
            ("lambda_max_scale", lambda x: cp.lambda_max(x),
             (2, 2), np.array([[1.0, 0.0], [0.0, 1.0]]), 2.0, 6),
            ("lambda_sum_largest",
             lambda x: cp.lambda_sum_largest(x, 2),
             (3, 3), np.diag([0.9, 0.8, 0.3]), 1, 6),
            ("sigma_max", lambda x: cp.sigma_max(x),
             (2, 2), np.array([[0.3, 0.1], [0.0, 0.2]]), 1, 6),
            ("norm_nuc", lambda x: cp.normNuc(x),
             (2, 2), np.array([[0.6, 0.0], [0.0, -0.4]]), 1, 6),
            ("norm_nuc_scale", lambda x: cp.normNuc(x),
             (2, 2), np.array([[1.2, 0.0], [0.0, -0.8]]), 2.0, 6),
        ]
        for name, atom_fn, x_shape, y_val, scale, places in cases:
            with self.subTest(case=name):
                self._assert_atom_matches_suppfunc_oracle(
                    atom_fn, x_shape, y_val,
                    perspective_scale=scale, places=places,
                )

    # ------------------------------------------------------------------
    # Concave atom negative-conjugates
    # ------------------------------------------------------------------

    def test_concave_atom_negative_conjugates_match_suppfunc_oracle(
        self,
    ) -> None:
        P_nsd = -np.diag([2.0, 1.0])
        cases = [
            ("log_neg_conj", lambda x: cp.log(x),
             (3,), np.array([-0.4, -0.7, -0.2]), 1, 6),
            ("log_neg_conj_scale", lambda x: cp.log(x),
             (3,), np.array([-0.8, -1.4, -0.4]), 2.0, 6),
            ("entr_neg_conj", lambda x: cp.entr(x),
             (3,), np.array([0.2, -0.5, 0.8]), 1, 5),
            ("entr_neg_conj_scale", lambda x: cp.entr(x),
             (3,), np.array([0.2, -0.5, 0.8]),
             np.array([2.0, 0.5, 1.5]), 5),
            ("quad_form_neg_conj",
             lambda x: cp.quad_form(x, P_nsd),
             (2,), np.array([0.6, -0.3]), 1, 6),
            ("quad_form_neg_conj_scale",
             lambda x: cp.quad_form(x, P_nsd),
             (2,), np.array([0.6, -0.3]), 2.0, 5),
            ("log_det_neg_conj", lambda x: cp.log_det(x),
             (2, 2), -np.eye(2), 1, 5),
            ("log_det_neg_conj_scale", lambda x: cp.log_det(x),
             (2, 2), -2.0 * np.eye(2), 2.0, 5),
            ("power_p0p5_neg_conj",
             lambda x: cp.power(x, 0.5),
             (3,), np.array([-0.5, -1.0, -0.3]), 1, 5),
            ("power_p0p5_neg_conj_scale",
             lambda x: cp.power(x, 0.5),
             (3,), np.array([-1.0, -2.0, -0.6]), 2.0, 5),
            ("power_p0p33_neg_conj",
             lambda x: cp.power(x, 1.0 / 3.0),
             (2,), np.array([-0.4, -0.8]), 1, 5),
        ]
        for name, atom_fn, x_shape, y_val, scale, places in cases:
            with self.subTest(case=name):
                self._assert_atom_matches_suppfunc_oracle(
                    atom_fn, x_shape, y_val,
                    perspective_scale=scale, negative=True,
                    places=places,
                )

    # ------------------------------------------------------------------
    # Multi-argument term conjugates
    # ------------------------------------------------------------------

    def test_multi_argument_term_conjugates_match_suppfunc_oracle(
        self,
    ) -> None:
        term_cases = [
            (
                "rel_entr_full_term",
                lambda x1, x2: cp.rel_entr(x1, x2),
                ((2,), (2,)),
                (np.array([0.2, -0.1]), np.array([-0.7, -0.9])),
                1.0, False, 5,
            ),
            (
                "kl_div_full_term",
                lambda x1, x2: cp.kl_div(x1, x2),
                ((2,), (2,)),
                (np.array([0.2, -0.1]), np.array([-0.5, -0.2])),
                1.0, False, 5,
            ),
            (
                "maximum_full_term",
                lambda x1, x2: cp.maximum(x1, x2),
                ((2,), (2,)),
                (np.array([1.2, 0.8]), np.array([0.8, 1.2])),
                2.0, False, 6,
            ),
            (
                "minimum_full_negative_term",
                lambda x1, x2: cp.minimum(x1, x2),
                ((2,), (2,)),
                (np.array([-1.2, -0.8]), np.array([-0.8, -1.2])),
                2.0, True, 6,
            ),
        ]
        for name, atom_fn, x_shapes, y_vals, scale, neg, places in (
            term_cases
        ):
            with self.subTest(case=name):
                self._assert_term_matches_suppfunc_oracle(
                    atom_fn, x_shapes, y_vals,
                    perspective_scale=scale, negative=neg,
                    places=places,
                )

    # ------------------------------------------------------------------
    # Parameter-matrix quad_form
    # ------------------------------------------------------------------

    def test_quad_form_parameter_conjugate_matches_constant_oracle(
        self,
    ) -> None:
        P_param = cp.Parameter(
            (2, 2), PSD=True,
            value=np.array([[1.5, 0.2], [0.2, 1.0]]),
        )
        y_val = np.array([0.7, -0.3])
        conj_val = self._atom_conjugate_value(
            lambda x: cp.quad_form(x, P_param), (2,), y_val,
        )
        oracle_val = self._suppfunc_epigraph_oracle_value(
            lambda x: cp.quad_form(x, P_param.value), (2,), y_val,
        )
        self.assertAlmostEqual(conj_val, oracle_val, places=5)
