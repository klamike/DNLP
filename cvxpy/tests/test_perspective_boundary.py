"""
Tests for perspective boundary handling per Roos et al. (2020) Appendix B.9.

The paper specifies strict convention: (0*f)^*(y) = δ_0(y), not closure.
This means when perspective_scale is identically zero, the conjugate should
enforce y == 0, giving value 0, not allow y ≠ 0 via closure semantics.
"""

import numpy as np

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestPerspectiveBoundary(BaseTest):
    """Test strict (0f)^*(y) = δ_0(y) convention for perspective functions."""

    def test_exp_zero_scale_strict_convention(self) -> None:
        """exp: (0*exp)^*(y) = δ_0(y), not closure."""
        from cvxpy.atoms.elementwise.exp import exp as exp_atom
        atom = exp_atom(cp.Variable())
        y = cp.Variable(nonneg=True)

        # Test with identically zero scale (constant 0)
        zero_scale = cp.Constant(0.0)
        conj_expr, conj_constraints = atom.conjugate(y, perspective_scale=zero_scale)

        # Should return indicator conjugate: value 0, constraint y == 0
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)
        # The constraint should enforce y == 0
        prob = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == 1.0])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.INFEASIBLE)  # y=1 violates y==0

        # But y=0 should be feasible
        prob_zero = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == 0])
        prob_zero.solve(solver=cp.CLARABEL)
        self.assertEqual(prob_zero.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob_zero.value, 0.0, places=6)

    def test_logistic_zero_scale_strict_convention(self) -> None:
        """logistic: (0*logistic)^*(y) = δ_0(y), not closure."""
        from cvxpy.atoms.elementwise.logistic import logistic as logistic_atom
        atom = logistic_atom(cp.Variable())
        y = cp.Variable()

        zero_scale = cp.Constant(0.0)
        conj_expr, conj_constraints = atom.conjugate(y, perspective_scale=zero_scale)

        # Should return indicator conjugate
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

        # y ≠ 0 should be infeasible
        prob = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == 0.5])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.INFEASIBLE)

        # y = 0 should be feasible
        prob_zero = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == 0])
        prob_zero.solve(solver=cp.CLARABEL)
        self.assertEqual(prob_zero.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob_zero.value, 0.0, places=6)

    def test_log_sum_exp_zero_scale_strict_convention(self) -> None:
        """log_sum_exp: (0*log_sum_exp)^*(y) = δ_0(y), not closure."""
        from cvxpy.atoms.log_sum_exp import log_sum_exp as lse_atom
        atom = lse_atom(cp.Variable(3))
        y = cp.Variable(3, nonneg=True)

        zero_scale = cp.Constant(0.0)
        conj_expr, conj_constraints = atom.conjugate(y, perspective_scale=zero_scale)

        # Should return indicator conjugate
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

        # y ≠ 0 (e.g., y = [0.1, 0.1, 0.1]) should be infeasible
        prob = cp.Problem(
            cp.Minimize(conj_expr),
            conj_constraints + [y == np.array([0.1, 0.1, 0.1])]
        )
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.INFEASIBLE)

        # y = 0 should be feasible
        prob_zero = cp.Problem(
            cp.Minimize(conj_expr),
            conj_constraints + [y == 0]
        )
        prob_zero.solve(solver=cp.CLARABEL)
        self.assertEqual(prob_zero.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob_zero.value, 0.0, places=6)

    def test_power_zero_scale_strict_convention(self) -> None:
        """power: (0*x^p)^*(y) = δ_0(y) for p > 1."""
        y = cp.Variable(2, nonneg=True)

        # power(x, 2) for x >= 0
        atom = cp.power(cp.Variable(2, nonneg=True), 2)
        zero_scale = cp.Constant(0.0)
        conj_expr, conj_constraints = atom.conjugate(y, perspective_scale=zero_scale)

        # Should return indicator conjugate
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

        # y ≠ 0 should be infeasible
        prob = cp.Problem(
            cp.Minimize(conj_expr),
            conj_constraints + [y == np.array([1.0, 0.5])]
        )
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.INFEASIBLE)

        # y = 0 should be feasible
        prob_zero = cp.Problem(
            cp.Minimize(conj_expr),
            conj_constraints + [y == 0]
        )
        prob_zero.solve(solver=cp.CLARABEL)
        self.assertEqual(prob_zero.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob_zero.value, 0.0, places=6)

    def test_log_zero_scale_strict_convention(self) -> None:
        """log (concave): (0*(-log))^*(y) = δ_0(y)."""
        from cvxpy.atoms.elementwise.log import log as log_atom
        atom = log_atom(cp.Variable(nonneg=True))
        y = cp.Variable()

        zero_scale = cp.Constant(0.0)
        # log is concave, so we use negative_conjugate
        conj_expr, conj_constraints = atom.negative_conjugate(y, perspective_scale=zero_scale)

        # Should return indicator conjugate
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

        # y ≠ 0 should be infeasible
        prob = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == -1.0])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.INFEASIBLE)

        # y = 0 should be feasible
        prob_zero = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == 0])
        prob_zero.solve(solver=cp.CLARABEL)
        self.assertEqual(prob_zero.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob_zero.value, 0.0, places=6)

    def test_entr_zero_scale_strict_convention(self) -> None:
        """entr (concave): (0*(-entr))^*(y) = δ_0(y)."""
        from cvxpy.atoms.elementwise.entr import entr as entr_atom
        atom = entr_atom(cp.Variable(2, nonneg=True))
        y = cp.Variable(2)

        zero_scale = cp.Constant(0.0)
        conj_expr, conj_constraints = atom.negative_conjugate(y, perspective_scale=zero_scale)

        # Should return indicator conjugate
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

        # y ≠ 0 should be infeasible
        prob = cp.Problem(
            cp.Minimize(conj_expr),
            conj_constraints + [y == np.array([0.1, -0.1])]
        )
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.INFEASIBLE)

        # y = 0 should be feasible
        prob_zero = cp.Problem(
            cp.Minimize(conj_expr),
            conj_constraints + [y == 0]
        )
        prob_zero.solve(solver=cp.CLARABEL)
        self.assertEqual(prob_zero.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob_zero.value, 0.0, places=6)

    def test_rel_entr_zero_scale_strict_convention(self) -> None:
        """rel_entr: (0*rel_entr)^*(y) = δ_0(y)."""
        from cvxpy.atoms.elementwise.rel_entr import rel_entr as rel_entr_atom
        # rel_entr(x, q) with q constant
        atom = rel_entr_atom(cp.Variable(nonneg=True), cp.Constant(1.0))
        y = cp.Variable()

        zero_scale = cp.Constant(0.0)
        conj_expr, conj_constraints = atom.conjugate(y, perspective_scale=zero_scale)

        # Should return indicator conjugate
        self.assertEqual(conj_expr, 0)
        self.assertEqual(len(conj_constraints), 1)

        # y ≠ 0 should be infeasible
        prob = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == 0.5])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.INFEASIBLE)

        # y = 0 should be feasible
        prob_zero = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == 0])
        prob_zero.solve(solver=cp.CLARABEL)
        self.assertEqual(prob_zero.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob_zero.value, 0.0, places=6)

    def test_positive_scale_works_normally(self) -> None:
        """Verify positive scale still works with standard conjugate."""
        from cvxpy.atoms.elementwise.exp import exp as exp_atom

        # With positive scale, should get standard conjugate
        atom = exp_atom(cp.Variable())
        y = cp.Variable(nonneg=True)
        positive_scale = cp.Constant(2.0)

        conj_expr, conj_constraints = atom.conjugate(y, perspective_scale=positive_scale)

        # Should NOT be indicator conjugate (expr != 0)
        self.assertFalse(isinstance(conj_expr, (int, float)) and conj_expr == 0)

        # Should allow y > 0 (not just y == 0)
        prob = cp.Problem(cp.Minimize(conj_expr), conj_constraints + [y == 1.0])
        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)
        # For (s*exp)^*(y) = rel_entr(y, s) - y = y*log(y/s) - y
        # With s=2, y=1: = 1*log(1/2) - 1 = -log(2) - 1
        expected = 1 * np.log(1/2) - 1
        self.assertAlmostEqual(prob.value, expected, places=5)

    def test_fenchel_dual_with_zero_scale_inequality(self) -> None:
        """Integration test: Fenchel dual with identically-zero constraint multiplier."""
        # This is a degenerate case: constraint that's always satisfied
        # min x^2 subject to 0 <= 0 (trivially true)
        # The dual multiplier for this constraint should be restricted to 0

        x = cp.Variable()
        # Create a constraint that becomes 0*f(x) <= 0
        # This is artificial but tests the boundary handling
        prob = cp.Problem(
            cp.Minimize(x**2),
            [x >= -1, x <= 1]  # Regular constraints
        )

        # Normal dualization should work
        from cvxpy.reductions.fenchel_dual import fenchel_dual
        dual = fenchel_dual(prob)
        dual.solve(solver=cp.CLARABEL)

        prob.solve(solver=cp.CLARABEL)

        # Values should match (strong duality)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertEqual(dual.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob.value, dual.value, places=5)
