"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for Fenchel dualization adjoint methods and symbolic adjoint framework.
"""

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.fenchel_dual import fenchel_dual
from cvxpy.tests.base_test import BaseTest


class TestSymbolicAdjoint(BaseTest):
    """Test the symbolic adjoint framework in fenchel_dual.py"""

    def test_inner_product_real(self) -> None:
        """Test _inner_product for real vectors"""
        from cvxpy.reductions.fenchel_dual import _inner_product

        y = cp.Variable(3)
        z = cp.Constant([1.0, 2.0, 3.0])
        result = _inner_product(y, z)

        # Should be sum(y * z)
        self.assertEqual(result.shape, ())
        y.value = np.array([4.0, 5.0, 6.0])
        expected = np.dot([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(result.value, expected)

    def test_inner_product_complex(self) -> None:
        """Test _inner_product for complex vectors"""
        from cvxpy.reductions.fenchel_dual import _inner_product

        y = cp.Variable(2, complex=True)
        z = cp.Constant([1.0 + 1.0j, 2.0 - 1.0j])
        result = _inner_product(y, z)

        # Should be Re(sum(conj(y) * z))
        self.assertEqual(result.shape, ())
        y.value = np.array([1.0 + 0.0j, 0.0 + 1.0j])
        # conj([1+0j, 0+1j]) * [1+1j, 2-1j] = [1+1j, -1+2j]
        # sum = 0+3j, Re = 0
        self.assertAlmostEqual(result.value, 0.0)

    def test_merge_adj(self) -> None:
        """Test _merge_adj combines adjoint dictionaries"""
        from cvxpy.reductions.fenchel_dual import _merge_adj
        from cvxpy.expressions.constants import Constant

        x1 = cp.Variable(2)
        x2 = cp.Variable(3)

        d1 = {x1.id: Constant([1.0, 2.0])}
        d2 = {x2.id: Constant([3.0, 4.0, 5.0])}
        merged = _merge_adj(d1, d2)

        self.assertEqual(len(merged), 2)
        self.assertIn(x1.id, merged)
        self.assertIn(x2.id, merged)

    def test_merge_adj_overlapping(self) -> None:
        """Test _merge_adj sums when variable IDs overlap"""
        from cvxpy.reductions.fenchel_dual import _merge_adj
        from cvxpy.expressions.constants import Constant

        x = cp.Variable(2)

        d1 = {x.id: Constant([1.0, 2.0])}
        d2 = {x.id: Constant([3.0, 4.0])}
        merged = _merge_adj(d1, d2)

        self.assertEqual(len(merged), 1)
        # Should be [1+3, 2+4] = [4, 6]
        self.assertTrue(np.allclose(merged[x.id].value, [4.0, 6.0]))

    def test_symbolic_adjoint_variable(self) -> None:
        """Test symbolic adjoint for Variable leaf"""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint

        x = cp.Variable(3)
        y_var = cp.Variable(3)

        adj_dict, const_term = _symbolic_adjoint(x, y_var)

        self.assertEqual(len(adj_dict), 1)
        self.assertIn(x.id, adj_dict)
        # Adjoint should be y_var itself
        self.assertEqual(adj_dict[x.id].id, y_var.id)
        # Constant term should be zero
        self.assertAlmostEqual(const_term.value, 0.0)

    def test_symbolic_adjoint_constant(self) -> None:
        """Test symbolic adjoint for Constant leaf"""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint

        c = cp.Constant([1.0, 2.0, 3.0])
        y_var = cp.Variable(3)

        adj_dict, const_term = _symbolic_adjoint(c, y_var)

        self.assertEqual(len(adj_dict), 0)  # No variables
        # Constant term should be <y_var, c> = sum(y_var * c)
        y_var.value = np.array([4.0, 5.0, 6.0])
        expected = np.dot([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(const_term.value, expected)

    def test_symbolic_adjoint_add(self) -> None:
        """Test symbolic adjoint for AddExpression"""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint

        x1 = cp.Variable(2)
        x2 = cp.Variable(2)
        expr = x1 + x2 + cp.Constant([1.0, 2.0])
        y_var = cp.Variable(2)

        adj_dict, const_term = _symbolic_adjoint(expr, y_var)

        # Should have adjoints for both x1 and x2
        self.assertEqual(len(adj_dict), 2)
        self.assertIn(x1.id, adj_dict)
        self.assertIn(x2.id, adj_dict)

    def test_symbolic_adjoint_neg(self) -> None:
        """Test symbolic adjoint for NegExpression"""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint

        x = cp.Variable(2)
        expr = -x
        y_var = cp.Variable(2)

        adj_dict, const_term = _symbolic_adjoint(expr, y_var)

        self.assertEqual(len(adj_dict), 1)
        # Adjoint should be -y_var
        y_var.value = np.array([1.0, 2.0])
        self.assertTrue(np.allclose(adj_dict[x.id].value, [-1.0, -2.0]))

    def test_symbolic_adjoint_multiply_constant(self) -> None:
        """Test symbolic adjoint for element-wise multiply with constant"""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint

        x = cp.Variable(3)
        c = cp.Constant([2.0, 3.0, 4.0])
        expr = cp.multiply(c, x)
        y_var = cp.Variable(3)

        adj_dict, const_term = _symbolic_adjoint(expr, y_var)

        self.assertEqual(len(adj_dict), 1)
        # Adjoint should be c * y_var
        y_var.value = np.array([1.0, 1.0, 1.0])
        self.assertTrue(np.allclose(adj_dict[x.id].value, [2.0, 3.0, 4.0]))

    def test_symbolic_adjoint_matmul(self) -> None:
        """Test symbolic adjoint for matrix multiplication"""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint

        A = cp.Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
        x = cp.Variable(2)
        expr = A @ x
        y_var = cp.Variable(2)

        adj_dict, const_term = _symbolic_adjoint(expr, y_var)

        self.assertEqual(len(adj_dict), 1)
        # Adjoint should be A.T @ y_var
        y_var.value = np.array([1.0, 1.0])
        expected = np.array([[1.0, 3.0], [2.0, 4.0]]) @ np.array([1.0, 1.0])
        # The adjoint expression needs to be evaluated
        adj_expr = adj_dict[x.id]
        self.assertTrue(np.allclose(adj_expr.value, expected))

    def test_symbolic_adjoint_sum_full(self) -> None:
        """Test symbolic adjoint for full sum"""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint

        x = cp.Variable((2, 3))
        expr = cp.sum(x)
        y_var = cp.Variable()  # scalar

        adj_dict, const_term = _symbolic_adjoint(expr, y_var)

        self.assertEqual(len(adj_dict), 1)
        # Adjoint should broadcast y_var to shape (2, 3)
        self.assertEqual(adj_dict[x.id].shape, (2, 3))

    def test_symbolic_adjoint_sum_axis(self) -> None:
        """Test symbolic adjoint for axis sum"""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint

        x = cp.Variable((3, 4))
        expr = cp.sum(x, axis=0)
        y_var = cp.Variable(4)

        adj_dict, const_term = _symbolic_adjoint(expr, y_var)

        self.assertEqual(len(adj_dict), 1)
        # Adjoint should broadcast y_var along axis 0
        self.assertEqual(adj_dict[x.id].shape, (3, 4))


class TestWrapsAdjoint(BaseTest):
    """Test adjoint methods for wrap atoms"""

    def test_wrap_identity_adjoint(self) -> None:
        """Test that Wrap.adjoint returns identity"""
        from cvxpy.atoms.affine.wraps import Wrap

        x = cp.Variable(3)
        wrapped = Wrap(x)
        y_var = cp.Variable(3)

        adj = wrapped.adjoint(y_var)

        self.assertEqual(len(adj), 1)
        self.assertEqual(adj[0][0], 0)  # arg index 0
        self.assertEqual(adj[0][1].id, y_var.id)  # identity

    def test_nonneg_wrap_adjoint(self) -> None:
        """Test nonneg_wrap adjoint"""
        from cvxpy.atoms.affine.wraps import nonneg_wrap

        x = cp.Variable(4)
        wrapped = nonneg_wrap(x)
        y_var = cp.Variable(4)

        adj = wrapped.adjoint(y_var)

        self.assertEqual(len(adj), 1)
        self.assertEqual(adj[0][0], 0)
        self.assertEqual(adj[0][1].id, y_var.id)

    def test_psd_wrap_adjoint(self) -> None:
        """Test psd_wrap adjoint"""
        from cvxpy.atoms.affine.wraps import psd_wrap

        X = cp.Variable((3, 3), symmetric=True)
        wrapped = psd_wrap(X)
        Y_var = cp.Variable((3, 3))

        adj = wrapped.adjoint(Y_var)

        self.assertEqual(len(adj), 1)
        self.assertEqual(adj[0][0], 0)
        self.assertEqual(adj[0][1].shape, (3, 3))

    def test_wrap_in_fenchel_dual(self) -> None:
        """Test Fenchel dualization with wrapped variable"""
        x = cp.Variable(3)
        wrapped = cp.atoms.affine.wraps.nonneg_wrap(x)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(wrapped)), [wrapped <= 5])

        try:
            dual_prob = fenchel_dual(prob)
            self.assertIsNotNone(dual_prob)
        except Exception as e:
            self.fail(f"Fenchel dualization failed with wrap: {e}")


class TestConvAdjoint(BaseTest):
    """Test adjoint methods for convolution atoms"""

    def test_convolve_adjoint_basic(self) -> None:
        """Test basic convolve adjoint correctness"""
        from cvxpy.atoms.affine.conv import convolve

        c = cp.Constant([1.0, 2.0, 3.0])
        x = cp.Variable(2)
        y_expr = convolve(c, x)

        # Output should have length 3+2-1=4
        self.assertEqual(y_expr.shape, (4,))

        y_var = cp.Variable(4)
        adj = y_expr.adjoint(y_var)

        self.assertEqual(len(adj), 1)
        self.assertEqual(adj[0][0], 1)  # adjoint w.r.t. arg 1 (x)
        self.assertEqual(adj[0][1].shape, (2,))  # same shape as x

    def test_convolve_adjoint_values(self) -> None:
        """Test convolve adjoint with concrete values"""
        from cvxpy.atoms.affine.conv import convolve

        # Simple test: c=[1, 0], x=[a, b]
        # y = [a, b, 0] (convolution)
        # Adjoint: given y_dual=[y0, y1, y2], x_adj = [y0, y1]
        c = cp.Constant([1.0, 0.0])
        x = cp.Variable(2)
        y_expr = convolve(c, x)
        y_var = cp.Variable(3)

        adj = y_expr.adjoint(y_var)
        adj_expr = adj[0][1]

        y_var.value = np.array([1.0, 2.0, 3.0])
        # Expected: [1, 2] (first two elements of y_var)
        # Actually: Toeplitz(col=[1,0,0], row=[1,0,0]) = [[1,0],[0,1],[0,0]]
        # C.T @ [1,2,3] = [[1,0,0],[0,1,0]] @ [1,2,3] = [1, 2]
        self.assertTrue(np.allclose(adj_expr.value, [1.0, 2.0]))

    def test_convolve_adjoint_nontrivial(self) -> None:
        """Test convolve adjoint with non-trivial kernel"""
        from cvxpy.atoms.affine.conv import convolve

        c = cp.Constant([1.0, 2.0])
        x = cp.Variable(3)
        y_expr = convolve(c, x)
        y_var = cp.Variable(4)

        adj = y_expr.adjoint(y_var)
        adj_expr = adj[0][1]

        # Convolution matrix C:
        # y[0] = 1*x[0]
        # y[1] = 2*x[0] + 1*x[1]
        # y[2] = 2*x[1] + 1*x[2]
        # y[3] = 2*x[2]
        #
        # C = [[1, 0, 0],
        #      [2, 1, 0],
        #      [0, 2, 1],
        #      [0, 0, 2]]
        #
        # C.T = [[1, 2, 0, 0],
        #        [0, 1, 2, 0],
        #        [0, 0, 1, 2]]

        y_var.value = np.array([1.0, 1.0, 1.0, 1.0])
        expected = np.array([1.0 + 2.0, 1.0 + 2.0, 1.0 + 2.0])  # [3, 3, 3]
        self.assertTrue(np.allclose(adj_expr.value, expected))

    def test_conv_adjoint_2d(self) -> None:
        """Test deprecated conv with 2D column vectors"""
        from cvxpy.atoms.affine.conv import conv

        c = cp.Constant(np.array([[1.0], [2.0]]))
        x = cp.Variable((2, 1))
        y_expr = conv(c, x)

        # Output should be (3, 1)
        self.assertEqual(y_expr.shape, (3, 1))

        y_var = cp.Variable((3, 1))
        adj = y_expr.adjoint(y_var)

        self.assertEqual(len(adj), 1)
        self.assertEqual(adj[0][0], 1)
        self.assertEqual(adj[0][1].shape, (2, 1))

    def test_convolve_nonconstant_error(self) -> None:
        """Test that non-constant first arg raises error during construction"""
        from cvxpy.atoms.affine.conv import convolve

        c = cp.Variable(2)  # Variable, not constant
        x = cp.Variable(3)

        # Should raise during construction, not during adjoint()
        with self.assertRaises(ValueError) as cm:
            y_expr = convolve(c, x)

        self.assertIn("constant", str(cm.exception).lower())

    def test_convolve_parameter_without_value_error(self) -> None:
        """Test that Parameter without value raises error"""
        from cvxpy.atoms.affine.conv import convolve

        c = cp.Parameter(2)  # No value assigned
        x = cp.Variable(3)
        y_expr = convolve(c, x)
        y_var = cp.Variable(4)

        with self.assertRaises(NotImplementedError):
            y_expr.adjoint(y_var)

    def test_convolve_in_fenchel_dual(self) -> None:
        """Test Fenchel dualization with convolve in objective"""
        c = cp.Constant([1.0, -0.5, 0.25])
        x = cp.Variable(4)
        y = cp.atoms.affine.conv.convolve(c, x)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(y)), [x >= 0])

        try:
            dual_prob = fenchel_dual(prob)
            self.assertIsNotNone(dual_prob)
            # Should have dual variables
            self.assertGreater(len(dual_prob.variables()), 0)
        except Exception as e:
            self.fail(f"Fenchel dualization failed with convolve: {e}")


class TestComposedAtomIntegration(BaseTest):
    """Test that composed affine functions work with Fenchel dualization"""

    def test_vec_in_fenchel_dual(self) -> None:
        """Test vec (returns reshape)"""
        from cvxpy.atoms.affine.vec import vec

        X = cp.Variable((2, 3))
        x_vec = vec(X, order='F')
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x_vec)), [X >= 0])

        try:
            dual_prob = fenchel_dual(prob)
            self.assertIsNotNone(dual_prob)
        except Exception as e:
            self.fail(f"Fenchel dualization failed with vec: {e}")

    def test_squeeze_in_fenchel_dual(self) -> None:
        """Test squeeze (returns reshape)"""
        from cvxpy.atoms.affine.squeeze import squeeze

        X = cp.Variable((3, 1, 4))
        X_squeezed = squeeze(X, axis=1)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(X_squeezed)), [X >= 0])

        try:
            dual_prob = fenchel_dual(prob)
            self.assertIsNotNone(dual_prob)
        except Exception as e:
            self.fail(f"Fenchel dualization failed with squeeze: {e}")

    def test_stack_in_fenchel_dual(self) -> None:
        """Test stack (returns concatenate of reshapes)"""
        from cvxpy.atoms.affine.stack import stack

        x = cp.Variable(3)
        y = cp.Variable(3)
        stacked = stack([x, y], axis=0)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(stacked)), [x >= 0, y >= 0])

        try:
            dual_prob = fenchel_dual(prob)
            self.assertIsNotNone(dual_prob)
        except Exception as e:
            self.fail(f"Fenchel dualization failed with stack: {e}")

    def test_diff_in_fenchel_dual(self) -> None:
        """Test diff (returns slicing operations)"""
        from cvxpy.atoms.affine.diff import diff

        x = cp.Variable(5)
        dx = diff(x, k=1, axis=0)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(dx)), [x[0] == 1])

        try:
            dual_prob = fenchel_dual(prob)
            self.assertIsNotNone(dual_prob)
        except Exception as e:
            self.fail(f"Fenchel dualization failed with diff: {e}")

    def test_bmat_in_fenchel_dual(self) -> None:
        """Test bmat (returns vstack of hstacks)"""
        from cvxpy.atoms.affine.bmat import bmat

        A = cp.Variable((2, 2))
        B = cp.Variable((2, 3))
        C = cp.Variable((3, 2))
        D = cp.Variable((3, 3))

        M = bmat([[A, B], [C, D]])
        prob = cp.Problem(cp.Minimize(cp.sum_squares(M)), [A >= 0, D >= 0])

        try:
            dual_prob = fenchel_dual(prob)
            self.assertIsNotNone(dual_prob)
        except Exception as e:
            self.fail(f"Fenchel dualization failed with bmat: {e}")


class TestFenchelIntegration(BaseTest):
    """End-to-end integration tests for Fenchel dualization"""

    def test_wraps_with_constraints(self) -> None:
        """Test wrapped variables in both objective and constraints"""
        x = cp.Variable(3)
        wrapped = cp.atoms.affine.wraps.nonneg_wrap(x)

        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(wrapped) + cp.sum(wrapped)),
            [wrapped <= 10]
        )

        dual_prob = fenchel_dual(prob)
        self.assertIsNotNone(dual_prob)

        # Solve both and verify strong duality
        prob.solve(solver=cp.CLARABEL)
        dual_prob.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(prob.value, dual_prob.value, places=3)

    def test_conv_with_multiple_constraints(self) -> None:
        """Test convolve with various constraint types"""
        c = cp.Constant([1.0, -0.5, 0.25])
        x = cp.Variable(4)
        y = cp.atoms.affine.conv.convolve(c, x)

        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(y)),
            [x >= 0, x <= 5, cp.sum(x) <= 10]
        )

        dual_prob = fenchel_dual(prob)
        self.assertIsNotNone(dual_prob)

        prob.solve(solver=cp.CLARABEL)
        dual_prob.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(prob.value, dual_prob.value, places=3)

    def test_multiple_composed_atoms(self) -> None:
        """Test problem with multiple composed atoms"""
        from cvxpy.atoms.affine.diff import diff
        from cvxpy.atoms.affine.vec import vec

        X = cp.Variable((3, 4))
        x_vec = vec(X, order='F')
        X_diff = diff(X, k=1, axis=0)

        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x_vec) + cp.sum_squares(X_diff)),
            [X >= 0]
        )

        dual_prob = fenchel_dual(prob)
        self.assertIsNotNone(dual_prob)

        prob.solve(solver=cp.CLARABEL)
        dual_prob.solve(solver=cp.CLARABEL)

        self.assertAlmostEqual(prob.value, dual_prob.value, places=3)

    def test_per_variable_stationarity(self) -> None:
        """Test that stationarity produces per-variable constraints"""
        x = cp.Variable(2, name='x')
        y = cp.Variable(3, name='y')

        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x) + cp.sum_squares(y)),
            [x >= 0, y >= 0]
        )

        dual_prob, data = cp.reductions.fenchel_dual.FenchelDual().apply(prob)

        # Should have separate stationarity constraints for x and y
        fenchel_data = data["fenchel"]
        stationarity_map = fenchel_data["stationarity_map"]

        # stationarity_map should have entries (one per original variable)
        self.assertGreaterEqual(len(stationarity_map), 2)

        # Verify both variables are in the map values
        var_ids_in_map = set(stationarity_map.values())
        self.assertIn(x.id, var_ids_in_map)
        self.assertIn(y.id, var_ids_in_map)


class TestComplexFenchelDualization(BaseTest):
    """Test Fenchel dualization with complex variables."""

    def test_transpose_hermitian_adjoint(self) -> None:
        """Test that transpose adjoint uses Hermitian transpose for complex."""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint
        from cvxpy.atoms.affine.conj import conj

        # Complex matrix variable
        X = cp.Variable((2, 3), complex=True)
        expr = X.T  # transpose
        y_var = cp.Variable((3, 2), complex=True)

        adj_dict, const_term = _symbolic_adjoint(expr, y_var)

        # The adjoint should be conj(y_var.T) = y_var^H
        self.assertEqual(len(adj_dict), 1)
        self.assertIn(X.id, adj_dict)

        # Set values to test
        X.value = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
        y_var.value = np.array([[1+0j, 0+1j], [1-1j, 0+0j], [1+1j, 1-1j]])

        # Adjoint expression should equal conj(y_var.T)
        expected = np.conj(y_var.value.T)
        adj_value = adj_dict[X.id].value

        self.assertTrue(np.allclose(adj_value, expected))

    def test_complex_variable_dualization(self) -> None:
        """Test Fenchel dual with complex variables."""
        # Simple problem: minimize ||x||_2^2 subject to Re(x[0]) = 1
        # This is definitely feasible: x = [1, 0]
        x = cp.Variable(2, complex=True)

        primal = cp.Problem(
            cp.Minimize(cp.sum_squares(x)),
            [cp.real(x[0]) == 1, cp.imag(x[0]) == 0]
        )
        primal.solve(solver=cp.SCS, eps=1e-6)

        # Dualize
        dual_prob = fenchel_dual(primal)
        dual_prob.solve(solver=cp.SCS, eps=1e-6)

        # Check strong duality
        self.assertIn(primal.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        self.assertIn(dual_prob.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        self.assertAlmostEqual(primal.value, dual_prob.value, places=3)

    def test_complex_inner_product(self) -> None:
        """Test complex inner product in stationarity."""
        from cvxpy.reductions.fenchel_dual import _inner_product

        y = cp.Variable(2, complex=True)
        z = cp.Constant([1+1j, 2-1j])

        inner = _inner_product(y, z)

        # Should be Re(sum(conj(y) * z))
        y.value = np.array([1+0j, 0+1j])
        # conj([1, i]) * [1+i, 2-i] = [1, -i] * [1+i, 2-i] = [1+i, -2i-1] = [1+i, -1-2i]
        # sum = 1+i-1-2i = -i
        # Re(-i) = 0
        self.assertAlmostEqual(inner.value, 0.0)

    def test_complex_matmul_adjoint(self) -> None:
        """Test matrix multiplication adjoint with complex variables."""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint

        A = cp.Constant(np.array([[1+1j, 2+0j], [0+1j, 1-1j]]))
        x = cp.Variable(2, complex=True)
        expr = A @ x
        y_var = cp.Variable(2, complex=True)

        adj_dict, const_term = _symbolic_adjoint(expr, y_var)

        # Adjoint should be A^H @ y_var
        y_var.value = np.array([1+0j, 0+1j])
        expected = np.conj(A.value.T) @ y_var.value

        self.assertTrue(np.allclose(adj_dict[x.id].value, expected))

    def test_complex_sum_adjoint(self) -> None:
        """Test sum adjoint with complex variables."""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint

        X = cp.Variable((2, 3), complex=True)
        expr = cp.sum(X)
        y_var = cp.Variable()  # scalar dual variable

        adj_dict, const_term = _symbolic_adjoint(expr, y_var)

        # Adjoint should broadcast y_var to (2, 3)
        self.assertEqual(adj_dict[X.id].shape, (2, 3))

        # All entries should be y_var (real sum, so no conjugation needed in broadcast)
        y_var.value = 2.0
        expected = np.full((2, 3), 2.0, dtype=complex)
        self.assertTrue(np.allclose(adj_dict[X.id].value, expected))

    def test_complex_conj_adjoint(self) -> None:
        """Test conj atom adjoint."""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint
        from cvxpy.atoms.affine.conj import conj

        x = cp.Variable(2, complex=True)
        expr = conj(x)
        y_var = cp.Variable(2, complex=True)

        adj_dict, const_term = _symbolic_adjoint(expr, y_var)

        # Adjoint of conj is conj: <y, conj(x)> = <conj(y), x>
        y_var.value = np.array([1+1j, 2-1j])
        expected = np.conj(y_var.value)

        self.assertTrue(np.allclose(adj_dict[x.id].value, expected))

    def test_complex_real_imag_adjoints(self) -> None:
        """Test real and imag atom adjoints."""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint
        from cvxpy.atoms.affine.real import real
        from cvxpy.atoms.affine.imag import imag

        x = cp.Variable(2, complex=True)

        # Real part: <y, Re(x)> = <y, x> (y real)
        real_expr = real(x)
        y_real = cp.Variable(2)
        adj_dict_real, _ = _symbolic_adjoint(real_expr, y_real)

        y_real.value = np.array([1.0, 2.0])
        # Adjoint should be y_real (as complex)
        self.assertEqual(adj_dict_real[x.id].shape, (2,))

        # Imag part: <y, Im(x)> = <-iy, x> (y real)
        imag_expr = imag(x)
        y_imag = cp.Variable(2)
        adj_dict_imag, _ = _symbolic_adjoint(imag_expr, y_imag)

        y_imag.value = np.array([1.0, 2.0])
        # Adjoint should be -i * y_imag
        self.assertEqual(adj_dict_imag[x.id].shape, (2,))

    def test_complex_validation_warning(self) -> None:
        """Test that missing complex conjugation raises warning."""
        from cvxpy.reductions.fenchel_dual import _symbolic_adjoint
        import warnings

        # Create a mock affine atom that doesn't conjugate properly
        class BadComplexAtom(cp.atoms.affine.affine_atom.AffAtom):
            def __init__(self, x):
                super().__init__(x)

            def numeric(self, values):
                return values[0]

            def shape_from_args(self):
                return self.args[0].shape

            def sign_from_args(self):
                return (False, False)

            def is_atom_convex(self):
                return True

            def is_atom_concave(self):
                return True

            def adjoint(self, y_var):
                # WRONG: should conjugate for complex, but doesn't
                return [(0, y_var)]

        x = cp.Variable(2, complex=True)
        bad_expr = BadComplexAtom(x)
        y_var = cp.Variable(2, complex=True)

        # Should raise warning about missing conjugation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _symbolic_adjoint(bad_expr, y_var)

            # Verify warning was issued
            self.assertEqual(len(w), 1)
            self.assertIn("missing conjugation", str(w[0].message).lower())
