"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

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

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy import linalg as LA

from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.interface.matrix_utilities import is_sparse
from cvxpy.utilities.linalg import sparse_cholesky
from cvxpy.utilities.warn import warn


class CvxPyDomainError(Exception):
    pass


class QuadForm(Atom):
    _allow_complex = True
    block_indices = None  # For compatibility with SymbolicQuadForm

    def __init__(self, x, P) -> None:
        """Atom representing :math:`x^T P x`."""
        super(QuadForm, self).__init__(x, P)

    def numeric(self, values):
        prod = values[1].dot(values[0])
        if self.args[0].is_complex():
            quad = np.dot(np.conj(values[0]).T, prod)
        else:
            quad = np.dot(np.transpose(values[0]), prod)
        return np.real(quad)

    def validate_arguments(self) -> None:
        super(QuadForm, self).validate_arguments()
        n = self.args[1].shape[0]
        if self.args[1].shape[1] != n or self.args[0].shape not in [(n, 1), (n,)]:
            raise ValueError("Invalid dimensions for arguments to quad_form.")
        if not self.args[1].is_hermitian():
            raise ValueError("Quadratic form matrices must be symmetric/Hermitian.")

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (self.is_atom_convex(), self.is_atom_concave())

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        P = self.args[1]
        return P.is_constant() and P.is_psd()

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        P = self.args[1]
        return P.is_constant() and P.is_nsd()

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return (self.args[0].is_nonneg() and self.args[1].is_nonneg()) or \
               (self.args[0].is_nonpos() and self.args[1].is_nonneg())

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return (self.args[0].is_nonneg() and self.args[1].is_nonpos()) or \
               (self.args[0].is_nonpos() and self.args[1].is_nonpos())

    def is_quadratic(self) -> bool:
        """Is the atom quadratic?
        """
        return True

    def has_quadratic_term(self) -> bool:
        """Always a quadratic term.
        """
        return True

    def is_pwl(self) -> bool:
        """Is the atom piecewise linear?
        """
        return False

    def name(self) -> str:
        return f"{type(self).__name__}({self.args[0]}, {self.args[1]})"

    def format_labeled(self) -> str:
        if self._label is not None:
            return self._label
        return (
            f"{type(self).__name__}({self.args[0].format_labeled()}, "
            f"{self.args[1].format_labeled()})"
        )

    def _grad(self, values):
        x = np.array(values[0])
        P = np.array(values[1])
        D = (P + np.conj(P.T)) @ x
        return [sp.csc_array([D.ravel(order="F")]).T]

    def conjugate(self, y, perspective_scale=1):
        """Fenchel conjugate of (1/2) x'Px is (1/2) y' P^{-1} y.

        For the atom x'Px (without the 1/2 factor), the conjugate is
        (1/4) y' P^{-1} y.

        More precisely: f(x) = x'Px, so f*(y) = sup_x { y'x - x'Px }
        = (1/4) y' P^{-1} y (when P is PSD).

        Requires P to be a constant PSD matrix.
        """
        P = self.args[1]
        if not P.is_constant():
            raise NotImplementedError(
                "Fenchel conjugate of quad_form requires constant P."
            )
        if not P.is_psd():
            raise NotImplementedError(
                "Fenchel conjugate of quad_form requires PSD P."
            )

        P_val = P.value.toarray() if is_sparse(P.value) else np.asarray(P.value)
        P_num = np.asarray(
            P_val,
            dtype=np.complex128 if np.iscomplexobj(P_val) else float,
        )
        # Guard against minor numerical asymmetry in user-provided constants.
        P_num = 0.5 * (P_num + np.conjugate(P_num.T))

        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(scale)
        if not scale.is_scalar(): raise ValueError("Perspective-conjugate of quad_form requires a scalar multiplier.")
        if scale.is_complex() and not scale.is_real():
            raise ValueError(
                "Perspective-conjugate of quad_form requires a real multiplier."
            )
        if scale.is_nonneg() and scale.is_nonpos():
            return self.indicator_conjugate([y == 0])
        scale_value = np.asarray(scale.value) if scale.value is not None else None
        scale_is_one = (
            scale.is_constant()
            and scale_value is not None
            and scale_value.ndim == 0
            and float(scale_value.item()) == 1.0
        )

        if P.parameters():
            return self._parameterized_psd_conjugate(y, scale)

        # Improved singular PSD handling with adaptive tolerance
        n = P_num.shape[0]
        evals, evecs = np.linalg.eigh(P_num)

        # Adaptive tolerance based on matrix norm and condition number
        # This is more robust than fixed tolerance
        max_eval = np.max(np.abs(evals))
        tol = np.finfo(float).eps * max(1.0, max_eval) * n

        # Validate that P is PSD (no significantly negative eigenvalues)
        min_eval = np.min(evals)
        if min_eval < -tol:
            raise ValueError(
                f"quad_form conjugate requires PSD matrix P, but got "
                f"min eigenvalue {min_eval:.3e} < -{tol:.3e}. "
                f"Matrix may be numerically indefinite."
            )

        # Determine positive eigenvalues (non-zero in PSD sense)
        pos = evals > tol
        rank = np.sum(pos)

        if scale_is_one:
            # Keep closed form in the unscaled case so atom-level value
            # evaluation remains available.
            # Compute P^{-1} using pseudoinverse for robustness.
            P_inv = np.linalg.pinv(P_num, rcond=tol/max_eval if max_eval > 0 else 1e-15)
            # Symmetrize the pseudoinverse
            P_inv = 0.5 * (P_inv + np.conjugate(P_inv.T))

            conj_expr = 0.25 * QuadForm(y, Constant(P_inv))
            constraints = []

            if rank < n:
                # Domain for singular PSD P: y must lie in range(P).
                # Use eigendecomposition for stable range projection
                Q_null = evecs[:, ~pos]  # Null space basis
                # y must be orthogonal to null space: Q_null^H y = 0
                constraints.append(Constant(Q_null.conj().T) @ y == 0)

            return conj_expr, constraints

        # For scaled case, use eigendecomposition
        if not np.any(pos):
            return self.indicator_conjugate([y == 0])

        B = evecs[:, pos] @ np.diag(np.sqrt(evals[pos]))

        w = Variable(B.shape[1], complex=np.iscomplexobj(B), name="w_quad_form_persp")
        y_from_w = reshape(Constant(B) @ w, y.shape, order="F")
        return quad_over_lin(w, 4.0 * scale), [y == y_from_w]

    def _parameterized_psd_conjugate(self, y, scale):
        """Conjugate epigraph for PSD parameter matrices via Schur complement."""
        P = self.args[1]
        n = y.size
        y_col = reshape(y, (n, 1), order="F")
        t = Variable(name="t_quad_form_param_conj")
        top = hstack([reshape(4.0 * t, (1, 1), order="F"), y_col.H])
        bottom = hstack([y_col, scale * P])
        constraints = [vstack([top, bottom]) >> 0]
        if not scale.is_nonneg():
            constraints.append(scale >= 0)
        return t, constraints

    def negative_conjugate(self, y, perspective_scale=1):
        """Conjugate of -quad_form(x, P), requiring constant NSD P.

        For NSD P, -x'Px = x'(-P)x with PSD -P, so this delegates to the
        convex quad_form conjugate on -P.
        """
        P = self.args[1]
        if P.parameters():
            raise NotImplementedError(
                "Fenchel conjugate of -quad_form with Parameter matrix "
                "is not implemented."
            )
        if not P.is_constant():
            raise NotImplementedError(
                "Fenchel conjugate of -quad_form requires constant P."
            )
        if not P.is_nsd():
            raise NotImplementedError(
                "Fenchel conjugate of -quad_form requires NSD P."
            )

        return QuadForm(self.args[0], Constant(-P.value)).conjugate(
            y,
            perspective_scale=perspective_scale,
        )

    def shape_from_args(self) -> Tuple[int, ...]:
        return tuple()


class SymbolicQuadForm(Atom):
    """
    Symbolic form of QuadForm when quadratic matrix is not known (yet).

    Parameters
    ----------
    x : Variable or Expression
        The input expression.
    P : ndarray or sparse matrix
        The quadratic matrix.
    expr : Expression
        The original expression that this represents.
    block_indices : list of np.ndarray, optional
        For non-scalar outputs, maps each output element j to input indices.
        block_indices[j] is an array of indices that output[j] depends on.
        Supports both contiguous and non-contiguous blocks.
        If None, uses existing scalar/diagonal behavior.
    """
    def __init__(self, x, P, expr, block_indices=None) -> None:
        self.original_expression = expr
        self.block_indices = block_indices
        super(SymbolicQuadForm, self).__init__(x, P)
        self.P = self.args[1]

    def get_data(self):
        return [self.original_expression, self.block_indices]

    def _grad(self, values):
        raise NotImplementedError()

    def is_atom_concave(self) -> bool:
        return self.original_expression.is_atom_concave()

    def is_atom_convex(self) -> bool:
        return self.original_expression.is_atom_convex()

    def is_decr(self, idx) -> bool:
        return self.original_expression.is_decr(idx)

    def is_incr(self, idx) -> bool:
        return self.original_expression.is_incr(idx)

    def shape_from_args(self) -> Tuple[int, ...]:
        return self.original_expression.shape_from_args()

    def sign_from_args(self) -> Tuple[bool, bool]:
        return self.original_expression.sign_from_args()

    def is_quadratic(self) -> bool:
        return True


def decomp_quad(P, cond=None, rcond=None, lower=True, check_finite: bool = True):
    """
    Compute a matrix decomposition.

    Compute sgn, scale, M such that P = sgn * scale * dot(M, M.T).
    The strategy of determination of eigenvalue negligibility follows
    the pinvh contributions from the scikit-learn project to scipy.

    Parameters
    ----------
    P : matrix or ndarray
        A real symmetric positive or negative (semi)definite input matrix
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue
        are considered negligible.
        If None or -1, suitable machine precision is used (default).
    lower : bool, optional
        Whether the array data is taken from the lower or upper triangle of P.
        The default is to take it from the lower triangle.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        The default is True; disabling may give a performance gain
        but may result in problems (crashes, non-termination) if the inputs
        contain infinities or NaNs.

    Returns
    -------
    scale : float
        induced matrix 2-norm of P
    M1, M2 : 2d ndarray
        A rectangular ndarray such that P = scale * (dot(M1, M1.T) - dot(M2, M2.T))

    """
    if is_sparse(P):
        # TODO: consider using QDLDL instead, if available.
        try:
            sign, L, p = sparse_cholesky(P)
            if sign > 0:
                return 1.0, L[p, :], np.empty((0, 0))
            else:
                return 1.0, np.empty((0, 0)), L[:, p]
        except (ValueError, ModuleNotFoundError):
            P = np.array(P.todense())  # make dense (needs to happen for eigh).
    w, V = LA.eigh(P, lower=lower, check_finite=check_finite)

    if rcond is not None:
        cond = rcond
    if cond in (None, -1):
        t = V.dtype.char.lower()
        factor = {'f': 1e3, 'd': 1e6}
        cond = factor[t] * np.finfo(t).eps

    scale = max(np.absolute(w))
    if scale == 0:
        w_scaled = w
    else:
        w_scaled = w / scale
    maskp = w_scaled > cond
    maskn = w_scaled < -cond
    # TODO: allow indefinite quad_form
    if np.any(maskp) and np.any(maskn):
        warn("Forming a nonconvex expression quad_form(x, indefinite).")
    M1 = V[:, maskp] * np.sqrt(w_scaled[maskp])
    M2 = V[:, maskn] * np.sqrt(-w_scaled[maskn])
    return scale, M1, M2


def quad_form(x, P, assume_PSD: bool = False):
    """ Alias for :math:`x^T P x`.

    Parameters
    ----------
    x : vector argument.
    P : matrix argument.
    assume_PSD : P is assumed to be PSD without checking.
    """
    x, P = map(Expression.cast_to_const, (x, P))
    # Check dimensions.
    if not P.ndim == 2 or P.shape[0] != P.shape[1] or max(x.shape, (1,))[0] != P.shape[0]:
        raise Exception("Invalid dimensions for arguments to quad_form.")
    if x.is_constant():
        return x.T.conjugate() @ P @ x
    elif P.is_constant():
        if assume_PSD:
            P = psd_wrap(P)
        return QuadForm(x, P)
    else:
        raise Exception(
            "At least one argument to quad_form must be non-variable."
        )
