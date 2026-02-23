"""
Copyright 2013 Steven Diamond

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
from numpy.lib.array_utils import normalize_axis_tuple
from scipy.special import logsumexp

from cvxpy.atoms.affine.broadcast_to import broadcast_to
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.elementwise.rel_entr import rel_entr
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression


class log_sum_exp(AxisAtom):
    """:math:`\\log\\sum_i e^{x_i}`

    """

    def __init__(self, x, axis=None, keepdims: bool = False) -> None:
        super(log_sum_exp, self).__init__(x, axis=axis, keepdims=keepdims)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Evaluates e^x elementwise, sums, and takes the log.
        """
        return logsumexp(values[0], axis=self.axis, keepdims=self.keepdims)

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return self._axis_grad(values)

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray or None.
        """
        denom = np.exp(logsumexp(value, axis=None, keepdims=True))
        nom = np.exp(value)
        D = nom/denom
        return D

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Non-negative when arg is non-negative.
        return (self.args[0].is_nonneg(), False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _broadcast_scale(self, scale, y):
        if scale.is_scalar():
            return scale
        if scale.shape != self.shape:
            raise ValueError(
                "Perspective scale for log_sum_exp conjugate must be scalar "
                "or match the log_sum_exp output shape."
            )
        if self.axis is None:
            return scale
        if self.keepdims:
            return broadcast_to(scale, y.shape)

        axes = normalize_axis_tuple(self.axis, y.ndim)
        expanded_shape = []
        out_dim = 0
        for dim in range(y.ndim):
            if dim in axes:
                expanded_shape.append(1)
            else:
                expanded_shape.append(scale.shape[out_dim])
                out_dim += 1
        return broadcast_to(reshape(scale, tuple(expanded_shape), order="F"), y.shape)

    def conjugate(self, y, perspective_scale=1):
        """Fenchel conjugate of log_sum_exp.

        For scale s > 0:
            (s*log_sum_exp)^*(y) = sum_i y_i*log(y_i/s) subject to sum(y) = s, y >= 0

        For perspective functions (Roos et al. 2020, Appendix B.9):
        - (0*log_sum_exp)^*(y) = δ_0(y): strict convention, not closure
        """
        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))
        if scale.is_complex() and not scale.is_real():
            raise NotImplementedError(
                "Complex perspective multipliers are not supported for log_sum_exp conjugates."
            )

        # Strict perspective convention: (0*f)^*(y) = δ_0(y)
        # When scale is identically zero, enforce y == 0
        if scale.is_nonneg() and scale.is_nonpos():
            return self.indicator_conjugate([y == 0])

        # For s > 0: standard conjugate with simplex constraint
        scale_full = self._broadcast_scale(scale, y)
        conj_expr = self._axis_sum(rel_entr(y, scale_full))
        constraints = [y >= 0, self._axis_sum(y) == scale]
        if not scale.is_nonneg():
            constraints.append(scale >= 0)
        return conj_expr, constraints
