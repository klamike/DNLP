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
import scipy.sparse as sp

from cvxpy.atoms.atom import Atom
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression


class normNuc(Atom):
    """Sum of the singular values.
    """
    _allow_complex = True

    def __init__(self, A) -> None:
        super(normNuc, self).__init__(A)

    def numeric(self, values):
        """Returns the nuclear norm (i.e. the sum of the singular values) of A.
        """
        return np.linalg.norm(values[0], 'nuc')

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # Grad UV^T
        U, _, V = np.linalg.svd(values[0], full_matrices=False)
        D = U.dot(V)
        return [sp.csc_array([D.ravel(order='F')]).T]

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (True, False)

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
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def conjugate(self, y, perspective_scale=1):
        """Fenchel conjugate of nuclear norm is spectral-norm ball indicator."""
        if y.ndim != 2:
            raise ValueError("Fenchel conjugate of norm_nuc expects a matrix dual variable.")
        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))
        if not scale.is_scalar():
            raise ValueError("Perspective scale for norm_nuc conjugate must be scalar.")
        if scale.is_complex() and not scale.is_real():
            raise NotImplementedError(
                "Complex perspective multipliers are not supported for norm_nuc conjugates."
            )
        from cvxpy.atoms.sigma_max import sigma_max
        constraints = [sigma_max(y) <= scale]
        if not scale.is_nonneg():
            constraints.append(scale >= 0)
        return self.indicator_conjugate(constraints)
