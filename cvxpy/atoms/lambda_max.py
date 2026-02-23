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
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy import linalg as LA

from cvxpy.atoms.atom import Atom
from cvxpy.atoms.affine.trace import trace
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression


class lambda_max(Atom):
    """ Maximum eigenvalue; :math:`\\lambda_{\\max}(A)`.
    """

    def __init__(self, A) -> None:
        super(lambda_max, self).__init__(A)

    def numeric(self, values):
        """Returns the largest eigenvalue of A.

        Requires that A be symmetric.
        """
        lo = hi = self.args[0].shape[0]-1
        return LA.eigvalsh(values[0], subset_by_index=(lo, hi))[0]

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0].H == self.args[0]]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        w, v = LA.eigh(values[0])
        d = np.zeros(w.shape)
        d[-1] = 1
        d = np.diag(d)
        D = v.dot(d).dot(v.T)
        return [sp.csc_array([D.ravel(order='F')]).T]

    def validate_arguments(self) -> None:
        """Verify that the argument A is a square matrix.
        """
        if not self.args[0].ndim == 2 or self.args[0].shape[0] != self.args[0].shape[1]:
            raise ValueError("The argument '%s' to lambda_max must resolve to a square matrix."
                             % self.args[0].name())

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (False, False)

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

    @property
    def value(self):
        if not np.allclose(self.args[0].value, self.args[0].value.T.conj()):
            raise ValueError("Input matrix was not Hermitian/symmetric.")
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()

    def conjugate(self, y, perspective_scale=1):
        """Fenchel conjugate of lambda_max is PSD+trace indicator."""
        if y.ndim != 2 or y.shape[0] != y.shape[1]:
            raise ValueError("Fenchel conjugate of lambda_max expects a square dual matrix.")
        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))
        if not scale.is_scalar():
            raise ValueError("Perspective scale for lambda_max conjugate must be scalar.")
        if scale.is_complex() and not scale.is_real():
            raise NotImplementedError(
                "Complex perspective multipliers are not supported for lambda_max conjugates."
            )
        return self.indicator_conjugate([y == y.H, y >> 0, trace(y) == scale])
