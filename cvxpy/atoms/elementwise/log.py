"""
Copyright 2013 Steven Diamond, Eric Chu

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

from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.rel_entr import rel_entr
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.utilities import bounds as bounds_utils


class log(Elementwise):
    """Elementwise :math:`\\log x`.
    """

    def __init__(self, x) -> None:
        super(log, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise natural log of x.
        """
        return np.log(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always unknown.
        return (False, False)

    def bounds_from_args(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns bounds for log based on argument bounds."""
        lb, ub = self.args[0].get_bounds()
        return bounds_utils.log_bounds(lb, ub)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return True

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        rows = self.args[0].size
        cols = self.size
        # Outside domain or on boundary.
        if np.min(values[0]) <= 0:
            # Non-differentiable.
            return [None]
        else:
            grad_vals = 1.0/values[0]
            return [log.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def conjugate(self, y, perspective_scale=1):
        self.raise_convex_only_conjugate_not_implemented("log")

    def negative_conjugate(self, y, perspective_scale=1):
        # ((s * (-log))^*)(y) = s * (-1 - log(-y / s)) for s > 0.
        # Per the paper's perspective convention at s = 0:
        # (0 * f)^*(y) = delta_{0}(y), so the domain collapses to y == 0.
        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))
        if scale.is_nonneg() and scale.is_nonpos():
            return self.indicator_conjugate([y == 0])
        return rel_entr(scale, -y) - scale, [y <= 0, scale >= 0]

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >= 0]
