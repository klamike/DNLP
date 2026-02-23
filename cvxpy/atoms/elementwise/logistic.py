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

from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.rel_entr import rel_entr
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression


class logistic(Elementwise):
    """:math:`\\log(1 + e^{x})`

    This is a special case of log(sum(exp)) that is evaluates to a vector rather
    than to a scalar which is useful for logistic regression.
    """

    def __init__(self, x) -> None:
        super(logistic, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Evaluates e^x elementwise, adds 1, and takes the log.
        """
        return np.logaddexp(0, values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
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
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def conjugate(self, y, perspective_scale=1):
        """Fenchel conjugate of logistic(x)=log(1+exp(x)).

        For scale s >= 0:
            (s*logistic)^*(y) = rel_entr(y, s) + rel_entr(s-y, s),
        with domain 0 <= y <= s.
        """
        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))
        if scale.is_complex() and not scale.is_real():
            raise NotImplementedError(
                "Complex perspective multipliers are not supported for logistic conjugates."
            )
        if not y.is_real():
            raise ValueError("Fenchel conjugate of logistic is defined for real dual variables.")

        scale_arg = promote(scale, y.shape) if scale.is_scalar() else scale
        if scale_arg.shape != y.shape:
            raise ValueError(
                "Perspective scale for logistic conjugate must be scalar or match the dual shape."
            )

        conj_expr = rel_entr(y, scale_arg) + rel_entr(scale_arg - y, scale_arg)
        constraints = [scale_arg >= 0, y >= 0, y <= scale_arg]
        return conj_expr, constraints

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
        grad_vals = np.exp(values[0] - np.logaddexp(0, values[0]))
        return [logistic.elemwise_grad_to_diag(grad_vals, rows, cols)]
