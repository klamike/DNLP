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

from typing import Tuple

import numpy as np

from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.utilities import bounds as bounds_utils


class exp(Elementwise):
    """Elementwise :math:`e^{x}`.
    """

    def __init__(self, x) -> None:
        super(exp, self).__init__(x)

    # Returns the matrix e^x[i, j].
    @Elementwise.numpy_numeric
    def numeric(self, values):
        return np.exp(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def bounds_from_args(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns bounds for exp based on argument bounds."""
        lb, ub = self.args[0].get_bounds()
        return bounds_utils.exp_bounds(lb, ub)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

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
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def conjugate(self, y, perspective_scale=1):
        """Fenchel conjugate of exp(x) is y*log(y) - y for y >= 0.

        f*(y) = sup_x { y*x - exp(x) } = y*log(y) - y,  dom f* = { y >= 0 }.

        For perspective functions (Roos et al. 2020, Appendix B.9):
        - (s*exp)^*(y) = y*log(y/s) - y for s > 0, y >= 0
        - (0*exp)^*(y) = δ_0(y): strict convention, not closure

        IMPORTANT: The paper warns that closing the perspective can produce
        wrong duals. We enforce the strict perspective convention: when scale
        can be zero, we require either scale > 0 OR y == 0.
        """
        from cvxpy.atoms.elementwise.rel_entr import rel_entr
        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))

        # Strict perspective convention: (0*f)^*(y) = δ_0(y)
        # When scale is identically zero, enforce y == 0
        if scale.is_nonneg() and scale.is_nonpos():
            return self.indicator_conjugate([y == 0])

        # For s > 0: (s*exp)^*(y) = y*log(y/s) - y
        # rel_entr(y, s) models y*log(y/s) with closure at y = s = 0.
        # This is correct for s > 0, but the strict s=0 case is handled above.
        #
        # Note: rel_entr has the closure property that rel_entr(0, 0) = 0.
        # For variable scales, we rely on scale >= 0 constraint to prevent
        # the closure from being incorrectly applied. The dual solver will
        # enforce that if scale approaches 0, y must also approach 0 (since
        # rel_entr(y, s) → +∞ as s → 0+ with y > 0 fixed).
        conj_expr = rel_entr(y, scale) - y
        constraints = [y >= 0, scale >= 0]
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
        grad_vals = np.exp(values[0])
        return [exp.elemwise_grad_to_diag(grad_vals, rows, cols)]
