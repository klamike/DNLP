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
from scipy.special import xlogy

from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.affine.promote import promote
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable

# TODO(akshayka): DGP support.


class entr(Elementwise):
    """Elementwise :math:`-x\\log x`.
    """

    def __init__(self, x) -> None:
        super(entr, self).__init__(x)

    def numeric(self, values):
        x = values[0]
        results = -xlogy(x, x)
        # Return -inf outside the domain
        if np.isscalar(results):
            if np.isnan(results):
                return -np.inf
        else:
            results[np.isnan(results)] = -np.inf
        return results

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always unknown.
        return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

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
            grad_vals = -np.log(values[0]) - 1
            return [entr.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def conjugate(self, y, perspective_scale=1):
        self.raise_convex_only_conjugate_not_implemented("entr")

    def negative_conjugate(self, y, perspective_scale=1):
        # ((s * (-entr))^*)(y) = s * exp(y / s - 1), with closure at s=0.
        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))
        scale_arg = promote(scale, y.shape) if scale.is_scalar() else scale
        if scale_arg.shape != y.shape:
            raise ValueError(
                "Perspective scale for -entr conjugate must be scalar or "
                "match the dual shape."
            )
        z = Variable(y.shape, name="z_neg_entr_conj")
        return z, [ExpCone(y - scale_arg, scale_arg, z), scale_arg >= 0]

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >= 0]
