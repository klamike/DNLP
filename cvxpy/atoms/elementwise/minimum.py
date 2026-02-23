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

from functools import reduce
from typing import Any, List, Tuple

import numpy as np

from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.utilities import bounds as bounds_utils


class minimum(Elementwise):
    """Elementwise minimum of a sequence of expressions.
    """

    def __init__(self, arg1, arg2, *args) -> None:
        """Requires at least 2 arguments.
        """
        super(minimum, self).__init__(arg1, arg2, *args)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise maximum.
        """
        return reduce(np.minimum, values)

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        is_pos = all(arg.is_nonneg() for arg in self.args)
        is_neg = any(arg.is_nonpos() for arg in self.args)
        return (is_pos, is_neg)

    def bounds_from_args(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns bounds for elementwise minimum based on argument bounds."""
        bounds_list = [arg.get_bounds() for arg in self.args]
        return bounds_utils.minimum_bounds(bounds_list)

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

    def is_pwl(self) -> bool:
        """Is the atom piecewise linear?
        """
        return all(arg.is_pwl() for arg in self.args)

    def negative_conjugate_term(
        self,
        nonconstant_arg_indices,
        dual_vars,
        perspective_scale=1,
    ):
        # For f(x_1, ..., x_m) = minimum(x_1, ..., x_m),
        # ((s * (-f))^*)(y) over non-constant args is:
        # - all variable-arg duals y_i <= 0
        # - slack := s + sum_i y_i
        # - if there are no constant args: slack == 0 (pure indicator)
        # - with constant args c_j: slack >= 0 and objective min_j(c_j) * slack.
        nonconstant_arg_indices = tuple(nonconstant_arg_indices)
        if len(dual_vars) != len(nonconstant_arg_indices):
            raise ValueError(
                "minimum negative term-conjugate expects one dual variable per "
                "non-constant minimum argument."
            )
        if not nonconstant_arg_indices:
            raise NotImplementedError(
                "Fenchel conjugate of -minimum requires at least one "
                "non-constant argument."
            )

        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))

        sum_duals = dual_vars[0]
        constraints = [dual_vars[0] <= 0]
        for dual_var in dual_vars[1:]:
            sum_duals = sum_duals + dual_var
            constraints.append(dual_var <= 0)

        scale_arg = promote(scale, sum_duals.shape) if scale.is_scalar() else scale
        if scale_arg.shape != sum_duals.shape:
            raise ValueError(
                "Perspective scale for -minimum conjugate must be scalar or "
                "match the minimum output shape."
            )

        slack = sum_duals + scale_arg
        constant_args = [
            self.args[idx] for idx in range(len(self.args))
            if idx not in nonconstant_arg_indices
        ]
        if not constant_args:
            constraints.append(slack == 0)
            return self.indicator_conjugate(constraints)

        constant_floor = constant_args[0]
        for const_arg in constant_args[1:]:
            constant_floor = minimum(constant_floor, const_arg)
        constraints.append(slack >= 0)
        return multiply(constant_floor, slack), constraints

    def _grad(self, values) -> List[Any]:
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        min_vals = np.array(self.numeric(values))
        unused = np.array(np.ones(min_vals.shape), dtype=bool)
        grad_list = []
        for idx, value in enumerate(values):
            rows = self.args[idx].size
            cols = self.size
            grad_vals = (value == min_vals) & unused
            # Remove all the min_vals that were used.
            unused[value == min_vals] = 0
            grad_list += [minimum.elemwise_grad_to_diag(grad_vals,
                                                        rows, cols)]
        return grad_list
