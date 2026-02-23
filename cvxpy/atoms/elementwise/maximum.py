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


class maximum(Elementwise):
    """Elementwise maximum of a sequence of expressions.
    """

    def __init__(self, arg1, arg2, *args) -> None:
        """Requires at least 2 arguments.
        """
        super(maximum, self).__init__(arg1, arg2, *args)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise maximum.
        """
        return reduce(np.maximum, values)

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Reduces the list of argument signs according to the following rules:
        #     POSITIVE, ANYTHING = POSITIVE
        #     ZERO, UNKNOWN = POSITIVE
        #     ZERO, ZERO = ZERO
        #     ZERO, NEGATIVE = ZERO
        #     UNKNOWN, NEGATIVE = UNKNOWN
        #     NEGATIVE, NEGATIVE = NEGATIVE
        is_pos = any(arg.is_nonneg() for arg in self.args)
        is_neg = all(arg.is_nonpos() for arg in self.args)
        return (is_pos, is_neg)

    def bounds_from_args(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns bounds for elementwise maximum based on argument bounds."""
        bounds_list = [arg.get_bounds() for arg in self.args]
        return bounds_utils.maximum_bounds(bounds_list)

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

    def is_pwl(self) -> bool:
        """Is the atom piecewise linear?
        """
        return all(arg.is_pwl() for arg in self.args)

    def conjugate_term(self, nonconstant_arg_indices, dual_vars, perspective_scale=1):
        # For f(x_1, ..., x_m) = maximum(x_1, ..., x_m),
        # ((s * f)^*)(y) over non-constant args is:
        # - all variable-arg duals y_i >= 0
        # - if there are no constant args: sum_i y_i == s (pure indicator)
        # - with constant args c_j: sum_i y_i <= s and objective max_j(c_j)*(sum_i y_i - s).
        nonconstant_arg_indices = tuple(nonconstant_arg_indices)
        if len(dual_vars) != len(nonconstant_arg_indices):
            raise ValueError(
                "maximum term-conjugate expects one dual variable per "
                "non-constant maximum argument."
            )
        if not nonconstant_arg_indices:
            raise NotImplementedError(
                "Fenchel conjugate of maximum requires at least one "
                "non-constant argument."
            )

        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))

        sum_duals = dual_vars[0]
        constraints = [dual_vars[0] >= 0]
        for dual_var in dual_vars[1:]:
            sum_duals = sum_duals + dual_var
            constraints.append(dual_var >= 0)

        scale_arg = promote(scale, sum_duals.shape) if scale.is_scalar() else scale
        if scale_arg.shape != sum_duals.shape:
            raise ValueError(
                "Perspective scale for maximum conjugate must be scalar or "
                "match the maximum output shape."
            )

        constant_args = [
            self.args[idx] for idx in range(len(self.args))
            if idx not in nonconstant_arg_indices
        ]
        if not constant_args:
            constraints.append(sum_duals == scale_arg)
            return self.indicator_conjugate(constraints)

        constant_ceil = constant_args[0]
        for const_arg in constant_args[1:]:
            constant_ceil = maximum(constant_ceil, const_arg)
        slack = scale_arg - sum_duals
        constraints.append(slack >= 0)
        return -multiply(constant_ceil, slack), constraints

    def _grad(self, values) -> List[Any]:
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        max_vals = self.numeric(values)
        unused = np.ones(max_vals.shape, dtype=bool)
        grad_list = []
        for idx, value in enumerate(values):
            rows = self.args[idx].size
            cols = self.size
            grad_vals = (value == max_vals) & unused
            # Remove all the max_vals that were used.
            unused[value == max_vals] = 0
            grad_list += [maximum.elemwise_grad_to_diag(grad_vals,
                                                        rows, cols)]
        return grad_list
