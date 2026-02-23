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

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csc_array
from scipy.special import kl_div as kl_div_scipy

from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable


class kl_div(Elementwise):
    """:math:`x\\log(x/y) - x + y`

    For disambiguation between kl_div and rel_entr, see https://github.com/cvxpy/cvxpy/issues/733
    """

    def __init__(self, x, y) -> None:
        super(kl_div, self).__init__(x, y)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        x = values[0]
        y = values[1]
        return kl_div_scipy(x, y)

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
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _grad(self, values) -> List[Optional[csc_array]]:
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        if np.min(values[0]) <= 0 or np.min(values[1]) <= 0:
            # Non-differentiable.
            return [None, None]
        else:
            div = values[0]/values[1]
            grad_vals = [np.log(div), 1 - div]
            grad_list = []
            for idx in range(len(values)):
                rows = self.args[idx].size
                cols = self.size
                grad_list += [kl_div.elemwise_grad_to_diag(grad_vals[idx],
                                                           rows, cols)]
            return grad_list

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >= 0, self.args[1] >= 0]

    @staticmethod
    def _scale_arg(scale, shape):
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))
        if scale.is_complex() and not scale.is_real():
            raise NotImplementedError(
                "Complex perspective multipliers are not supported for kl_div conjugates."
            )
        scale_arg = promote(scale, shape) if scale.is_scalar() else scale
        if scale_arg.shape != shape:
            raise ValueError(
                "Perspective scale for kl_div conjugate must be scalar or match the dual shape."
            )
        return scale_arg

    def conjugate(self, y, perspective_scale=1):
        """Conjugate wrt first argument when second argument is constant."""
        if not self.args[1].is_constant():
            raise NotImplementedError(
                "Fenchel conjugate of kl_div requires constant second argument."
            )
        if not y.is_real():
            raise ValueError("Fenchel conjugate of kl_div is defined for real dual variables.")

        scale_arg = self._scale_arg(perspective_scale, y.shape)
        second_arg = self.args[1]
        second_arg = promote(second_arg, y.shape) if second_arg.is_scalar() else second_arg
        if second_arg.shape != y.shape:
            raise ValueError(
                "Second argument of kl_div must be scalar or match the dual shape."
            )
        if scale_arg.is_nonneg() and scale_arg.is_nonpos():
            return self.indicator_conjugate([y == 0])

        z = Variable(y.shape, nonneg=True, name="z_kl_div_conj")
        return multiply(second_arg, z - scale_arg), [ExpCone(y, scale_arg, z), second_arg >= 0]

    def conjugate_term(self, nonconstant_arg_indices, dual_vars, perspective_scale=1):
        if nonconstant_arg_indices == (0,):
            if len(dual_vars) != 1:
                raise ValueError("kl_div unary term-conjugate expects one dual variable.")
            return self.conjugate(dual_vars[0], perspective_scale=perspective_scale)

        if nonconstant_arg_indices == (0, 1):
            if len(dual_vars) != 2:
                raise ValueError("kl_div full term-conjugate expects two dual variables.")
            u, v = dual_vars
            if not (u.is_real() and v.is_real()):
                raise ValueError("Fenchel conjugate of kl_div is defined for real dual variables.")
            if u.shape != v.shape:
                raise ValueError("kl_div full term-conjugate expects matching dual shapes.")
            scale_arg = self._scale_arg(perspective_scale, u.shape)
            if scale_arg.is_nonneg() and scale_arg.is_nonpos():
                return self.indicator_conjugate([u == 0, v == 0])
            return self.indicator_conjugate([ExpCone(u, scale_arg, scale_arg - v)])

        return super(kl_div, self).conjugate_term(
            nonconstant_arg_indices,
            dual_vars,
            perspective_scale=perspective_scale,
        )
