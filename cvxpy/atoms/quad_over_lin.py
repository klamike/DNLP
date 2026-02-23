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

from typing import List, Optional, Tuple, Union

import numpy as np
import scipy as scipy
import scipy.sparse as sp

import cvxpy.utilities as u
from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import sum as cp_sum
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.geo_mean import geo_mean
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.constants.parameter import is_param_free
from cvxpy.expressions.variable import Variable


class quad_over_lin(AxisAtom):
    """:math:`(sum_{ij}X^2_{ij})/y`

    When axis is specified, computes the sum of squares along that axis,
    returning a vector instead of a scalar.
    """
    _allow_complex = True

    def __init__(
        self,
        x,
        y,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> None:
        self.axis = axis
        self.keepdims = keepdims
        # Call Atom.__init__ directly since we have two args
        Atom.__init__(self, x, y)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the sum of the entries of x squared over y.
        """
        x_val = values[0]
        y_val = values[1].item()
        if self.args[0].is_complex():
            squared = np.square(x_val.imag) + np.square(x_val.real)
        else:
            squared = np.square(x_val)

        return squared.sum(axis=self.axis, keepdims=self.keepdims) / y_val

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        # y > 0.
        return [self.args[1] >= 0]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # Gradient not implemented for axis case
        if self.axis is not None:
            return [None, None]

        X = values[0]
        y = values[1]
        if y <= 0:
            return [None, None]
        else:
            # DX = 2X/y, Dy = -||X||^2_2/y^2
            if self.args[0].is_complex():
                Dy = -(np.square(X.real) + np.square(X.imag)).sum()/np.square(y)
            else:
                Dy = -np.square(X).sum()/np.square(y)

            # Ensure Dy is a scalar for proper sparse array construction
            Dy = float(np.asarray(Dy).item() if np.asarray(Dy).ndim > 0 else Dy)
            Dy = sp.csc_array([[Dy]])
            DX = 2.0*X/y
            # Use F-order to match CVXPY's vectorization convention
            DX = np.reshape(DX, (self.args[0].size, 1), order='F')
            DX = scipy.sparse.csc_array(DX)
            return [DX, Dy]

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        # Disable DPP when the second argument is a parameter.
        if u.scopes.dpp_scope_active():
            return is_param_free(self.args[1])
        else:
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
        return (idx == 0) and self.args[idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return ((idx == 0) and self.args[idx].is_nonpos()) or (idx == 1)

    def validate_arguments(self) -> None:
        """Check dimensions of arguments.
        """
        if not self.args[1].is_scalar():
            raise ValueError("The second argument to quad_over_lin must be a scalar.")
        if self.args[1].is_complex():
            raise ValueError("The second argument to quad_over_lin cannot be complex.")
        # AxisAtom.validate_arguments handles axis validation
        super(quad_over_lin, self).validate_arguments()

    def is_quadratic(self) -> bool:
        """Quadratic if x is affine and y is constant.
        """
        return self.args[0].is_affine() and self.args[1].is_constant()

    def has_quadratic_term(self) -> bool:
        """A quadratic term if y is constant.
        """
        return self.args[1].is_constant()

    def conjugate(self, y, perspective_scale=1):
        """Fenchel conjugate of quad_over_lin(x, t) = ||x||^2 / t.

        When t is a fixed positive constant c:
        f(x) = ||x||^2 / c
        f*(y) = (c/4) ||y||^2

        Requires the denominator t to be a positive constant.
        """
        t = self.args[1]
        if not t.is_constant():
            raise NotImplementedError(
                "Fenchel conjugate of quad_over_lin requires constant denominator."
            )
        if t.value is not None and float(np.asarray(t.value).item()) <= 0:
            raise ValueError(
                "Fenchel conjugate of quad_over_lin requires positive denominator."
            )

        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))
        if scale.is_complex():
            raise ValueError(
                "Perspective scale for quad_over_lin conjugate must be real."
            )
        scale_is_constant = scale.is_constant() and scale.value is not None

        # Vector arguments with axis=0 behave like axis=None for this atom.
        if self.axis is None or len(self.args[0].shape) <= 1:
            if not scale.is_scalar():
                raise ValueError(
                    "Perspective scale for quad_over_lin conjugate must be scalar "
                    "when axis=None."
                )
            if scale_is_constant:
                # (s * ||x||^2 / t)^*(y) = t/(4s) * ||y||^2 = ||y||^2 / (4s/t), s >= 0.
                return quad_over_lin(y, (4.0 * scale) / t), []
            z = Variable(nonneg=True, name="z_quad_over_lin_conj")
            return (t / 4.0) * z, [quad_over_lin(y, scale) <= z]

        if self.axis not in (0, 1):
            raise NotImplementedError(
                "Fenchel conjugate of quad_over_lin currently supports axis in {0, 1}."
            )
        if len(y.shape) != 2:
            raise ValueError(
                "Fenchel conjugate of axis quad_over_lin expects matrix-valued y."
            )

        def _slice_count():
            return y.shape[1] if self.axis == 0 else y.shape[0]

        def _slice_expr(idx):
            if self.axis == 0:
                return y[:, idx]
            return y[idx, :]

        if scale.is_scalar():
            if scale_is_constant:
                # Sum slicewise conjugates for a scalar perspective scale.
                conj_vec = quad_over_lin(
                    y,
                    (4.0 * scale) / t,
                    axis=self.axis,
                    keepdims=self.keepdims,
                )
                return cp_sum(conj_vec), []
            conj_vec = quad_over_lin(y, scale, axis=self.axis, keepdims=self.keepdims)
            z = Variable(conj_vec.shape, nonneg=True, name="z_quad_over_lin_axis_conj")
            return (t / 4.0) * cp_sum(z), [conj_vec <= z]

        # Vector/matrix perspective scales are interpreted per reduced slice.
        scale_vec = reshape(scale, (scale.size,), order="F")
        num_slices = _slice_count()
        if scale_vec.size != num_slices:
            raise ValueError(
                "Perspective scale for axis quad_over_lin conjugate must have "
                "one entry per reduced slice."
            )

        if scale_is_constant:
            conj_expr = Constant(0.0)
            for idx in range(num_slices):
                conj_expr = conj_expr + quad_over_lin(
                    _slice_expr(idx),
                    (4.0 * scale_vec[idx]) / t,
                )
            return conj_expr, []

        z = Variable(num_slices, nonneg=True, name="z_quad_over_lin_axis_conj")
        constraints = []
        for idx in range(num_slices):
            constraints.append(
                quad_over_lin(_slice_expr(idx), scale_vec[idx]) <= z[idx]
            )
        return (t / 4.0) * cp_sum(z), constraints

    def _normalize_perspective_scale(self, perspective_scale):
        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))
        if scale.is_complex():
            raise ValueError(
                "Perspective scale for quad_over_lin conjugate must be real."
            )
        return scale

    def _sum_quad_over_lin_slices(self, y_x, scale_vec):
        if self.axis not in (0, 1):
            raise ValueError(
                "Axis slicewise quad_over_lin support requires axis in {0, 1}."
            )
        if len(y_x.shape) != 2:
            raise ValueError(
                "Axis slicewise quad_over_lin support requires a matrix argument."
            )

        num_slices = y_x.shape[1] if self.axis == 0 else y_x.shape[0]
        if scale_vec.size != num_slices:
            raise ValueError(
                "quad_over_lin full-conjugate scale must have one entry per slice."
            )

        total = Constant(0.0)
        for idx in range(num_slices):
            y_slice = y_x[:, idx] if self.axis == 0 else y_x[idx, :]
            total = total + quad_over_lin(y_slice, 4.0 * scale_vec[idx])
        return total

    def _full_conjugate_rhs(self, y_x, scale):
        if self.axis is None or len(y_x.shape) <= 1:
            if not scale.is_scalar():
                raise ValueError(
                    "quad_over_lin full-conjugate requires scalar scale when "
                    "axis=None."
                )
            return quad_over_lin(y_x, 4.0 * scale)

        if self.axis not in (0, 1):
            raise NotImplementedError(
                "quad_over_lin full-conjugate currently supports axis in {0, 1}."
            )
        if scale.is_scalar():
            return quad_over_lin(y_x, 4.0 * scale)

        scale_vec = reshape(scale, (scale.size,), order="F")
        return self._sum_quad_over_lin_slices(y_x, scale_vec)

    def _denom_only_weight(self, scale):
        """Scalar nonnegative weight B for B / t forms."""
        q_expr = quad_over_lin(
            self.args[0],
            1.0,
            axis=self.axis,
            keepdims=self.keepdims,
        )
        if q_expr.is_scalar():
            return scale * q_expr
        return cp_sum(multiply(scale, q_expr))

    def conjugate_term(self, nonconstant_arg_indices, dual_vars, perspective_scale=1):
        """Fenchel term conjugate for multi-argument dualization paths."""
        scale = self._normalize_perspective_scale(perspective_scale)

        if nonconstant_arg_indices == (0,):
            if len(dual_vars) != 1:
                raise ValueError(
                    "quad_over_lin unary term-conjugate expects one dual variable."
                )
            return self.conjugate(dual_vars[0], perspective_scale=scale)

        if nonconstant_arg_indices == (0, 1):
            if len(dual_vars) != 2:
                raise ValueError(
                    "quad_over_lin full term-conjugate expects two dual variables."
                )
            y_x, y_t = dual_vars
            if not y_t.is_scalar():
                raise ValueError("quad_over_lin full-conjugate requires scalar y_t.")
            if scale.is_nonneg() and scale.is_nonpos():
                return self.indicator_conjugate([y_x == 0, y_t <= 0])
            rhs = self._full_conjugate_rhs(y_x, scale)
            return self.indicator_conjugate([y_t + rhs <= 0])

        if nonconstant_arg_indices == (1,):
            if len(dual_vars) != 1:
                raise ValueError(
                    "quad_over_lin denominator-only term-conjugate expects "
                    "one dual variable."
                )
            y_t = dual_vars[0]
            if not y_t.is_scalar():
                raise ValueError(
                    "quad_over_lin(constant, t) conjugate requires scalar "
                    "dual variable."
                )
            weight = self._denom_only_weight(scale)
            if not weight.is_scalar():
                raise ValueError(
                    "quad_over_lin(constant, t) conjugate weight must be scalar."
                )
            if weight.is_nonneg() and weight.is_nonpos():
                return self.indicator_conjugate([y_t <= 0])

            constraints = [y_t <= 0]
            if not weight.is_nonneg():
                constraints.append(weight >= 0)
            return -2.0 * geo_mean(hstack([weight, -y_t])), constraints

        return super(quad_over_lin, self).conjugate_term(
            nonconstant_arg_indices,
            dual_vars,
            perspective_scale=scale,
        )

    def is_qpwa(self) -> bool:
        """Quadratic of piecewise affine if x is PWL and y is constant.
        """
        return self.args[0].is_pwl() and self.args[1].is_constant()
