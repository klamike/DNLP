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
import abc
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from cvxpy.constraints.constraint import Constraint

import numpy as np

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.settings as s
from cvxpy import interface as intf
from cvxpy import utilities as u
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.utilities import bounds as bounds_utils
from cvxpy.utilities import performance_utils as perf
from cvxpy.utilities.deterministic import unique_list


class Atom(Expression):
    """ Abstract base class for atoms. """
    _allow_complex = False
    # args are the expressions passed into the Atom constructor.

    def __init__(self, *args) -> None:
        self.id = lu.get_id()
        # Throws error if args is empty.
        if len(args) == 0:
            raise TypeError(
                "No arguments given to %s." % self.__class__.__name__
            )
        # Convert raw values to Constants.
        self.args = [Atom.cast_to_const(arg) for arg in args]
        self.validate_arguments()
        self._shape = self.shape_from_args()
        if not s.ALLOW_ND_EXPR and len(self._shape) > 2:
            raise ValueError("Atoms must be at most 2D.")
        super(Atom, self).__init__()

    def name(self) -> str:
        """Returns the string representation of the function call.
        """
        if self.get_data() is None:
            data = []
        else:
            data = [str(elem) for elem in self.get_data()]
        return f"{self.__class__.__name__}({', '.join([arg.name() for arg in self.args] + data)})"

    def _uses_default_name(self) -> bool:
        """Return True if this class uses Atom.name without override."""
        return type(self).name is Atom.name

    def format_labeled(self):
        """Format atom with labels, mirroring name() where safe.

        - If this atom or any ancestor has set a label, return it.
        - If the subclass didn't override name() (i.e., function-style default),
          mirror Atom.name but recurse with child.format_labeled() and include
          get_data() strings to preserve no-label parity.
        - Otherwise, fall back to Expression.format_labeled(); specialized
          subclasses with custom name() should implement their own
          format_labeled() to preserve custom syntax and precedence while
          recursing into children.
        """
        if self._label is not None:
            return self._label
        if self._uses_default_name():
            data = self.get_data()
            data_strs = [] if data is None else [str(elem) for elem in data]
            arg_text = [arg.format_labeled() for arg in self.args]
            return f"{type(self).__name__}({', '.join(arg_text + data_strs)})"
        # Defer to Expression default (label or name) when subclass has a custom name().
        return super().format_labeled()

    def validate_arguments(self) -> None:
        """Raises an error if the arguments are invalid.
        """
        if not self._allow_complex and any(arg.is_complex() for arg in self.args):
            raise ValueError(
                "Arguments to %s cannot be complex." % self.__class__.__name__
            )

    @abc.abstractmethod
    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape of the expression.
        """
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @abc.abstractmethod
    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        raise NotImplementedError()

    def bounds_from_args(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns bounds (lower, upper) of the expression based on argument bounds.

        Default implementation returns unbounded. Override in subclasses that can
        compute tighter bounds from their arguments.

        Returns
        -------
        tuple of np.ndarray
            (lower_bound, upper_bound) arrays with shape matching self.shape.
        """
        return bounds_utils.unbounded(self.shape)

    @perf.compute_once
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns bounds (lower, upper) of the expression.

        Combines bounds_from_args() with sign information for potentially tighter bounds.

        Returns
        -------
        tuple of np.ndarray
            (lower_bound, upper_bound) arrays with shape matching self.shape.
        """
        # Get bounds from argument propagation
        lb, ub = self.bounds_from_args()

        # Refine using sign information
        lb, ub = bounds_utils.refine_bounds_from_sign(lb, ub, self.is_nonneg(), self.is_nonpos())

        return (lb, ub)

    @perf.compute_once
    def is_nonneg(self) -> bool:
        """Is the expression nonnegative?
        """
        return self.sign_from_args()[0]

    @perf.compute_once
    def is_nonpos(self) -> bool:
        """Is the expression nonpositive?
        """
        return self.sign_from_args()[1]

    @perf.compute_once
    def is_imag(self) -> bool:
        """Is the expression imaginary?
        """
        # Default is false.
        return False

    @perf.compute_once
    def is_complex(self) -> bool:
        """Is the expression complex valued?
        """
        # Default is false.
        return False

    @abc.abstractmethod
    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        raise NotImplementedError()

    def is_atom_affine(self) -> bool:
        """Is the atom affine?
        """
        return self.is_atom_concave() and self.is_atom_convex()

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_atom_quasiconvex(self) -> bool:
        """Is the atom quasiconvex?
        """
        return self.is_atom_convex()

    def is_atom_quasiconcave(self) -> bool:
        """Is the atom quasiconcave?
        """
        return self.is_atom_concave()

    def is_atom_log_log_affine(self) -> bool:
        """Is the atom log-log affine?
        """
        return self.is_atom_log_log_concave() and self.is_atom_log_log_convex()

    @abc.abstractmethod
    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        raise NotImplementedError()

    @perf.compute_once
    def is_convex(self) -> bool:
        """Is the expression convex?
        """
        # Applies DCP composition rule.
        if self.is_constant():
            return True
        elif self.is_atom_convex():
            for idx, arg in enumerate(self.args):
                if not (arg.is_affine() or
                        (arg.is_convex() and self.is_incr(idx)) or
                        (arg.is_concave() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    @perf.compute_once
    def is_concave(self) -> bool:
        """Is the expression concave?
        """
        # Applies DCP composition rule.
        if self.is_constant():
            return True
        elif self.is_atom_concave():
            for idx, arg in enumerate(self.args):
                if not (arg.is_affine() or
                        (arg.is_concave() and self.is_incr(idx)) or
                        (arg.is_convex() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    def is_dpp(self, context='dcp') -> bool:
        """The expression is a disciplined parameterized expression.
        """
        if context.lower() == 'dcp':
            return self.is_dcp(dpp=True)
        elif context.lower() == 'dgp':
            return self.is_dgp(dpp=True)
        else:
            raise ValueError('Unsupported context ', context)

    @perf.compute_once
    def is_log_log_convex(self) -> bool:
        """Is the expression log-log convex?
        """
        # Verifies DGP composition rule.
        if self.is_log_log_constant():
            return True
        elif self.is_atom_log_log_convex():
            for idx, arg in enumerate(self.args):
                if not (arg.is_log_log_affine() or
                        (arg.is_log_log_convex() and self.is_incr(idx)) or
                        (arg.is_log_log_concave() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    @perf.compute_once
    def is_log_log_concave(self) -> bool:
        """Is the expression log-log concave?
        """
        # Verifies DGP composition rule.
        if self.is_log_log_constant():
            return True
        elif self.is_atom_log_log_concave():
            for idx, arg in enumerate(self.args):
                if not (arg.is_log_log_affine() or
                        (arg.is_log_log_concave() and self.is_incr(idx)) or
                        (arg.is_log_log_convex() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    @perf.compute_once
    def _non_const_idx(self) -> List[int]:
        return [i for i, arg in enumerate(self.args) if not arg.is_constant()]

    @perf.compute_once
    def _is_real(self) -> bool:
        # returns true if this atom is a real function:
        #   the atom must have exactly one argument that is not a constant
        #   that argument must be a scalar
        #   the output must be a scalar
        non_const = self._non_const_idx()
        return (self.is_scalar() and len(non_const) == 1 and
                self.args[non_const[0]].is_scalar())

    @perf.compute_once
    def is_quasiconvex(self) -> bool:
        """Is the expression quaisconvex?
        """
        from cvxpy.atoms.max import max as max_atom

        # Verifies the DQCP composition rule.
        if self.is_convex():
            return True
        if type(self) in (cvxtypes.maximum(), max_atom):
            return all(arg.is_quasiconvex() for arg in self.args)
        non_const = self._non_const_idx()
        if self._is_real() and self.is_incr(non_const[0]):
            return self.args[non_const[0]].is_quasiconvex()
        if self._is_real() and self.is_decr(non_const[0]):
            return self.args[non_const[0]].is_quasiconcave()
        if self.is_atom_quasiconvex():
            for idx, arg in enumerate(self.args):
                if not (arg.is_affine() or
                        (arg.is_convex() and self.is_incr(idx)) or
                        (arg.is_concave() and self.is_decr(idx))):
                    return False
            return True
        return False

    @perf.compute_once
    def is_quasiconcave(self) -> bool:
        """Is the expression quasiconcave?
        """
        from cvxpy.atoms.min import min as min_atom

        # Verifies the DQCP composition rule.
        if self.is_concave():
            return True
        if type(self) in (cvxtypes.minimum(), min_atom):
            return all(arg.is_quasiconcave() for arg in self.args)
        non_const = self._non_const_idx()
        if self._is_real() and self.is_incr(non_const[0]):
            return self.args[non_const[0]].is_quasiconcave()
        if self._is_real() and self.is_decr(non_const[0]):
            return self.args[non_const[0]].is_quasiconvex()
        if self.is_atom_quasiconcave():
            for idx, arg in enumerate(self.args):
                if not (arg.is_affine() or
                        (arg.is_concave() and self.is_incr(idx)) or
                        (arg.is_convex() and self.is_decr(idx))):
                    return False
            return True
        return False

    def canonicalize(self):
        """Represent the atom as an affine objective and conic constraints.
        """
        # Constant atoms are treated as a leaf.
        if self.is_constant() and not self.parameters():
            # Non-parameterized expressions are evaluated immediately.
            return Constant(self.value).canonical_form
        else:
            arg_objs = []
            constraints = []
            for arg in self.args:
                obj, constr = arg.canonical_form
                arg_objs.append(obj)
                constraints += constr
            # Special info required by the graph implementation.
            data = self.get_data()
            graph_obj, graph_constr = self.graph_implementation(arg_objs,
                                                                self.shape,
                                                                data)
            return graph_obj, constraints + graph_constr

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List['Constraint']]:
        """Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        raise NotImplementedError()

    @property
    def value(self):
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()

    def _value_impl(self):
        # shapes with 0's dropped in presolve.
        if 0 in self.shape:
            result = np.array([])
        else:
            arg_values = []
            for arg in self.args:
                # A argument without a value makes all higher level
                # values None.
                # But if the atom is constant with non-constant
                # arguments it doesn't depend on its arguments,
                # so it isn't None.
                arg_val = arg._value_impl()
                if arg_val is None and not self.is_constant():
                    return None
                else:
                    arg_values.append(arg_val)
            result = self.numeric(arg_values)
        return result

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.
        None indicates variable values unknown or outside domain.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        # Short-circuit to all zeros if known to be constant.
        if self.is_constant():
            return u.grad.constant_grad(self)

        # Returns None if variable values not supplied.
        arg_values = []
        for arg in self.args:
            if arg.value is None:
                return u.grad.error_grad(self)
            else:
                arg_values.append(arg.value)

        # A list of gradients w.r.t. arguments
        grad_self = self._grad(arg_values)
        # The Chain rule.
        result = {}
        for idx, arg in enumerate(self.args):
            # A dictionary of gradients w.r.t. variables
            # Partial argument / Partial x.
            grad_arg = arg.grad
            for key in grad_arg:
                # None indicates gradient is not defined.
                if grad_arg[key] is None or grad_self[idx] is None:
                    result[key] = None
                else:
                    if np.isscalar(grad_arg[key]) or np.isscalar(grad_self[idx]):
                        D = grad_arg[key] * grad_self[idx]
                    else:
                        D = grad_arg[key] @ grad_self[idx]
                    # Convert 1x1 matrices to scalars.
                    if not np.isscalar(D) and D.shape == (1, 1):
                        D = D[0, 0]

                    if key in result:
                        result[key] += D
                    else:
                        result[key] = D

        return result

    @abc.abstractmethod
    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        raise NotImplementedError()

    @property
    def domain(self) -> List['Constraint']:
        """A list of constraints describing the closure of the region
           where the expression is finite.
        """
        return self._domain() + [con for arg in self.args for con in arg.domain]

    def _domain(self) -> List['Constraint']:
        """Returns constraints describing the domain of the atom.
        """
        # Default is no constraints.
        return []

    @staticmethod
    def numpy_numeric(numeric_func):
        """Wraps an atom's numeric function that requires numpy ndarrays as input.
           Ensures both inputs and outputs are the correct matrix types.
        """

        def new_numeric(self, values):
            interface = intf.DEFAULT_INTF
            values = [interface.const_to_matrix(v, convert_scalars=True)
                      for v in values]
            result = numeric_func(self, values)
            return intf.DEFAULT_INTF.const_to_matrix(result)
        return new_numeric

    def conjugate(self, y, perspective_scale=1):
        """Returns tuple (`expr`, `constraints`) where `expr` is f*(y).

        When `perspective_scale` is provided, implementations may return
        ((perspective_scale * f)^*)(y). Atoms that do not support scaled
        perspective-conjugates should raise NotImplementedError.
        """
        if not self.is_atom_convex():
            self.raise_convex_only_conjugate_not_implemented(type(self).__name__)
        return self._generic_term_conjugate(
            nonconstant_arg_indices=(0,),
            dual_vars=(y,),
            perspective_scale=perspective_scale,
            negate_atom=False,
        )

    def negative_conjugate(self, y, perspective_scale=1):
        """Returns conjugate of -f for concave atoms.

        Implementations may return ((perspective_scale * (-f))^*)(y).
        """
        if not self.is_atom_concave():
            raise NotImplementedError(
                f"Fenchel conjugate of -{type(self).__name__} is not implemented."
            )
        return self._generic_term_conjugate(
            nonconstant_arg_indices=(0,),
            dual_vars=(y,),
            perspective_scale=perspective_scale,
            negate_atom=True,
        )

    def conjugate_term(self, nonconstant_arg_indices, dual_vars, perspective_scale=1):
        """Conjugate for objective terms formed from this atom.

        The default implementation supports unary terms where only the first
        atom argument is non-constant, i.e., term shape
            f(affine_arg0, constant_arg1, ...).
        """
        nonconstant_arg_indices = tuple(nonconstant_arg_indices)
        if len(dual_vars) != len(nonconstant_arg_indices):
            raise ValueError(
                f"{type(self).__name__}.conjugate_term expected one dual variable per "
                "non-constant argument."
            )
        if nonconstant_arg_indices == (0,) and len(dual_vars) == 1 and type(self).conjugate is not Atom.conjugate:
            try:
                return self.conjugate(dual_vars[0], perspective_scale=perspective_scale)
            except NotImplementedError:
                pass
        return self._generic_term_conjugate(
            nonconstant_arg_indices=nonconstant_arg_indices,
            dual_vars=dual_vars,
            perspective_scale=perspective_scale,
            negate_atom=False,
        )

    def negative_conjugate_term(
        self,
        nonconstant_arg_indices,
        dual_vars,
        perspective_scale=1,
    ):
        """Conjugate of -f for objective terms formed from this atom."""
        nonconstant_arg_indices = tuple(nonconstant_arg_indices)
        if len(dual_vars) != len(nonconstant_arg_indices):
            raise ValueError(
                f"{type(self).__name__}.negative_conjugate_term expected one dual "
                "variable per non-constant argument."
            )
        if nonconstant_arg_indices == (0,) and len(dual_vars) == 1 and type(self).negative_conjugate is not Atom.negative_conjugate:
            try:
                return self.negative_conjugate(
                    dual_vars[0],
                    perspective_scale=perspective_scale,
                )
            except NotImplementedError:
                pass
        return self._generic_term_conjugate(
            nonconstant_arg_indices=nonconstant_arg_indices,
            dual_vars=dual_vars,
            perspective_scale=perspective_scale,
            negate_atom=True,
        )

    @staticmethod
    def _fenchel_as_expr(expr):
        return expr if isinstance(expr, Expression) else Constant(np.asarray(expr))

    @staticmethod
    def _fenchel_flatten(expr):
        from cvxpy.atoms.affine.reshape import reshape
        return reshape(expr, (expr.size,), order="F")

    @staticmethod
    def _fenchel_broadcast_scale(scale, target_shape):
        from cvxpy.atoms.affine.binary_operators import multiply

        if scale.shape == target_shape:
            return scale
        if scale.is_scalar():
            if target_shape == ():
                return scale
            return multiply(Constant(np.ones(target_shape, dtype=float)), scale)
        raise ValueError(
            f"Perspective scale shape {scale.shape} is incompatible with target shape {target_shape}."
        )

    @staticmethod
    def _fenchel_slice(flat_var, offset, shape):
        from cvxpy.atoms.affine.reshape import reshape

        size = int(np.prod(shape, dtype=int)) if shape != () else 1
        segment = flat_var[offset:offset + size]
        if shape == ():
            return segment[0], offset + 1
        return reshape(segment, shape, order="F"), offset + size

    def _generic_term_conjugate(
        self,
        nonconstant_arg_indices,
        dual_vars,
        perspective_scale=1,
        negate_atom=False,
    ):
        from cvxpy.atoms.affine.hstack import hstack
        from cvxpy.expressions.variable import Variable
        from cvxpy.transforms.suppfunc import SuppFunc

        nonconstant_arg_indices = tuple(nonconstant_arg_indices)
        if len(dual_vars) != len(nonconstant_arg_indices):
            raise ValueError(
                f"{type(self).__name__} generic Fenchel term-conjugate expects one "
                "dual variable per non-constant argument."
            )
        if not nonconstant_arg_indices:
            raise NotImplementedError(
                f"{type(self).__name__} generic Fenchel term-conjugate requires at "
                "least one non-constant argument."
            )
        if any(idx < 0 or idx >= len(self.args) for idx in nonconstant_arg_indices):
            raise ValueError("Invalid non-constant argument index in Fenchel term-conjugate.")
        if any(self.args[idx].is_complex() for idx in nonconstant_arg_indices) or any(dv.is_complex() for dv in dual_vars):
            raise NotImplementedError(
                f"Generic Fenchel term-conjugate for {type(self).__name__} does not "
                "currently support complex non-constant arguments."
            )
        for idx, dual_var in zip(nonconstant_arg_indices, dual_vars):
            if dual_var.shape != self.args[idx].shape:
                raise ValueError(
                    f"Dual variable shape {dual_var.shape} does not match argument "
                    f"shape {self.args[idx].shape} for {type(self).__name__}."
                )

        scale = self._fenchel_as_expr(perspective_scale)
        if scale.is_complex() and not scale.is_real():
            raise NotImplementedError(
                f"Generic Fenchel term-conjugate for {type(self).__name__} does not "
                "support complex perspective scales."
            )
        scale = self._fenchel_broadcast_scale(scale, self.shape)
        if not scale.is_nonneg():
            raise ValueError(
                f"Perspective scale for {type(self).__name__} term-conjugate must be nonnegative."
            )

        total_dim = int(sum(self.args[idx].size for idx in nonconstant_arg_indices) + self.size)
        support_var = Variable((total_dim,))

        lifted_args = list(self.args)
        offset = 0
        for idx in nonconstant_arg_indices:
            lifted_arg, offset = self._fenchel_slice(support_var, offset, self.args[idx].shape)
            lifted_args[idx] = lifted_arg
        t_var, offset = self._fenchel_slice(support_var, offset, self.shape)
        if offset != total_dim:
            raise RuntimeError("Fenchel support-variable assembly has inconsistent dimensions.")

        lifted_atom = self.copy(args=lifted_args)
        param_map = {}
        for param in lifted_atom.parameters() + scale.parameters():
            if param.value is None:
                raise NotImplementedError(
                    f"Generic Fenchel term-conjugate for {type(self).__name__} requires "
                    "all Parameters to have numeric values."
                )
            param_map[id(param)] = Constant(np.asarray(param.value))
        if param_map:
            lifted_atom = lifted_atom.tree_copy(param_map)
            scale = scale.tree_copy(param_map)

        core_expr = -lifted_atom if negate_atom else lifted_atom
        if not core_expr.is_convex():
            op = "(-atom)" if negate_atom else "atom"
            raise NotImplementedError(
                f"Generic Fenchel term-conjugate requires convex {op} after fixing "
                f"constant arguments for {type(self).__name__}."
            )

        set_constraints = [core_expr <= t_var]
        set_constraints.extend(core_expr.domain)

        support = SuppFunc(support_var, set_constraints)
        direction_parts = [self._fenchel_flatten(dv) for dv in dual_vars]
        direction_parts.append(-self._fenchel_flatten(scale))
        direction = hstack(direction_parts)
        return support(direction), []

    @staticmethod
    def indicator_conjugate(constraints):
        """Return an indicator-form conjugate represented by explicit constraints."""
        return 0, constraints

    @staticmethod
    def raise_convex_only_conjugate_not_implemented(atom_name):
        raise NotImplementedError(
            "Fenchel conjugate is currently defined only for convex atoms; "
            f"{atom_name} is concave."
        )

    def atoms(self) -> List['Atom']:
        """A list of the atom types present amongst this atom's arguments.
        """
        atom_list = []
        for arg in self.args:
            atom_list += arg.atoms()
        return unique_list(atom_list + [type(self)])
