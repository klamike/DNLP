"""Fenchel dual composition rules."""

from functools import singledispatch
from typing import Optional, Tuple

import numpy as np

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import DivExpression, MulExpression, multiply
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import Sum, sum as cp_sum
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.elementwise.exp import exp as exp_atom
from cvxpy.atoms.elementwise.logistic import logistic as logistic_atom
from cvxpy.atoms.elementwise.maximum import maximum as maximum_atom
from cvxpy.atoms.elementwise.minimum import minimum as minimum_atom
from cvxpy.atoms.log_sum_exp import log_sum_exp as log_sum_exp_atom
from cvxpy.atoms.max import max as max_atom
from cvxpy.atoms.sum_largest import sum_largest as sum_largest_atom
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.fenchel_ir import FenchelIR


def _to_expr(x):
    return x if isinstance(x, Expression) else Constant(np.asarray(x))


def _extract_uniform_constant_scalar(node: Expression) -> Optional[float]:
    if not node.is_constant() or node.value is None:
        return None
    arr = np.asarray(node.value)
    if arr.ndim == 0:
        return float(arr.item())
    if arr.size == 0:
        return None
    first = arr.flat[0]
    if np.allclose(arr, first):
        return float(first)
    return None


def _peel_scalar_factor(node: Expression) -> Optional[Tuple[Expression, float, Optional[Expression]]]:
    """If node is scalar-multiplication/division, return reduced child and factors."""

    def scalar_factor(constant_expr):
        if not (constant_expr.is_constant() and constant_expr.is_scalar()):
            return None
        if constant_expr.parameters():
            return 1.0, constant_expr
        if constant_expr.value is None:
            raise NotImplementedError("Scalar multipliers without values are not supported.")
        return float(np.asarray(constant_expr.value).item()), None

    if isinstance(node, (MulExpression, multiply)):
        lhs, rhs = node.args
        lhs_factor = scalar_factor(lhs)
        if lhs_factor is not None:
            numeric, symbolic = lhs_factor
            return rhs, numeric, symbolic
        rhs_factor = scalar_factor(rhs)
        if rhs_factor is not None:
            numeric, symbolic = rhs_factor
            return lhs, numeric, symbolic
        return None

    if isinstance(node, DivExpression):
        lhs, rhs = node.args
        if rhs.is_constant() and rhs.is_scalar() and rhs.parameters():
            return lhs, 1.0, 1 / rhs
        denom = _extract_uniform_constant_scalar(rhs)
        if denom is None:
            return None
        if denom == 0:
            raise ValueError("Division by zero in objective decomposition.")
        return lhs, 1.0 / denom, None

    return None


def _expr_sign(expr: Expression):
    if expr.is_nonneg() and expr.is_nonpos():
        return 0
    if expr.is_nonneg():
        return 1
    if expr.is_nonpos():
        return -1
    if expr.is_constant() and expr.value is not None:
        arr = np.asarray(expr.value)
        if np.all(arr >= 0):
            return 1 if np.any(arr > 0) else 0
        if np.all(arr <= 0):
            return -1 if np.any(arr < 0) else 0
    return None


def _broadcast_scale_to_shape(scale: Expression, target_shape):
    if scale.shape == target_shape:
        return scale
    if scale.is_scalar():
        if target_shape == ():
            return scale
        return multiply(Constant(np.ones(target_shape, dtype=float)), scale)
    try:
        np.broadcast_to(np.empty(scale.shape, dtype=np.dtype([])), shape=target_shape)
    except ValueError as exc:
        raise ValueError(
            f"Cannot broadcast multiplier shape {scale.shape} to target shape {target_shape}."
        ) from exc
    return multiply(Constant(np.ones(target_shape, dtype=float)), scale)


def _conjugate_objective_piece(sum_outputs, conj_expr):
    if sum_outputs:
        return cp_sum(conj_expr)
    if conj_expr.is_scalar():
        return conj_expr
    return cp_sum(conj_expr)


def _expand_multiplier_for_axis_sum(multiplier: Expression, arg_shape, axis, keepdims: bool):
    """Expand multiplier for cp.sum(arg, axis=...) so it matches arg's shape."""
    if axis is None:
        return multiplier

    if isinstance(axis, tuple):
        axis_tuple = axis
    else:
        axis_tuple = (axis,)
    ndim = len(arg_shape)
    axis_tuple = tuple(a if a >= 0 else a + ndim for a in axis_tuple)

    if keepdims:
        expanded = multiplier
    elif multiplier.is_scalar():
        expanded = multiplier
    else:
        out_shape = multiplier.shape
        reshape_shape = []
        out_idx = 0
        for dim in range(ndim):
            if dim in axis_tuple:
                reshape_shape.append(1)
            else:
                if out_idx >= len(out_shape):
                    raise ValueError("Multiplier shape is incompatible with summed expression.")
                reshape_shape.append(out_shape[out_idx])
                out_idx += 1
        if out_idx != len(out_shape):
            raise ValueError("Multiplier shape is incompatible with summed expression.")
        expanded = reshape(multiplier, tuple(reshape_shape), order="F")

    return _broadcast_scale_to_shape(expanded, arg_shape)


@singledispatch
def fenchel_dual_var_attributes(atom: Atom, nonconstant_arg_indices):
    return tuple({} for _ in tuple(nonconstant_arg_indices))


@fenchel_dual_var_attributes.register
def _(atom: exp_atom, nonconstant_arg_indices):
    if tuple(nonconstant_arg_indices) == (0,):
        return ({"nonneg": True},)
    return tuple({} for _ in tuple(nonconstant_arg_indices))


@fenchel_dual_var_attributes.register
def _(atom: logistic_atom, nonconstant_arg_indices):
    if tuple(nonconstant_arg_indices) == (0,):
        return ({"nonneg": True},)
    return tuple({} for _ in tuple(nonconstant_arg_indices))


@fenchel_dual_var_attributes.register
def _(atom: log_sum_exp_atom, nonconstant_arg_indices):
    if tuple(nonconstant_arg_indices) == (0,):
        return ({"nonneg": True},)
    return tuple({} for _ in tuple(nonconstant_arg_indices))


@fenchel_dual_var_attributes.register
def _(atom: max_atom, nonconstant_arg_indices):
    if tuple(nonconstant_arg_indices) == (0,):
        return ({"nonneg": True},)
    return tuple({} for _ in tuple(nonconstant_arg_indices))


@fenchel_dual_var_attributes.register
def _(atom: sum_largest_atom, nonconstant_arg_indices):
    if tuple(nonconstant_arg_indices) == (0,):
        return ({"nonneg": True},)
    return tuple({} for _ in tuple(nonconstant_arg_indices))


@fenchel_dual_var_attributes.register
def _(atom: maximum_atom, nonconstant_arg_indices):
    return tuple({"nonneg": True} for _ in tuple(nonconstant_arg_indices))


@fenchel_dual_var_attributes.register
def _(atom: minimum_atom, nonconstant_arg_indices):
    return tuple({"nonpos": True} for _ in tuple(nonconstant_arg_indices))


def _make_dual_vars(atom: Atom, nonconstant_arg_indices):
    attrs_seq = fenchel_dual_var_attributes(atom, nonconstant_arg_indices)
    if len(attrs_seq) != len(nonconstant_arg_indices):
        raise ValueError(
            f"fenchel_dual_var_attributes returned "
            f"{len(attrs_seq)} entries for {len(nonconstant_arg_indices)} non-constant args."
        )

    dual_vars = []
    for idx, attrs in zip(nonconstant_arg_indices, attrs_seq):
        arg = atom.args[idx]
        kwargs = dict(attrs)
        nonneg = bool(kwargs.get("nonneg", False))
        nonpos = bool(kwargs.get("nonpos", False))

        if arg.is_complex() and (nonneg or nonpos):
            raise NotImplementedError(
                f"Signed real dual variable requested for complex argument {idx} of {type(atom).__name__}."
            )

        if "complex" not in kwargs:
            kwargs["complex"] = arg.is_complex() and not (nonneg or nonpos)

        dual_vars.append(
            Variable(
                arg.shape,
                name=f"y_{type(atom).__name__}_{idx}" if len(nonconstant_arg_indices) > 1 else f"y_{type(atom).__name__}",
                **kwargs,
            )
        )
    return tuple(dual_vars)


def _add_weighted_affine_term(node: Expression, multiplier: Expression, ir: FenchelIR, *, sum_outputs=False, source=""):
    """Encode <multiplier, node> contribution, with node affine."""
    if sum_outputs:
        effective_multiplier = multiplier if multiplier.is_scalar() else cp_sum(multiplier)
        dual = _broadcast_scale_to_shape(effective_multiplier, node.shape)
        ir.add_coupling(dual, node, source=source)
        return

    if node.is_scalar():
        ir.add_coupling(multiplier, node, source=source)
        return

    dual = _broadcast_scale_to_shape(multiplier, node.shape)
    ir.add_coupling(dual, node, source=source)


def _add_weighted_constant_term(node: Expression, multiplier: Expression, ir: FenchelIR, *, sum_outputs=False):
    """Add scalar constant offset induced by weighted constant expression."""
    if sum_outputs:
        effective_multiplier = multiplier if multiplier.is_scalar() else cp_sum(multiplier)
        ir.add_objective(effective_multiplier * cp_sum(node))
        return

    if node.is_scalar():
        ir.add_objective(multiplier * node)
        return

    scale = _broadcast_scale_to_shape(multiplier, node.shape)
    ir.add_objective(cp_sum(multiply(scale, node)))


def _dualize_atom(node: Atom, multiplier: Expression, ir: FenchelIR, *, sum_outputs=False, source=""):
    nonconstant_arg_indices = tuple(idx for idx, arg in enumerate(node.args) if not arg.is_constant())
    if not nonconstant_arg_indices:
        _add_weighted_constant_term(node, multiplier, ir, sum_outputs=sum_outputs)
        return

    scale = _broadcast_scale_to_shape(multiplier, node.shape)
    sign = _expr_sign(scale)
    if sign is None:
        raise NotImplementedError(
            f"Composition multiplier for atom {type(node).__name__} must have known sign."
        )
    if sign == 0:
        return

    dual_vars = _make_dual_vars(node, nonconstant_arg_indices)
    if sign > 0:
        conj_expr, conj_constraints = node.conjugate_term(nonconstant_arg_indices, dual_vars, perspective_scale=scale)
    else:
        conj_expr, conj_constraints = node.negative_conjugate_term(nonconstant_arg_indices, dual_vars, perspective_scale=-scale)

    for con in conj_constraints:
        ir.add_constraint(con)

    conj_expr = _to_expr(conj_expr)
    ir.add_objective(-_conjugate_objective_piece(sum_outputs, conj_expr))

    for idx, dual_var in zip(nonconstant_arg_indices, dual_vars):
        dualize_weighted_expression(node.args[idx], dual_var, ir, source=f"arg:{type(node).__name__}[{idx}]")


def dualize_weighted_expression(node: Expression, multiplier, ir: FenchelIR, *, sum_outputs=False, source="objective"):
    """Recursively dualize multiplier * node (no lifting fallback)."""
    multiplier = _to_expr(multiplier)
    sign = _expr_sign(multiplier)
    if sign == 0:
        return

    if node.is_constant():
        _add_weighted_constant_term(node, multiplier, ir, sum_outputs=sum_outputs)
        return

    if isinstance(node, AddExpression):
        for arg in node.args:
            dualize_weighted_expression(arg, multiplier, ir, sum_outputs=sum_outputs, source=source)
        return

    if isinstance(node, NegExpression):
        dualize_weighted_expression(node.args[0], -multiplier, ir, sum_outputs=sum_outputs, source=source)
        return

    if isinstance(node, multiply):
        lhs, rhs = node.args
        if lhs.is_constant() and not rhs.is_constant():
            scaled_multiplier = multiply(
                _broadcast_scale_to_shape(multiplier, node.shape),
                _broadcast_scale_to_shape(lhs, node.shape),
            )
            dualize_weighted_expression(rhs, scaled_multiplier, ir, sum_outputs=sum_outputs, source=source)
            return
        if rhs.is_constant() and not lhs.is_constant():
            scaled_multiplier = multiply(
                _broadcast_scale_to_shape(multiplier, node.shape),
                _broadcast_scale_to_shape(rhs, node.shape),
            )
            dualize_weighted_expression(lhs, scaled_multiplier, ir, sum_outputs=sum_outputs, source=source)
            return

    if isinstance(node, DivExpression):
        lhs, rhs = node.args
        if rhs.is_constant() and not lhs.is_constant():
            scaled_multiplier = multiply(
                _broadcast_scale_to_shape(multiplier, node.shape),
                _broadcast_scale_to_shape(1 / rhs, node.shape),
            )
            dualize_weighted_expression(lhs, scaled_multiplier, ir, sum_outputs=sum_outputs, source=source)
            return

    peeled = _peel_scalar_factor(node)
    if peeled is not None:
        child, numeric_factor, symbolic_factor = peeled
        next_multiplier = numeric_factor * multiplier
        if symbolic_factor is not None:
            next_multiplier = next_multiplier * symbolic_factor
        dualize_weighted_expression(child, next_multiplier, ir, sum_outputs=sum_outputs, source=source)
        return

    if isinstance(node, Sum):
        if node.axis is None:
            dualize_weighted_expression(node.args[0], multiplier, ir, sum_outputs=True, source=source)
            return
        expanded_multiplier = _expand_multiplier_for_axis_sum(
            multiplier, node.args[0].shape, node.axis, node.keepdims
        )
        dualize_weighted_expression(node.args[0], expanded_multiplier, ir, source=source)
        return

    if node.is_affine():
        _add_weighted_affine_term(node, multiplier, ir, sum_outputs=sum_outputs, source=source)
        return

    if not isinstance(node, Atom):
        raise NotImplementedError(f"Unexpected expression type {type(node).__name__} in Fenchel composition.")

    _dualize_atom(node, multiplier, ir, sum_outputs=sum_outputs, source=source)
