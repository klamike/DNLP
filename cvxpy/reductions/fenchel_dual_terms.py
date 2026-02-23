"""
Copyright 2026 Michael Klamkin

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

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import DivExpression, MulExpression, multiply
from cvxpy.atoms.affine.sum import Sum, sum as cp_sum
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable


@dataclass
class AtomTerm:
    """A term of the form coeff * f(affine_args) in the objective."""

    atom: Atom
    affine_args: tuple
    nonconstant_arg_indices: tuple
    coefficient: float = 1.0
    scalar_multiplier: Expression = None
    sum_outputs: bool = False


def _extract_uniform_constant_scalar(node: Expression) -> Optional[float]:
    """Return scalar value if constant expression is uniform; else None."""
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
            raise NotImplementedError(
                "Scalar multipliers without values are not supported for Fenchel dualization."
            )
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
        if denom == 0: raise ValueError("Division by zero in objective decomposition.")
        return lhs, 1.0 / denom, None

    return None


def _multiplier_sign(coeff, scalar_multiplier):
    if coeff == 0:
        return 0
    sign = 1 if coeff > 0 else -1
    if scalar_multiplier is None:
        return sign
    if scalar_multiplier.is_nonneg() and scalar_multiplier.is_nonpos():
        return 0
    if scalar_multiplier.is_nonneg():
        return sign
    if scalar_multiplier.is_nonpos():
        return -sign
    return None


def _lift_nonaffine_atom(node, coeff, scalar_multiplier, lifted_constraints):
    sign = _multiplier_sign(coeff, scalar_multiplier)
    if sign is None:
        raise NotImplementedError(
            f"Cannot lift nested non-affine atom {type(node).__name__} with unknown multiplier sign."
        )
    if sign == 0:
        return Constant(np.zeros(node.shape))

    if sign > 0 and node.is_convex():
        is_incr = node.is_incr
        is_decr = node.is_decr
    elif sign < 0 and node.is_concave():
        # For a negative coefficient, objective sees the convex atom -node.
        is_incr = node.is_decr
        is_decr = node.is_incr
    else:
        raise NotImplementedError(
            f"Cannot lift nested non-affine atom {type(node).__name__} for this objective sign/curvature."
        )

    new_args = list(node.args)
    for idx, arg in enumerate(node.args):
        if arg.is_affine():
            continue
        if not arg.is_real():
            raise NotImplementedError(
                f"Cannot lift complex non-affine argument {idx} of {type(node).__name__}."
            )
        aux = Variable(arg.shape, name=f"lift_{type(node).__name__}_{idx}")
        if is_incr(idx):
            if not arg.is_convex():
                raise NotImplementedError(
                    f"Cannot lift argument {idx} of {type(node).__name__}: expected convex due to monotonicity."
                )
            lifted_constraints.append(arg <= aux)
        elif is_decr(idx):
            if not arg.is_concave():
                raise NotImplementedError(
                    f"Cannot lift argument {idx} of {type(node).__name__}: expected concave due to monotonicity."
                )
            lifted_constraints.append(arg >= aux)
        else:
            raise NotImplementedError(
                f"Cannot lift non-affine argument {idx} of nonmonotone atom {type(node).__name__}."
            )
        new_args[idx] = aux

    lifted = node.copy(args=new_args)
    if not all(arg.is_affine() for arg in lifted.args):
        raise NotImplementedError(
            f"Failed to lift nested non-affine atom {type(node).__name__}."
        )
    return lifted


def decompose_objective(expr, allow_lifting=True):
    """Decompose objective into atom terms and affine remainder.

    Supports sums of affine pieces and atom terms of the form
        coeff * atom(affine_args)
    plus outer full reductions `cp.sum(...)` on such terms.
    """
    atom_terms = []
    affine_parts = []
    lifted_constraints = []

    def append_affine(node, coeff, scalar_multiplier, sum_outputs):
        part = cp_sum(node) if sum_outputs else node
        if scalar_multiplier is not None:
            part = scalar_multiplier * part
        if coeff != 1.0:
            part = coeff * part
        affine_parts.append(part)

    def _walk(node, coeff=1.0, scalar_multiplier=None, sum_outputs=False):
        if coeff == 0:
            return

        if node.is_constant():
            append_affine(node, coeff, scalar_multiplier, sum_outputs)
            return

        if isinstance(node, AddExpression):
            for arg in node.args:
                _walk(arg, coeff, scalar_multiplier, sum_outputs)
            return

        if isinstance(node, NegExpression):
            _walk(node.args[0], -coeff, scalar_multiplier, sum_outputs)
            return

        peeled = _peel_scalar_factor(node)
        if peeled is not None:
            child, numeric_factor, symbolic_factor = peeled
            if symbolic_factor is not None:
                if scalar_multiplier is None:
                    scalar_multiplier = symbolic_factor
                else:
                    scalar_multiplier = scalar_multiplier * symbolic_factor
            _walk(child, coeff * numeric_factor, scalar_multiplier, sum_outputs)
            return

        if isinstance(node, Sum):
            # Axis reductions are handled via perspective scaling in constraint
            # dualization: <lam, sum_axis(v)> = sum(broadcast(lam) * v).
            _walk(node.args[0], coeff, scalar_multiplier, True)
            return

        if node.is_affine():
            append_affine(node, coeff, scalar_multiplier, sum_outputs)
            return

        if isinstance(node, Atom):
            if not all(arg.is_affine() for arg in node.args):
                if not allow_lifting:
                    raise NotImplementedError(
                        f"Cannot decompose nested non-affine atom {type(node).__name__} "
                        "for Fenchel dualization; reformulate with explicit epigraph variables."
                    )
                _walk(_lift_nonaffine_atom(node, coeff, scalar_multiplier, lifted_constraints), coeff, scalar_multiplier, sum_outputs)
                return
            nonconstant_arg_indices = tuple(
                idx for idx, arg in enumerate(node.args)
                if not arg.is_constant()
            )
            if not nonconstant_arg_indices:
                append_affine(node, coeff, scalar_multiplier, sum_outputs)
                return
            atom_terms.append(AtomTerm(
                atom=node, affine_args=tuple(node.args[idx] for idx in nonconstant_arg_indices),
                nonconstant_arg_indices=nonconstant_arg_indices, coefficient=coeff,
                scalar_multiplier=scalar_multiplier, sum_outputs=sum_outputs,
            ))
            return

        raise NotImplementedError(
            f"Unexpected expression type {type(node).__name__} during "
            "Fenchel decomposition."
        )

    _walk(expr)

    affine_term = None
    for part in affine_parts:
        affine_term = part if affine_term is None else affine_term + part
    return atom_terms, affine_term, lifted_constraints


def _term_scale(term, perspective_multiplier=None):
    """Build total scalar multiplier for one atom term."""
    scale = perspective_multiplier if perspective_multiplier is not None else Constant(1.0)
    if term.scalar_multiplier is not None:
        scale = scale * term.scalar_multiplier
    if term.coefficient != 1.0:
        scale = term.coefficient * scale
    return scale


def _broadcast_scale_to_shape(scale, target_shape):
    if scale.is_scalar() or scale.shape == target_shape:
        return scale
    try:
        np.broadcast_to(np.empty(scale.shape, dtype=np.dtype([])), shape=target_shape)
    except ValueError as exc:
        raise ValueError(
            f"Cannot broadcast multiplier shape {scale.shape} to atom output shape "
            f"{target_shape}."
        ) from exc
    return multiply(Constant(np.ones(target_shape, dtype=float)), scale)


def _dual_vars_for_term(term):
    """Create one dual variable for each non-constant affine atom argument."""
    atom_name = term.atom.__class__.__name__
    unary_first_arg = len(term.affine_args) == 1 and term.nonconstant_arg_indices == (0,)
    return tuple(Variable(affine_arg.shape,
                          name=(f"y_{atom_name}" if unary_first_arg else f"y_{atom_name}_{arg_idx}"),
                          complex=affine_arg.is_complex())
                 for arg_idx, affine_arg in zip(term.nonconstant_arg_indices, term.affine_args))


def dualize_atom_term(term, perspective_multiplier=None):
    """Dualize one decomposed objective term and return conjugate + couplings."""
    if perspective_multiplier is not None and not perspective_multiplier.is_nonneg():
        raise ValueError("Perspective-conjugate dualization requires nonnegative multipliers.")

    dual_vars = _dual_vars_for_term(term)
    scale = _term_scale(term, perspective_multiplier=perspective_multiplier)
    scale = _broadcast_scale_to_shape(scale, term.atom.shape)

    if scale.is_complex() and not scale.is_real():
        raise NotImplementedError(
            "Complex multipliers for nonlinear atom dualization are not supported."
        )

    atom = term.atom
    if scale.is_nonneg():
        conjugate_fn = atom.conjugate_term
        perspective_scale = scale
    elif scale.is_nonpos():
        conjugate_fn = atom.negative_conjugate_term
        perspective_scale = -scale
    else:
        raise NotImplementedError(
            "Parameterized scalar multipliers for nonlinear atoms must have "
            "known sign (nonnegative or nonpositive)."
        )
    conj_expr, conj_constraints = conjugate_fn(term.nonconstant_arg_indices, dual_vars, perspective_scale=perspective_scale)

    if not isinstance(conj_expr, Expression):
        conj_expr = Constant(conj_expr)

    couplings = list(zip(dual_vars, term.affine_args))
    return conj_expr, conj_constraints, couplings


def conjugate_objective_piece(term, conj_expr, allow_vector_entries=False):
    """Convert a conjugate expression into a scalar objective contribution."""
    if term.sum_outputs:
        return cp_sum(conj_expr)
    if conj_expr.is_scalar():
        return conj_expr
    if allow_vector_entries:
        return cp_sum(conj_expr)
    return cp_sum(conj_expr)
