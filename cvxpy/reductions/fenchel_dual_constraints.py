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

from functools import singledispatch

from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.nonpos import Inequality, NonNeg, NonPos
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.zero import Equality, Zero
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.fenchel_dual_terms import conjugate_objective_piece, decompose_objective, dualize_atom_term


def lower_variable_attributes(problem):
    """Lower variable attributes using CVXPY's standard attribute reduction."""
    lowered_problem, _ = CvxAttr2Constr(reduce_bounds=True).apply(problem)
    return lowered_problem


def _all_affine(args): return all(arg.is_affine() for arg in args)
def _all_affine_real(args): return all(arg.is_affine() and arg.is_real() for arg in args)


def _dualize_nonaffine_inequality(con, dual_obj_terms, dual_constraints, coupling_terms):
    """Dualize non-affine convex inequality using perspective-conjugate rules."""
    lam = Variable(con.expr.shape, name="lam_ineq", nonneg=True)
    atom_terms, affine_term, lifted_constraints = decompose_objective(con.expr, allow_lifting=True)

    for lifted in lifted_constraints:
        dualize_constraint_indicator(lifted, dual_obj_terms, dual_constraints, coupling_terms)

    for term in atom_terms:
        if term.coefficient == 0:
            continue
        conj_expr, conj_constraints, term_couplings = dualize_atom_term(term, perspective_multiplier=lam)
        dual_constraints.extend(conj_constraints)
        coupling_terms.extend(term_couplings)
        dual_obj_terms.append(-conjugate_objective_piece(term, conj_expr, allow_vector_entries=True))

    if affine_term is not None:
        coupling_terms.append((lam, affine_term))


def _append_or_extend(dual_constraints, cone_constraint):
    if isinstance(cone_constraint, (list, tuple)):
        dual_constraints.extend(cone_constraint)
    else:
        dual_constraints.append(cone_constraint)


def _add_affine_coupling(coupling_terms, expr, *, name, nonneg=False, **var_kwargs):
    if not expr.is_affine(): raise ValueError("Constraint expression must be affine.")
    dual_var = Variable(expr.shape, name=name, nonneg=nonneg, complex=(expr.is_complex() and not nonneg), **var_kwargs)
    coupling_terms.append((dual_var, expr))
    return dual_var


def _dualize_unary_cone_constraint(con, expr, *, name, dual_constraints, coupling_terms, nonneg=False, **var_kwargs):
    dual_var = _add_affine_coupling(coupling_terms, expr, name=name, nonneg=nonneg, **var_kwargs)
    _append_or_extend(dual_constraints, con._dual_cone(-dual_var))


def _dualize_multiarg_cone_constraint(con, exprs, *, names, dual_constraints, coupling_terms):
    exprs = tuple(exprs)
    if len(exprs) != len(names): raise ValueError("Dual variable names do not match cone argument count.")
    dual_vars = [
        _add_affine_coupling(coupling_terms, expr, name=name)
        for expr, name in zip(exprs, names)
    ]
    _append_or_extend(dual_constraints, con._dual_cone(*[-var for var in dual_vars]))


@singledispatch
def _dualize_constraint(con, _dual_obj_terms, dual_constraints, coupling_terms):
    exprs = None
    if hasattr(con, "expr"):
        exprs = (con.expr,)
    elif hasattr(con, "args"):
        exprs = tuple(con.args)
    if exprs and hasattr(con, "_dual_cone"):
        if not _all_affine(exprs):
            raise NotImplementedError(
                f"{type(con).__name__} is only supported with affine arguments."
            )
        if len(exprs) == 1:
            _dualize_unary_cone_constraint(con, exprs[0], name=f"nu_{type(con).__name__.lower()}", dual_constraints=dual_constraints, coupling_terms=coupling_terms)
            return
        names = tuple(f"nu_{type(con).__name__.lower()}_{i}" for i in range(len(exprs)))
        _dualize_multiarg_cone_constraint(con, exprs, names=names, dual_constraints=dual_constraints, coupling_terms=coupling_terms)
        return
    raise NotImplementedError(f"Unsupported constraint type for Fenchel dualization: {type(con).__name__}.")

@_dualize_constraint.register
def _(con: Inequality, dual_obj_terms, dual_constraints, coupling_terms):
    if con.expr.is_affine():
        _add_affine_coupling(coupling_terms, con.expr, name="lam_ineq", nonneg=True)
        return
    _dualize_nonaffine_inequality(con, dual_obj_terms, dual_constraints, coupling_terms)

@_dualize_constraint.register
def _(con: NonPos, _dual_obj_terms, dual_constraints, coupling_terms):
    _dualize_unary_cone_constraint(con, con.args[0], name="lam_nonpos",
                                   dual_constraints=dual_constraints, coupling_terms=coupling_terms)

@_dualize_constraint.register
def _(con: NonNeg, _dual_obj_terms, dual_constraints, coupling_terms):
    _dualize_unary_cone_constraint(con, con.args[0], name="lam_nonneg",
                                   dual_constraints=dual_constraints, coupling_terms=coupling_terms)

@_dualize_constraint.register
def _(con: Equality, _dual_obj_terms, dual_constraints, coupling_terms):
    _dualize_unary_cone_constraint(con, con.expr, name="nu_eq",
                                   dual_constraints=dual_constraints, coupling_terms=coupling_terms)

@_dualize_constraint.register
def _(con: Zero, _dual_obj_terms, dual_constraints, coupling_terms):
    _dualize_unary_cone_constraint(con, con.args[0], name="nu_zero",
                                   dual_constraints=dual_constraints, coupling_terms=coupling_terms)

@_dualize_constraint.register
def _(con: SOC, _dual_obj_terms, dual_constraints, coupling_terms):
    if not _all_affine(con.args): raise ValueError("SOC constraints must have affine arguments.")
    _dualize_multiarg_cone_constraint(con, con.args, names=("nu_soc_t", "nu_soc_x"),
                                      dual_constraints=dual_constraints, coupling_terms=coupling_terms)

@_dualize_constraint.register
def _(con: PSD, _dual_obj_terms, dual_constraints, coupling_terms):
    if not con.expr.is_affine(): raise ValueError("PSD constraints must be affine.")
    _dualize_unary_cone_constraint(con, con.expr, name="nu_psd",
                                   dual_constraints=dual_constraints, coupling_terms=coupling_terms, symmetric=True)

@_dualize_constraint.register
def _(con: ExpCone, _dual_obj_terms, dual_constraints, coupling_terms):
    if not _all_affine_real(con.args): raise ValueError("ExpCone constraints must have affine real arguments.")
    _dualize_multiarg_cone_constraint(con, con.args, names=("nu_exp_x", "nu_exp_y", "nu_exp_z"),
                                      dual_constraints=dual_constraints, coupling_terms=coupling_terms)

def dualize_constraint_indicator(con, dual_obj_terms, dual_constraints, coupling_terms):
    """Add dual terms induced by one primal constraint indicator."""
    _dualize_constraint(con, dual_obj_terms, dual_constraints, coupling_terms)
