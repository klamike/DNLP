"""
Constraint indicator dualization for Fenchel dual reduction.
"""

from functools import singledispatch

from cvxpy.constraints.cones import Cone
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.nonpos import Inequality, NonNeg, NonPos
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.zero import Equality, Zero
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.fenchel_ir import FenchelIR
from cvxpy.reductions.fenchel_rules import dualize_weighted_expression


def lower_variable_attributes(problem):
    """Lower variable attributes using CVXPY's standard attribute reduction."""
    lowered_problem, _ = CvxAttr2Constr(reduce_bounds=True).apply(problem)
    return lowered_problem


def _add_affine_coupling(ir: FenchelIR, expr, *, name, nonneg=False, **var_kwargs):
    if not expr.is_affine():
        raise ValueError("Constraint expression must be affine.")
    dual_var = Variable(
        expr.shape,
        name=name,
        nonneg=nonneg,
        complex=(expr.is_complex() and not nonneg),
        **var_kwargs,
    )
    ir.add_coupling(dual_var, expr, source=f"constraint:{name}")
    return dual_var


def _append_or_extend(ir: FenchelIR, cone_constraint):
    if isinstance(cone_constraint, (list, tuple)):
        for con in cone_constraint:
            ir.add_constraint(con)
        return
    ir.add_constraint(cone_constraint)


def _default_dual_var_name(con, index: int, total: int):
    base = type(con).__name__.lower()
    return f"nu_{base}" if total == 1 else f"nu_{base}_{index}"


def _dualize_cone(con: Cone, ir: FenchelIR, *, names=None, attrs=None):
    exprs = tuple(con.args)
    if not exprs:
        raise NotImplementedError(f"Unsupported cone type for Fenchel dualization: {type(con).__name__}.")
    if not all(expr.is_affine() for expr in exprs):
        raise NotImplementedError(f"{type(con).__name__} is only supported with affine arguments.")

    dual_vars = []
    total = len(exprs)
    for i, expr in enumerate(exprs):
        name = names[i] if names is not None else _default_dual_var_name(con, i, total)
        kwargs = attrs[i] if attrs is not None else {}
        dual_vars.append(_add_affine_coupling(ir, expr, name=name, **kwargs))

    try:
        dual_cone = con._dual_cone(*[-var for var in dual_vars])
    except (NotImplementedError, TypeError) as exc:
        raise NotImplementedError(
            f"Dual cone not implemented for {type(con).__name__}; Fenchel dualization does not support this cone yet."
        ) from exc
    _append_or_extend(ir, dual_cone)


@singledispatch
def dualize_constraint_indicator(con: Constraint, ir: FenchelIR):
    raise NotImplementedError(f"Unsupported constraint type for Fenchel dualization: {type(con).__name__}.")

@dualize_constraint_indicator.register
def _(con: Cone, ir: FenchelIR):
    _dualize_cone(con, ir)

@dualize_constraint_indicator.register
def _(con: SOC, ir: FenchelIR):
    _dualize_cone(con, ir, names=("nu_soc_t", "nu_soc_x"))

@dualize_constraint_indicator.register
def _(con: PSD, ir: FenchelIR):
    _dualize_cone(con, ir, names=("nu_psd",), attrs=({"symmetric": True},))

@dualize_constraint_indicator.register
def _(con: ExpCone, ir: FenchelIR):
    if any(not arg.is_real() for arg in con.args):
        raise ValueError("ExpCone constraints must have real arguments.")
    _dualize_cone(con, ir, names=("nu_exp_x", "nu_exp_y", "nu_exp_z"))

@dualize_constraint_indicator.register
def _(con: Inequality, ir: FenchelIR):
    expr = con.expr
    if expr.is_affine():
        _add_affine_coupling(ir, expr, name="lam_ineq", nonneg=True)
        return
    lam = Variable(expr.shape, name="lam_ineq", nonneg=True)
    dualize_weighted_expression(expr, lam, ir, source="ineq")

@dualize_constraint_indicator.register
def _(con: NonPos, ir: FenchelIR):
    expr = con.args[0]
    if expr.is_affine():
        _add_affine_coupling(ir, expr, name="lam_nonpos", nonneg=True)
        return
    lam = Variable(expr.shape, name="lam_nonpos", nonneg=True)
    dualize_weighted_expression(expr, lam, ir, source="nonpos")

@dualize_constraint_indicator.register
def _(con: NonNeg, ir: FenchelIR):
    expr = con.args[0]
    if expr.is_affine():
        _add_affine_coupling(ir, -expr, name="lam_nonneg", nonneg=True)
        return
    lam = Variable(expr.shape, name="lam_nonneg", nonneg=True)
    dualize_weighted_expression(-expr, lam, ir, source="nonneg")

@dualize_constraint_indicator.register
def _(con: Equality, ir: FenchelIR):
    _add_affine_coupling(ir, con.expr, name="nu_eq")

@dualize_constraint_indicator.register
def _(con: Zero, ir: FenchelIR):
    _add_affine_coupling(ir, con.args[0], name="nu_zero")
