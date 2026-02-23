"""
Fenchel dualization reduction built from recursive composition rules.
"""

import numpy as np

import cvxpy.settings as s
from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.affine.conj import conj as conj_atom
from cvxpy.atoms.affine.real import real as real_atom
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import sum as cp_sum
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem
from cvxpy.reductions.fenchel_dual_constraints import (
    dualize_constraint_indicator,
    lower_variable_attributes,
)
from cvxpy.reductions.fenchel_ir import FenchelIR
from cvxpy.reductions.fenchel_rules import dualize_weighted_expression
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution, failure_solution


def _validate_problem(problem):
    if not isinstance(problem.objective, (Minimize, Maximize)):
        raise NotImplementedError("Fenchel dualization requires a Minimize or Maximize objective.")
    if not problem.is_dcp():
        raise ValueError("Fenchel dualization requires a DCP problem.")
    if problem.parameters() and not problem.is_dpp():
        raise NotImplementedError("Fenchel dualization requires DPP when Parameters are present.")
    if any(
        bool(v.attributes["boolean"]) or bool(v.attributes["integer"])
        for v in problem.variables()
    ):
        raise ValueError("Fenchel dualization supports only continuous variables.")


def _prepare_problem(problem):
    return lower_variable_attributes(problem)


# ---------------------------------------------------------------------------
#  Symbolic adjoint helpers
# ---------------------------------------------------------------------------

def _inner_product(y, z):
    """Scalar inner product <y, z>, handling complex with Re(y^H z)."""
    is_complex = y.is_complex() or z.is_complex()
    if is_complex:
        return real_atom(cp_sum(multiply(conj_atom(y), z)))
    return cp_sum(multiply(y, z))


def _merge_adj(d1, d2):
    """Merge two {var_id: Expression} dicts, summing contributions."""
    result = dict(d1)
    for vid, expr in d2.items():
        if vid in result:
            result[vid] = result[vid] + expr
        else:
            result[vid] = expr
    return result


# ---------------------------------------------------------------------------
#  Complex adjoint validation
# ---------------------------------------------------------------------------

def _validate_complex_adjoint(affine_expr, y_var, adjoint_pairs):
    """Validate that adjoint correctly handles complex Hermitian transpose.

    For complex affine expressions f(x) = Ax + b, the adjoint should satisfy:
    <y, f(x)> = <A^H y, x> + <y, b>

    where A^H is the Hermitian (conjugate) transpose.

    Parameters
    ----------
    affine_expr : Expression
        The affine atom whose adjoint is being validated.
    y_var : Variable
        The dual variable.
    adjoint_pairs : list of (int, Expression)
        The result from affine_expr.adjoint(y_var).

    Raises
    ------
    ValueError
        If adjoint appears to violate Hermitian transpose property.
    """
    # Basic structural check: adjoint should return valid pairs
    if not isinstance(adjoint_pairs, list):
        raise ValueError(
            f"{type(affine_expr).__name__}.adjoint() must return a list of (arg_idx, adj_expr) pairs."
        )

    # For complex expressions, check that returned adjoint expressions
    # have compatible types and are not obviously wrong
    for arg_idx, adj_y in adjoint_pairs:
        if not isinstance(arg_idx, int) or arg_idx < 0 or arg_idx >= len(affine_expr.args):
            raise ValueError(
                f"{type(affine_expr).__name__}.adjoint() returned invalid arg_idx: {arg_idx}"
            )

        if not isinstance(adj_y, Expression):
            raise ValueError(
                f"{type(affine_expr).__name__}.adjoint() must return Expression objects, "
                f"got {type(adj_y).__name__}"
            )

        # If original arg is complex but adjoint is not, or vice versa,
        # this might indicate missing conjugation
        arg = affine_expr.args[arg_idx]
        if arg.is_complex() and y_var.is_complex():
            # Both complex - adjoint should involve conjugation somewhere
            # EXCEPT for atoms that just pass through or distribute (like AddExpression)
            # For these, the conjugation happens at deeper levels in the tree
            from cvxpy.atoms.affine.add_expr import AddExpression
            from cvxpy.atoms.affine.binary_operators import MulExpression, multiply, DivExpression
            from cvxpy.atoms.affine.reshape import reshape
            from cvxpy.atoms.affine.sum import Sum

            # Atoms where passing through y_var is mathematically correct
            passthrough_atoms = (AddExpression, MulExpression, multiply, DivExpression, reshape, Sum)
            is_passthrough = isinstance(affine_expr, passthrough_atoms)

            if not is_passthrough and not _contains_conjugation(adj_y):
                # Check if this is an identity map (adj_y is literally y_var, reshaped)
                # Some atoms legitimately have identity adjoints
                from cvxpy.atoms.affine.wraps import Wrap
                from cvxpy.atoms.affine.index import index as index_class

                identity_atoms = (Wrap, index_class)
                is_identity_atom = isinstance(affine_expr, identity_atoms)

                if not is_identity_atom:
                    import warnings
                    warnings.warn(
                        f"{type(affine_expr).__name__}.adjoint() for complex arguments "
                        f"may be missing conjugation. For complex affine maps f(x) = Ax + b, "
                        f"the adjoint must use Hermitian (conjugate) transpose: "
                        f"<y, f(x)> = <A^H y, x> + <y, b>. "
                        f"If this is a custom atom, please verify the adjoint is correct.",
                        UserWarning, stacklevel=3
                    )


def _contains_conjugation(expr):
    """Check if an expression contains any conjugation operations."""
    from cvxpy.atoms.affine.conj import conj as conj_atom

    # Check if expr itself is a conj
    if isinstance(expr, conj_atom):
        return True

    # Recursively check arguments
    if hasattr(expr, 'args'):
        for arg in expr.args:
            if _contains_conjugation(arg):
                return True

    return False


# ---------------------------------------------------------------------------
#  Core symbolic adjoint walker
# ---------------------------------------------------------------------------

def _symbolic_adjoint(affine_expr, y_var, validate_complex=True):
    """Compute the adjoint of an affine expression symbolically.

    Given ``affine_expr(x) = A x + b`` and a dual vector *y_var* (same
    shape as the expression output), return per-variable adjoint
    contributions and the constant pairing.

    Parameters
    ----------
    affine_expr : Expression
        The affine expression to compute adjoint of.
    y_var : Variable
        The dual variable (same shape as affine_expr).
    validate_complex : bool, optional
        If True, validate that complex adjoints preserve anti-linearity.
        Default: True.

    Returns
    -------
    adj_dict : dict[int, Expression]
        Maps each primal variable id to its adjoint contribution
        ``(dexpr/dx_i)^H y``, with the same shape as the variable.
    const_term : Expression
        Scalar expression ``<y, b>`` (the constant-offset pairing).
    """
    # ---- Leaf: Variable ----
    if isinstance(affine_expr, Variable):
        adj = y_var
        if y_var.shape != affine_expr.shape:
            adj = reshape(adj, affine_expr.shape, order="F")
        return {affine_expr.id: adj}, Constant(0.0)

    # ---- Leaf: Constant / Parameter / any constant expression ----
    if affine_expr.is_constant():
        return {}, _inner_product(y_var, affine_expr)

    # ---- Dispatch via the atom's adjoint() method ----
    pairs = affine_expr.adjoint(y_var)

    # Validate complex adjoint if requested and expression is complex
    if validate_complex and (affine_expr.is_complex() or y_var.is_complex()):
        _validate_complex_adjoint(affine_expr, y_var, pairs)

    total_adj, total_const = {}, Constant(0.0)
    for arg_idx, adj_y in pairs:
        adj_i, const_i = _symbolic_adjoint(affine_expr.args[arg_idx], adj_y, validate_complex)
        total_adj = _merge_adj(total_adj, adj_i)
        total_const = total_const + const_i
    return total_adj, total_const


# ---------------------------------------------------------------------------
#  Coupling normalization (unchanged)
# ---------------------------------------------------------------------------

def _normalize_couplings_to_affine(ir: FenchelIR):
    """Recursively expand coupling terms until all coupling expressions are affine."""
    pending = list(ir.couplings)
    ir.couplings = []
    while pending:
        coupling = pending.pop()
        expr = coupling.affine_expr
        if expr.is_affine():
            ir.add_coupling(coupling.dual_var, expr, source=coupling.source)
            continue
        local_ir = FenchelIR()
        dualize_weighted_expression(
            expr, coupling.dual_var, local_ir, source=f"normalize:{coupling.source}"
        )
        ir.objective_terms.extend(local_ir.objective_terms)
        ir.constraints.extend(local_ir.constraints)
        pending.extend(local_ir.couplings)


# ---------------------------------------------------------------------------
#  Stationarity assembly (per-variable, no CoeffExtractor)
# ---------------------------------------------------------------------------

def _assemble_stationarity(ir: FenchelIR, primal_variables):
    """Compute per-variable stationarity constraints from all couplings.

    Returns
    -------
    stationarity_constraints : list[Constraint]
        One ``adj == 0`` constraint per primal variable.
    stationarity_map : dict[int, int]
        Constraint id → primal variable id (for primal recovery).
    total_const : Expression
        Scalar constant-offset contribution to the dual objective.
    """
    adj_per_var = {}  # var_id → accumulated adjoint Expression
    total_const = Constant(0.0)

    for coupling in ir.couplings:
        adj_i, const_i = _symbolic_adjoint(coupling.affine_expr, coupling.dual_var)
        for vid, adj_expr in adj_i.items():
            if vid in adj_per_var:
                adj_per_var[vid] = adj_per_var[vid] + adj_expr
            else:
                adj_per_var[vid] = adj_expr
        total_const = total_const + const_i

    stationarity_constraints = []
    stationarity_map = {}
    for var in primal_variables:
        vid = var.id
        if vid in adj_per_var:
            con = (adj_per_var[vid] == 0)
            stationarity_constraints.append(con)
            stationarity_map[con.id] = vid
        else:
            # Primal variable not appearing in any coupling.
            # Add zero-adjoint constraint to ensure primal recovery is complete.
            # This handles unconstrained variables or variables only in objective.
            zero_adj = Constant(np.zeros(var.shape))
            con = (zero_adj == 0)  # Trivially satisfied, but needed for recovery
            stationarity_constraints.append(con)
            stationarity_map[con.id] = vid

    return stationarity_constraints, stationarity_map, total_const


# ---------------------------------------------------------------------------
#  Primal recovery from stationarity duals
# ---------------------------------------------------------------------------

def _recover_primal_vars_from_stationarity(
    solution, stationarity_map, primal_var_ids, primal_var_shapes,
):
    """Recover primal variable values from duals of stationarity constraints."""
    if not stationarity_map:
        return {}

    primal_var_id_set = set(primal_var_ids)
    primal_vars = {}
    for con_id, var_id in stationarity_map.items():
        if var_id not in primal_var_id_set:
            continue
        dual_val = solution.dual_vars.get(con_id)
        if dual_val is None:
            continue
        shape = primal_var_shapes.get(var_id, ())
        value = -np.asarray(dual_val)
        if shape == ():
            value = value.item()
        else:
            value = np.reshape(value, shape, order="F")
        primal_vars[var_id] = value
    return primal_vars


# ---------------------------------------------------------------------------
#  Build the dual problem
# ---------------------------------------------------------------------------

def _build_dual_problem(problem, return_data=False):
    _validate_problem(problem)

    # Handle Maximize by converting to Minimize(-expr), dualizing, then
    # flipping the dual sense back.
    flipped = isinstance(problem.objective, Maximize)
    if flipped:
        problem = Problem(Minimize(-problem.objective.expr), problem.constraints)

    original_var_ids = tuple(var.id for var in problem.variables())
    original_var_shapes = {var.id: var.shape for var in problem.variables()}
    problem = _prepare_problem(problem)
    primal_variables = problem.variables()

    ir = FenchelIR()
    dualize_weighted_expression(
        problem.objective.expr, Constant(1.0), ir, source="objective"
    )

    for con in problem.constraints:
        dualize_constraint_indicator(con, ir)
    _normalize_couplings_to_affine(ir)

    stationarity_constraints, stationarity_map, total_const = _assemble_stationarity(
        ir, primal_variables
    )
    for con in stationarity_constraints:
        ir.add_constraint(con)

    dual_obj = total_const
    for term_obj in ir.objective_terms:
        dual_obj = dual_obj + term_obj

    if not dual_obj.is_scalar():
        dual_obj = cp_sum(dual_obj)

    # Validate dual objective curvature before creating problem
    # This catches atom conjugate bugs early
    if flipped:
        # Original was Maximize, so we minimized -expr, dual is Minimize(-dual_obj)
        # The negated dual objective should be convex
        negated_dual_obj = -dual_obj
        if not negated_dual_obj.is_convex():
            raise ValueError(
                "Dual objective is not convex after sign flip. "
                "This indicates an error in atom conjugate computation. "
                f"Dual objective: {dual_obj}, "
                f"Curvature: convex={dual_obj.is_convex()}, concave={dual_obj.is_concave()}"
            )
        dual_problem = Problem(Minimize(negated_dual_obj), ir.constraints)
    else:
        # Original was Minimize, dual is Maximize(dual_obj)
        # The dual objective should be concave
        if not dual_obj.is_concave():
            raise ValueError(
                "Dual objective is not concave. "
                "This indicates an error in atom conjugate computation. "
                f"Dual objective: {dual_obj}, "
                f"Curvature: convex={dual_obj.is_convex()}, concave={dual_obj.is_concave()}"
            )
        dual_problem = Problem(Maximize(dual_obj), ir.constraints)

    if not return_data:
        return dual_problem

    data = {
        "stationarity_map": stationarity_map,
        "primal_var_ids": original_var_ids,
        "primal_var_shapes": original_var_shapes,
        "flipped": flipped,
    }
    return dual_problem, data


def fenchel_dual(problem):
    return _build_dual_problem(problem)


class FenchelDual(Reduction):
    """Reduction wrapper for composition-based Fenchel dualization."""

    def accepts(self, problem):
        try:
            _validate_problem(problem)
            _prepare_problem(problem)
            return True
        except (ValueError, NotImplementedError, TypeError):
            return False

    def apply(self, problem):
        dual_problem, fenchel_data = _build_dual_problem(problem, return_data=True)
        return dual_problem, {"fenchel": fenchel_data}

    def invert(self, solution, inverse_data):
        fenchel_data = inverse_data["fenchel"]
        flipped = fenchel_data.get("flipped", False)

        if solution.status in s.SOLUTION_PRESENT:
            primal_vars = _recover_primal_vars_from_stationarity(
                solution,
                fenchel_data["stationarity_map"],
                fenchel_data["primal_var_ids"],
                fenchel_data["primal_var_shapes"],
            )
            opt_val = solution.opt_val
            if flipped and opt_val is not None:
                opt_val = -opt_val
            return Solution(solution.status, opt_val, primal_vars, {}, solution.attr)

        if solution.status == s.INFEASIBLE:
            return failure_solution(s.UNBOUNDED, solution.attr)
        if solution.status == s.INFEASIBLE_INACCURATE:
            return failure_solution(s.UNBOUNDED_INACCURATE, solution.attr)
        if solution.status == s.UNBOUNDED:
            return failure_solution(s.INFEASIBLE, solution.attr)
        if solution.status == s.UNBOUNDED_INACCURATE:
            return failure_solution(s.INFEASIBLE_INACCURATE, solution.attr)
        if solution.status == s.INFEASIBLE_OR_UNBOUNDED:
            return failure_solution(s.INFEASIBLE_OR_UNBOUNDED, solution.attr)

        return Solution(solution.status, solution.opt_val, {}, {}, solution.attr)
