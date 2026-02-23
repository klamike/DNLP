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

import numpy as np

import cvxpy.settings as s
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.conj import conj as cp_conj
from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.imag import imag as imag_atom
from cvxpy.atoms.affine.real import real as cp_real, real as real_atom
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import sum as cp_sum
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.expressions.constants import Constant
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem
from cvxpy.reductions.fenchel_dual_constraints import dualize_constraint_indicator, lower_variable_attributes
from cvxpy.reductions.fenchel_dual_terms import conjugate_objective_piece, decompose_objective, dualize_atom_term
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.utilities.coeff_extractor import CoeffExtractor
import cvxpy.lin_ops.lin_op as lo


def _validate_problem(problem):
    if not isinstance(problem.objective, (Minimize, Maximize)):
        raise ValueError("Fenchel dualization requires a Minimize or Maximize objective.")
    if not problem.is_dcp(): raise ValueError("Fenchel dualization requires a DCP problem.")
    if problem.parameters() and not problem.is_dpp(): raise NotImplementedError("Fenchel dualization requires a DPP problem.")
    if any(bool(v.attributes["boolean"]) or bool(v.attributes["integer"]) for v in problem.variables()):
        raise ValueError("Fenchel dualization supports only continuous variables; boolean/integer attributes are not supported.")


def _as_minimization_problem(problem):
    if isinstance(problem.objective, Minimize):
        return problem, False
    return Problem(Minimize(-problem.objective.expr), problem.constraints), True


def _prepare_problem(problem):
    """Lower variable attributes and reject unsupported constraint classes."""
    problem = lower_variable_attributes(problem)
    return problem


def _requires_scipy_coeff_backend(problem):
    leaves = problem.variables() + problem.parameters() + problem.constants()
    return any(leaf.is_complex() for leaf in leaves)


def _build_parameter_vector_expr(parameters, param_id_map):
    param_by_id = {param.id: param for param in parameters}
    parts = []
    for param_id, _ in sorted(param_id_map.items(), key=lambda kv: kv[1]):
        if param_id == lo.CONSTANT_ID:
            continue
        param = param_by_id[param_id]
        parts.append(reshape(param, (param.size,), order="F"))
    if parts:
        return hstack(parts + [Constant(np.array([1.0]))])
    return Constant(np.array([1.0]))


def _extract_adjoint(affine_expr, inverse_data, y_var, coeff_extractor, param_vec_expr):
    """For affine_expr = A @ x + b, return (A* @ y_var, <y_var, b>)."""
    if not affine_expr.is_affine(): raise ValueError("Adjoint extraction requires an affine expression.")
    if affine_expr.parameters() and not affine_expr.is_dpp(): raise NotImplementedError("Fenchel dualization with Parameters requires affine expressions that satisfy DPP.")

    x_length = inverse_data.x_length
    adj_dtype = complex if any(var.is_complex() for var in inverse_data.id2var.values()) else float
    zero_adj = Constant(np.zeros(x_length, dtype=adj_dtype))

    if affine_expr.is_constant():
        is_complex_pairing = affine_expr.is_complex() or y_var.is_complex()
        const_term = cp_sum(multiply(cp_conj(y_var), affine_expr)) if is_complex_pairing else cp_sum(multiply(y_var, affine_expr))
        return zero_adj, cp_real(const_term) if is_complex_pairing else const_term

    if isinstance(affine_expr, AddExpression):
        total_adj = zero_adj
        total_const = Constant(0.0)
        for arg in affine_expr.args:
            adj_i, const_i = _extract_adjoint(arg, inverse_data, y_var, coeff_extractor, param_vec_expr)
            total_adj = total_adj + adj_i
            total_const = total_const + const_i
        return total_adj, total_const

    if isinstance(affine_expr, NegExpression):
        adj_i, const_i = _extract_adjoint(affine_expr.args[0], inverse_data, y_var, coeff_extractor, param_vec_expr)
        return -adj_i, -const_i

    if isinstance(affine_expr, real_atom):
        return _extract_adjoint(affine_expr.args[0], inverse_data, y_var, coeff_extractor, param_vec_expr)

    if isinstance(affine_expr, imag_atom):
        return _extract_adjoint(affine_expr.args[0], inverse_data, 1j * y_var, coeff_extractor, param_vec_expr)

    problem_data_tensor = coeff_extractor.affine(affine_expr)
    flat_problem_data = Constant(problem_data_tensor) @ param_vec_expr
    matrix_with_offset = reshape(flat_problem_data, (affine_expr.size, x_length + 1), order="F")
    A_expr = matrix_with_offset[:, :x_length]
    affine_const = reshape(matrix_with_offset[:, x_length], affine_expr.shape, order="F")
    y_vec = reshape(y_var, (y_var.size,), order="F")
    is_complex_pairing = affine_expr.is_complex() or y_var.is_complex()
    adj_vec = A_expr.H @ y_vec if is_complex_pairing else A_expr.T @ y_vec
    const_term = cp_sum(multiply(cp_conj(y_var), affine_const)) if is_complex_pairing else cp_sum(multiply(y_var, affine_const))
    const_term = cp_real(const_term) if is_complex_pairing else const_term
    return adj_vec, const_term


def _dualize_objective(atom_terms):
    dual_obj_terms = []
    dual_constraints = []
    coupling_terms = []

    for term in atom_terms:
        if term.coefficient == 0: continue
        conj_expr, conj_constraints, term_couplings = dualize_atom_term(term)
        dual_constraints.extend(conj_constraints)
        coupling_terms.extend(term_couplings)
        dual_obj_terms.append(-conjugate_objective_piece(term, conj_expr))

    return dual_obj_terms, dual_constraints, coupling_terms


def _assemble_stationarity(coupling_terms, affine_term, inverse_data, coeff_extractor, param_vec_expr):
    x_length = inverse_data.x_length
    adj_dtype = complex if any(var.is_complex() for var in inverse_data.id2var.values()) else float
    total_adj = Constant(np.zeros(x_length, dtype=adj_dtype))
    total_const = Constant(0.0)

    for dual_var, affine_arg in coupling_terms:
        adj_i, const_i = _extract_adjoint(affine_arg, inverse_data, dual_var, coeff_extractor, param_vec_expr)
        total_adj = total_adj + adj_i
        total_const = total_const + const_i

    if affine_term is not None:
        adj_obj, const_obj = _extract_adjoint(affine_term, inverse_data, Constant(1.0), coeff_extractor, param_vec_expr)
        total_adj = total_adj + adj_obj
        total_const = total_const + const_obj

    if x_length == 0:
        return None, total_const
    return total_adj == 0, total_const


def _recover_primal_vars_from_stationarity(solution, inverse_data, stationarity_id, primal_var_ids):
    if stationarity_id is None:
        return {}
    stationarity_dual = solution.dual_vars.get(stationarity_id, None)
    if stationarity_dual is None:
        return {}
    stationarity_dual = -np.asarray(stationarity_dual).reshape(-1, order="F")
    if stationarity_dual.size != inverse_data.x_length:
        return {}

    primal_vars = {}
    for var_id in primal_var_ids:
        offset_size = inverse_data.id_map.get(var_id)
        if offset_size is None:
            continue
        offset, size = offset_size
        shape = inverse_data.var_shapes[var_id]
        value = np.reshape(stationarity_dual[offset:offset + size], shape, order="F")
        primal_vars[var_id] = value.item() if shape == () else value
    return primal_vars


def _build_dual_problem(problem, return_data=False):
    _validate_problem(problem)
    original_var_ids = tuple(var.id for var in problem.variables())
    problem, dual_is_minimization = _as_minimization_problem(problem)
    problem = _prepare_problem(problem)

    atom_terms, affine_term, lifted_constraints = decompose_objective(problem.objective.expr, allow_lifting=True)
    if lifted_constraints:
        problem = Problem(problem.objective, list(problem.constraints) + list(lifted_constraints))

    inverse_data = InverseData(problem)
    canon_backend = s.SCIPY_CANON_BACKEND if _requires_scipy_coeff_backend(problem) else None
    coeff_extractor = CoeffExtractor(inverse_data, canon_backend=canon_backend)
    param_vec_expr = _build_parameter_vector_expr(problem.parameters(), inverse_data.param_id_map)

    dual_obj_terms, dual_constraints, coupling_terms = _dualize_objective(atom_terms)

    for con in problem.constraints:
        dualize_constraint_indicator(con, dual_obj_terms, dual_constraints, coupling_terms)

    stationarity, total_const = _assemble_stationarity(coupling_terms, affine_term, inverse_data, coeff_extractor, param_vec_expr)
    if stationarity is not None:
        dual_constraints.append(stationarity)

    dual_obj = total_const
    for term_obj in dual_obj_terms:
        dual_obj = dual_obj + term_obj

    if not dual_obj.is_scalar():
        dual_obj = cp_sum(dual_obj)

    dual_problem = None
    if dual_is_minimization:
        dual_problem = Problem(Minimize(-dual_obj), dual_constraints)
    else:
        dual_problem = Problem(Maximize(dual_obj), dual_constraints)

    if not return_data:
        return dual_problem

    data = {
        "inverse_data": inverse_data,
        "stationarity_id": None if stationarity is None else stationarity.id,
        "primal_var_ids": original_var_ids,
    }
    return dual_problem, data


def fenchel_dual(problem):
    return _build_dual_problem(problem)


class FenchelDual(Reduction):
    """Reduction wrapper for atom-level Fenchel dualization."""

    def accepts(self, problem):
        try:
            _validate_problem(problem)
            min_problem, _ = _as_minimization_problem(problem)
            _prepare_problem(min_problem)
            return True
        except (ValueError, NotImplementedError, TypeError):
            return False

    def apply(self, problem):
        dual_problem, fenchel_data = _build_dual_problem(problem, return_data=True)
        return dual_problem, {"fenchel": fenchel_data}

    def invert(self, solution, inverse_data):
        if solution.status in s.SOLUTION_PRESENT:
            fenchel_data = inverse_data["fenchel"]
            primal_vars = _recover_primal_vars_from_stationarity(
                solution,
                fenchel_data["inverse_data"],
                fenchel_data["stationarity_id"],
                fenchel_data["primal_var_ids"],
            )
            return Solution(solution.status, solution.opt_val, primal_vars, {}, solution.attr)

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
