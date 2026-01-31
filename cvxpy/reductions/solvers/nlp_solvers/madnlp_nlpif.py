import numpy as np

import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import NLPsolver
from cvxpy.utilities.citations import CITATION_DICT


class MadNLPProblem:
    def __init__(self, data, libMad, oracles):
        self.libMad = libMad
        self.m, self.n = len(data["cl"]), len(data["x0"])
        x0, lvar, uvar, lcon, ucon = data["x0"], data["lb"], data["ub"], data["cl"], data["cu"]
        y0 = np.zeros(self.m)  # TODO: check default init of madnlp

        self.callbacks = self._create_callbacks(self.libMad, self.n, self.m, oracles)
        self.nlp_ptr = self.libMad.create_nlpmodel(self.callbacks)
        self.libMad.set_numerics(self.nlp_ptr, self.n, self.m, x0, y0, lvar, uvar, lcon, ucon)

    @staticmethod
    def _create_callbacks(libMad, n, m, oracles):
        jac_row, jac_col = oracles.jacobianstructure()
        nnzj = len(jac_row)
        jac_row = np.array(jac_row, dtype=np.int64) + 1  # julia is 1-based
        jac_col = np.array(jac_col, dtype=np.int64) + 1  # julia is 1-based

        hess_row, hess_col = oracles.hessianstructure()
        nnzh = len(hess_row)
        hess_row = np.array(hess_row, dtype=np.int64) + 1  # julia is 1-based
        hess_col = np.array(hess_col, dtype=np.int64) + 1  # julia is 1-based

        @libMad.NlpConstrJacStructure
        def jac_struct(row_ptr, col_ptr, user_data):
            for i, (r, c) in enumerate(zip(jac_row, jac_col)):
                row_ptr[i], col_ptr[i] = r, c
            return 0

        @libMad.NlpLagHessStructure
        def hess_struct(row_ptr, col_ptr, user_data):
            for i, (r, c) in enumerate(zip(hess_row, hess_col)):
                row_ptr[i], col_ptr[i] = r, c
            return 0

        @libMad.NlpEvalObj
        def eval_f(x_ptr, f_ptr, user_data):
            x = np.array([x_ptr[i] for i in range(n)])
            f_ptr[0] = oracles.objective(x)
            return 0

        @libMad.NlpEvalConstr
        def eval_g(x_ptr, g_ptr, user_data):
            x = np.array([x_ptr[i] for i in range(n)])
            g = oracles.constraints(x)
            for i in range(m): g_ptr[i] = g[i]
            return 0

        @libMad.NlpEvalObjGrad
        def eval_grad_f(x_ptr, grad_ptr, user_data):
            x = np.array([x_ptr[i] for i in range(n)])
            grad = oracles.gradient(x)
            for i in range(n): grad_ptr[i] = grad[i]
            return 0

        @libMad.NlpEvalConstrJac
        def eval_jac_g(x_ptr, jac_ptr, user_data):
            x = np.array([x_ptr[i] for i in range(n)])
            jac = oracles.jacobian(x)
            if isinstance(jac, memoryview):
                jac_array = np.frombuffer(jac, dtype=np.float64)
            else:
                jac_array = np.array(jac).flatten()
            for i in range(nnzj): jac_ptr[i] = jac_array[i]
            return 0

        @libMad.NlpEvalLagHess
        def eval_h(obj_factor, x_ptr, lambda_ptr, hess_ptr, user_data):
            x = np.array([x_ptr[i] for i in range(n)])
            lam = np.array([lambda_ptr[i] for i in range(m)])
            hess = oracles.hessian(x, lam, obj_factor)
            if isinstance(hess, memoryview):
                hess_array = np.frombuffer(hess, dtype=np.float64)
            else:
                hess_array = np.array(hess).flatten()
            for i in range(nnzh): hess_ptr[i] = hess_array[i]
            return 0

        callbacks = {}
        callbacks['jac_struct'] = jac_struct
        callbacks['hess_struct'] = hess_struct
        callbacks['eval_f'] = eval_f
        callbacks['eval_g'] = eval_g
        callbacks['eval_grad_f'] = eval_grad_f
        callbacks['eval_jac_g'] = eval_jac_g
        callbacks['eval_h'] = eval_h
        callbacks['meta'] = {'nnzj': nnzj, 'nnzh': nnzh, 'n': n, 'm': m}

        return callbacks


class MADNLP(NLPsolver):
    STATUS_MAP = {
        1: s.OPTIMAL,                    # SOLVE_SUCCEEDED
        2: s.OPTIMAL_INACCURATE,         # SOLVED_TO_ACCEPTABLE_LEVEL
        3: s.SOLVER_ERROR,               # SEARCH_DIRECTION_BECOMES_TOO_SMALL
        4: s.UNBOUNDED,                  # DIVERGING_ITERATES
        5: s.INFEASIBLE,                 # INFEASIBLE_PROBLEM_DETECTED
        6: s.USER_LIMIT,                 # MAXIMUM_ITERATIONS_EXCEEDED
        7: s.USER_LIMIT,                 # MAXIMUM_WALLTIME_EXCEEDED
        -1: s.SOLVER_ERROR,              # RESTORATION_FAILED
        -2: s.SOLVER_ERROR,              # INVALID_NUMBER_DETECTED
        -3: s.SOLVER_ERROR,              # ERROR_IN_STEP_COMPUTATION
        -4: s.SOLVER_ERROR,              # NOT_ENOUGH_DEGREES_OF_FREEDOM
        -5: s.USER_LIMIT,                # USER_REQUESTED_STOP
        -6: s.SOLVER_ERROR,              # INTERNAL_ERROR
        -7: s.SOLVER_ERROR,              # INVALID_NUMBER_OBJECTIVE
        -8: s.SOLVER_ERROR,              # INVALID_NUMBER_GRADIENT
        -9: s.SOLVER_ERROR,              # INVALID_NUMBER_CONSTRAINTS
        -10: s.SOLVER_ERROR,             # INVALID_NUMBER_JACOBIAN
        -11: s.SOLVER_ERROR,             # INVALID_NUMBER_HESSIAN_LAGRANGIAN
    }

    def __init__(self): self.libMad = None
    def name(self): return 'MADNLP'
    def cite(self): return CITATION_DICT["MADNLP"]

    def import_solver(self):
        if self.libMad is None:
            import libmad
            self.libMad = libmad.libMad()

    def invert(self, solution, inverse_data):
        attr = {}
        status = self.STATUS_MAP.get(solution['status'], s.SOLVER_ERROR)

        if 'iterations' in solution: attr[s.NUM_ITERS] = solution['iterations']
        if 'solve_time' in solution: attr[s.SOLVE_TIME] = solution['solve_time']

        if status in s.SOLUTION_PRESENT:
            primal_val = solution['obj_val']
            opt_val = primal_val + inverse_data.offset
            primal_vars = {}
            x_opt = solution['x']
            for id, offset in inverse_data.var_offsets.items():
                shape = inverse_data.var_shapes[id]
                size = np.prod(shape, dtype=int)
                primal_vars[id] = np.reshape(x_opt[offset:offset+size], shape, order='F')
            return Solution(status, opt_val, primal_vars, {}, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        self.import_solver()

        from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import Oracles

        bounds = data["_bounds"]
        oracles = Oracles(bounds.new_problem, bounds.x0, len(bounds.cl), verbose=verbose)

        problem = MadNLPProblem(data, self.libMad, oracles)
        opts_ptr = self.libMad.create_and_set_options(solver_opts)
        solver_ptr = self.libMad.create_solver("madnlp", problem.nlp_ptr, opts_ptr)
        stats_ptr = self.libMad.solve("madnlp", solver_ptr, opts_ptr)

        solution = {}
        solution['status'] = self.libMad.get_status("madnlp", stats_ptr)
        solution['iterations'] = self.libMad.get_iters("madnlp", stats_ptr)

        if self.libMad.get_success("madnlp", stats_ptr):
            solution['obj_val'] = self.libMad.get_obj("madnlp", stats_ptr)
            solution['x'] = np.array(self.libMad.get_solution("madnlp", stats_ptr, problem.n))

        self.libMad.delete("madnlp", stats_ptr, solver_ptr, opts_ptr)

        return solution
