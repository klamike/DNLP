import ctypes, os, platform

from typing import Union
from ctypes import (
    c_int, c_double, c_longlong, c_bool,
    c_void_p, c_char_p,
    POINTER, CFUNCTYPE
)


class libMad:
    NlpConstrJacStructure = CFUNCTYPE(c_int, POINTER(c_longlong), POINTER(c_longlong), c_void_p)
    NlpLagHessStructure   = CFUNCTYPE(c_int, POINTER(c_longlong), POINTER(c_longlong), c_void_p)
    NlpEvalObj            = CFUNCTYPE(c_int, POINTER(c_double), POINTER(c_double), c_void_p)
    NlpEvalConstr         = CFUNCTYPE(c_int, POINTER(c_double), POINTER(c_double), c_void_p)
    NlpEvalObjGrad        = CFUNCTYPE(c_int, POINTER(c_double), POINTER(c_double), c_void_p)
    NlpEvalConstrJac      = CFUNCTYPE(c_int, POINTER(c_double), POINTER(c_double), c_void_p)
    NlpEvalLagHess        = CFUNCTYPE(c_int, c_double, POINTER(c_double), POINTER(c_double), POINTER(c_double), c_void_p)

    TYPES = [
        ("libmad_create_options_dict",    [POINTER(c_void_p)]),
        ("libmad_set_int64_option",       [c_void_p, c_char_p, c_longlong]),
        ("libmad_set_double_option",      [c_void_p, c_char_p, c_double]),
        ("libmad_set_bool_option",        [c_void_p, c_char_p, c_bool]),
        ("libmad_set_string_option",      [c_void_p, c_char_p, c_char_p]),
        ("libmad_delete_options_dict",    [c_void_p]),
        ("madnlp_create_solver",          [POINTER(c_void_p), c_void_p, c_void_p]),
        ("madnlp_delete_solver",          [c_void_p]),
        ("madnlp_solve",                  [c_void_p, c_void_p, POINTER(c_void_p)]),
        ("madnlp_get_obj",                [c_void_p, POINTER(c_double)]),
        ("madnlp_get_solution",           [c_void_p, POINTER(c_double)]),
        ("madnlp_get_constraints",        [c_void_p, POINTER(c_double)]),
        ("madnlp_get_multipliers",        [c_void_p, POINTER(c_double)]),
        ("madnlp_get_multipliers_L",      [c_void_p, POINTER(c_double)]),
        ("madnlp_get_multipliers_U",      [c_void_p, POINTER(c_double)]),
        ("madnlp_get_bound_multipliers",  [c_void_p, POINTER(c_double)]),
        ("madnlp_get_success",            [c_void_p, POINTER(c_bool)]),
        ("madnlp_get_iters",              [c_void_p, POINTER(c_longlong)]),
        ("madnlp_get_primal_feas",        [c_void_p, POINTER(c_double)]),
        ("madnlp_get_dual_feas",          [c_void_p, POINTER(c_double)]),
        ("madnlp_get_status",             [c_void_p, POINTER(c_longlong)]),
        ("madnlp_delete_stats",           [c_void_p]),
        ("libmad_nlpmodel_create",        [POINTER(c_void_p), c_char_p, c_longlong, c_longlong, c_longlong, c_longlong,
                                            NlpConstrJacStructure, NlpLagHessStructure, NlpEvalObj, NlpEvalConstr,
                                            NlpEvalObjGrad, NlpEvalConstrJac, NlpEvalLagHess, c_void_p]),
        ("libmad_nlpmodel_set_numerics",  [c_void_p, POINTER(c_double), POINTER(c_double), POINTER(c_double),
                                            POINTER(c_double), POINTER(c_double), POINTER(c_double)]),
    ]

    def __init__(self):
        self.lib = None
        self._load_library()
        self._setup_functions()

    def _load_library(self):
        self.lib = ctypes.CDLL(os.path.join(os.environ.get('LIBMAD_PATH'), {  # TODO: proper discovery
            "Darwin": "libMad.dylib", "Linux":  "libMad.so", "Windows":"libMad.dll",
        }[platform.system()]))

    def _bind(self, name, argtypes, restype=c_int):
        fn = getattr(self.lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    def _setup_functions(self):
        for name, args in libMad.TYPES: self._bind(name, args)

    @staticmethod
    def _check(ret: int, msg: str, **kwargs):
        if ret != 0: raise RuntimeError(f"{msg.format(**kwargs)} (code {ret})")

    def create_options(self) -> c_void_p:
        opts = c_void_p()
        self._check(
            self.lib.libmad_create_options_dict(ctypes.byref(opts)
        ),  "create_options failed")
        return opts

    def set_option(self, opts_ptr: c_void_p, key: str, value):
        k = key.encode("utf-8")
        if isinstance(value, bool):
            ret = self.lib.libmad_set_bool_option(opts_ptr, k, c_bool(value))
        elif isinstance(value, int):
            ret = self.lib.libmad_set_int64_option(opts_ptr, k, c_longlong(value))
        elif isinstance(value, float):
            ret = self.lib.libmad_set_double_option(opts_ptr, k, c_double(value))
        elif isinstance(value, str):
            ret = self.lib.libmad_set_string_option(opts_ptr, k, value.encode("utf-8"))
        else:
            raise TypeError(f"Unsupported option type for {key}: {type(value)}")
        
        self._check(ret, "set {type} option {key} to {value} failed", key=key, value=value, type=type(value))

    def set_options(self, opts_ptr: c_void_p, options: dict):
        for k, v in options.items():
            self.set_option(opts_ptr, k, v)

    def create_and_set_options(self, options: dict) -> c_void_p:
        opts = self.create_options()
        self.set_options(opts, options)
        return opts

    def create_nlpmodel(self, callbacks, user_data=None) -> c_void_p:
        meta = callbacks["meta"]
        m, n = meta["m"], meta["n"]
        nnzj, nnzh = meta["nnzj"], meta["nnzh"]

        nlp = c_void_p()
        self._check(self.lib.libmad_nlpmodel_create(
            ctypes.byref(nlp), b"cvxpy_nlpmodel",
            n, m, nnzj, nnzh,
            callbacks["jac_struct"], callbacks["hess_struct"],
            callbacks["eval_f"], callbacks["eval_g"],
            callbacks["eval_grad_f"], callbacks["eval_jac_g"],
            callbacks["eval_h"],
            user_data
        ), "create_nlpmodel failed")
        return nlp

    def set_numerics(self, nlp_ptr, n, m, x0, y0, lvar, uvar, lcon, ucon):
        x0_arr   = (c_double * n)(*x0)
        y0_arr   = (c_double * m)(*y0)
        lvar_arr = (c_double * n)(*lvar)
        uvar_arr = (c_double * n)(*uvar)
        lcon_arr = (c_double * m)(*lcon)
        ucon_arr = (c_double * m)(*ucon)

        self._check(self.lib.libmad_nlpmodel_set_numerics(
            nlp_ptr, x0_arr, y0_arr, lvar_arr, uvar_arr, lcon_arr, ucon_arr
        ), "set_numerics failed")

    def create_solver(self, solver: str, nlp_ptr: c_void_p, opts_ptr: c_void_p) -> c_void_p:
        if solver != "madnlp":
            raise RuntimeError(f"Unsupported solver: {solver}")
        solver_ptr = c_void_p()
        self._check(self.lib.madnlp_create_solver(
            ctypes.byref(solver_ptr), nlp_ptr, opts_ptr
        ), "create_solver failed")
        return solver_ptr

    def solve(self, solver: str, solver_ptr: c_void_p, opts_ptr: c_void_p) -> c_void_p:
        if solver != "madnlp":
            raise RuntimeError(f"Unsupported solver: {solver}")
        stats = c_void_p()
        self._check(self.lib.madnlp_solve(solver_ptr, opts_ptr, ctypes.byref(stats)), "solve failed")
        return stats

    def get_success(self, stats_ptr: c_void_p) -> bool:
        v = c_bool()
        self._check(self.lib.madnlp_get_success(stats_ptr, ctypes.byref(v)), "get_success failed")
        return bool(v.value)

    def get_status(self, stats_ptr: c_void_p) -> int:
        v = c_longlong()
        self._check(self.lib.madnlp_get_status(stats_ptr, ctypes.byref(v)), "get_status failed")
        return int(v.value)

    def get_iters(self, stats_ptr: c_void_p) -> int:
        v = c_longlong()
        self._check(self.lib.madnlp_get_iters(stats_ptr, ctypes.byref(v)), "get_iters failed")
        return int(v.value)

    def get_obj(self, stats_ptr: c_void_p) -> float:
        v = c_double()
        self._check(self.lib.madnlp_get_obj(stats_ptr, ctypes.byref(v)), "get_obj failed")
        return float(v.value)

    def get_solution(self, stats_ptr: c_void_p, n: int, tolist=False) -> Union[list[float], ctypes.Array]:
        x = (c_double * n)()
        self._check(self.lib.madnlp_get_solution(stats_ptr, x), "get_solution failed")
        return list(x) if tolist else x

    def delete_stats(self, stats_ptr: c_void_p):
        self._check(self.lib.madnlp_delete_stats(stats_ptr), "delete_stats failed")
    
    def delete_solver(self, solver_ptr: c_void_p):
        self._check(self.lib.madnlp_delete_solver(solver_ptr), "delete_solver failed")
    
    def delete_options(self, opts_ptr: c_void_p):
        self._check(self.lib.libmad_delete_options_dict(opts_ptr), "delete_options failed")

    def delete(self, stats_ptr: c_void_p, solver_ptr: c_void_p, opts_ptr: c_void_p):
        self.delete_stats(stats_ptr)
        self.delete_solver(solver_ptr)
        # self.delete_options(opts_ptr). # FIXME
