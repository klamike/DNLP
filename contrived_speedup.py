import statistics, time
import numpy as np, cvxpy as cp
from cvxpy.reductions.fenchel_dual import fenchel_dual

# for those who say solving the dual will never be faster than the primal...
# here is a counter-example

# magic numbers
M = 40
N = 400
SEED = 18
SOLVER_NAME = "SCS"

# set it to false for a less contrived example
VAR_BOUNDS = False


def build(m, n, seed, var_bounds):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x0 = rng.uniform(0.0, 1.0, n)
    b = A @ x0 + np.abs(rng.standard_normal(m)) + 0.2
    c = rng.standard_normal(n)
    x = cp.Variable(n, bounds=(0,1) if var_bounds else None)
    return cp.Problem(cp.Minimize(c @ x), [A @ x <= b] + ([] if var_bounds else [x >= 0, x <= 1]))

def main(m=M, n=N, seed=SEED, solver_name=SOLVER_NAME, repeats=3, var_bounds=VAR_BOUNDS):
    solver = getattr(cp, solver_name)

    def bench(make_prob):
        wall, solve, comp, iters = [], [], [], []
        status = value = None
        for _ in range(repeats):
            prob = make_prob()
            t0 = time.perf_counter()
            prob.solve(solver=solver)
            wall.append(time.perf_counter() - t0)
            st = prob.solver_stats
            if st.solve_time is not None:
                solve.append(st.solve_time)
            comp.append(prob.compilation_time)
            iters.append(st.num_iters)
            status, value = prob.status, prob.value
        return dict(status=status, value=value, wall=statistics.median(wall),
                    solve=(statistics.median(solve) if solve else None),
                    comp=statistics.median(comp), iters=int(statistics.median(iters)))

    print(f"Configuration: solver={solver_name}, m={m}, n={n}, seed={seed}, var_bounds={var_bounds}")

    ps = bench(lambda: build(m, n, seed, var_bounds))
    print(f"\nPrimal solve (median): status={ps['status']}, value={ps['value']:.8f}, wall={ps['wall']:.6f}s, compilation={ps['comp']:.6f}s, solver={ps['solve']:.6f}, iters={ps['iters']}")
    
    ds = bench(lambda: fenchel_dual(build(m, n, seed, var_bounds)))
    print(f"  Dual solve (median): status={ds['status']}, value={ds['value']:.8f}, wall={ds['wall']:.6f}s, compilation={ds['comp']:.6f}s, solver={ds['solve']:.6f}, iters={ds['iters']}")
    
    gap = abs(ps["value"] - ds["value"])
    speed, kind = ((ps["solve"] / ds["solve"], "solver") if ps["solve"] and ds["solve"]
                   else (ps["wall"] / ds["wall"], "wall"))
    print(f"\nAbsolute duality gap: {gap:.6e}")
    print(f"{kind.capitalize()}-time speedup (primal / dual): {speed:.3f}x")

if __name__ == "__main__":
    main()
