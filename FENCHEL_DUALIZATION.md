# Atom-Level Fenchel Dualization in CVXPY

This document describes the mathematical model and implementation details of the atom-level Fenchel dualization procedure currently implemented in CVXPY.

Implementation files:

- `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/reductions/fenchel_dual.py`
- `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/reductions/fenchel_dual_terms.py`
- `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/reductions/fenchel_dual_constraints.py`
- atom-specific conjugates in `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/*`

## 1. Mathematical Background

### 1.1 Fenchel Conjugate

For a proper closed convex function $f:\mathbb{R}^n \to \mathbb{R}\cup\{+\infty\}$,
$$
f^*(y) = \sup_x \left\{\langle y, x\rangle - f(x)\right\}.
$$

Fenchel-Young inequality:
$$
f(x) + f^*(y) \ge \langle x,y\rangle.
$$

### 1.2 Concave Terms

For concave $g$, the pipeline uses
$$
(-g)^*(y) = \sup_x \left\{\langle y,x\rangle + g(x)\right\},
$$
exposed through `negative_conjugate(...)` / `negative_conjugate_term(...)`.

### 1.3 Perspective-Conjugate Scaling

For $s \ge 0$, the implementation supports $(s f)^*(y)$ directly.

When $s>0$,
$$
(s f)^*(y) = s\, f^*(y/s).
$$

The implementation does not rely on explicit division by $s$ in general; each atom provides a closed or conic form with closure behavior at $s=0$.

### 1.4 Composite Form

The implemented pipeline targets problems that can be rewritten (after normalization and decomposition) as
$$
\min_x \;\;
\sum_{i=1}^m s_i\,\varphi_i\!\left(u_i(x)\right)
\;+\; a(x)
\;+\; \sum_{r=1}^R I_{K_r}\!\left(w_r(x)\right)
\;+\; \sum_{q=1}^Q I_{\{g_q\le 0\}}(x),
$$
where:

- $u_i(x)$ and $w_r(x)$ are affine maps of $x$,
- $a(x)$ is affine,
- each $s_i$ is a scalar multiplier (numeric or DPP-safe symbolic),
- each $\varphi_i$ is an atom-level function with a conjugate rule.

#### 1.4.1 Term-wise Fenchel expansion

For each convex atom term:
$$
s_i\,\varphi_i(u_i(x))
=
\sup_{y_i}
\left\{
\langle y_i,u_i(x)\rangle
-
\left(s_i\varphi_i\right)^*(y_i)
\right\}.
$$

For concave atom terms represented as $-\psi_i(u_i(x))$, the code uses:
$$
s_i(-\psi_i)(u_i(x))
=
\sup_{y_i}
\left\{
\langle y_i,u_i(x)\rangle
-
\left(s_i(-\psi_i)\right)^*(y_i)
\right\}.
$$

This is exactly what `conjugate_term(...)` and `negative_conjugate_term(...)` encode.

#### 1.4.2 Constraint indicators as support/cone terms

For cone constraints, the implementation uses the dual-cone form
$$
I_{K_r}(w)
=
\sup_{\eta_r}
\left\{
\langle \eta_r,w\rangle - I_{K_r^*}(-\eta_r)
\right\},
$$
which is why the code couples affine expressions with dual variables and then enforces `_dual_cone(-dual_var)`.

For non-affine convex inequalities:
$$
I_{\{g_q\le 0\}}(x)
=
\sup_{\lambda_q\ge 0}
\langle \lambda_q, g_q(x)\rangle.
$$
The perspective multiplier $\lambda_q$ is then pushed into each decomposed atom of $g_q(x)$ through perspective-conjugate rules.

#### 1.4.3 Saddle form and stationarity

After introducing all dual variables, the primal becomes a min-sup saddle expression of the form
$$
\inf_x \sup_{\text{dual vars}}
\left[
\sum_k \langle d_k, A_k x + b_k\rangle
-\sum_i \Gamma_i(d_i)
+ a(x)
\right],
$$
where each $\Gamma_i$ is a conjugate or indicator term produced by atom/constraint dualization.

Swapping $\inf$ and $\sup$ (under standard closed/proper convex assumptions) yields the dual maximization with:

1. dual objective contributions $-\Gamma_i(\cdot)$ plus affine constants from $b_k$,
2. feasibility constraints from atom domains and dual cones,
3. stationarity in $x$:
$$
\sum_k A_k^* d_k + c = 0,
$$
where $a(x)=\langle c,x\rangle + c_0$ and $A_k^*$ is the adjoint map (Hermitian transpose in complex pairings, transpose in real pairings).

This is the core mathematical reason the implementation tracks:

- coupling pairs $(d_k,\text{affine\_expr}_k)$,
- explicit conjugate-domain constraints,
- a final aggregated affine-adjoint equation.

## 2. Problem Normalization

In `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/reductions/fenchel_dual.py`:

1. Validate objective is `Minimize` or `Maximize`.
2. Require DCP.
3. If parameters are present, require DPP.
4. Reject boolean/integer variables.
5. Convert `Maximize(f)` to `Minimize(-f)` for uniform construction.

Variable attributes are lowered to explicit constraints using CVXPY’s standard reduction:
$$
\texttt{CvxAttr2Constr(reduce\_bounds=True)}.
$$
This is done in `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/reductions/fenchel_dual_constraints.py`.

## 3. Objective Decomposition

Implemented in `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/reductions/fenchel_dual_terms.py`.

The decomposition enforces the invariant that each nonlinear term is represented as an atom applied to affine arguments (possibly after lifting):
$$
F(x)
=
\sum_{t=1}^T c_t\,\sigma_t\,\phi_t(v_t(x))
\;+\; a(x),
$$
where:

- $c_t\in\mathbb{R}$ is a numeric coefficient,
- $\sigma_t$ is an optional scalar symbolic multiplier,
- $v_t(x)$ is affine in $x$,
- $a(x)$ is affine.

The tree walk returns three objects:

- `atom_terms`: each term is `AtomTerm(atom, affine_args, nonconstant_arg_indices, coefficient, scalar_multiplier, sum_outputs)`,
- `affine_term`: aggregate affine remainder,
- `lifted_constraints`: auxiliary constraints introduced to make nested atoms affine-argument compatible.

### 3.1 Walk rules and algebraic normalization

The recursive walker applies:

1. additive splitting:
$$
f_1(x)+f_2(x)\;\mapsto\;\text{walk}(f_1)+\text{walk}(f_2),
$$
2. sign push-through:
$$
-f(x)\;\mapsto\;(-1)\cdot f(x),
$$
3. scalar peeling:
$$
\alpha f(x)\;\mapsto\;(\alpha)\cdot f(x),\qquad
\frac{f(x)}{\beta}\;\mapsto\;\left(\frac{1}{\beta}\right)f(x),
$$
including symbolic scalar factors when DPP-safe.

For `cp.sum(...)`, the decomposition marks `sum_outputs=True`, so conjugate objective contributions are scalarized consistently later.

### 3.2 Affine vs atom split

- If a subtree is affine, it is accumulated into `affine_term`.
- If a subtree is an atom with affine arguments, it becomes one `AtomTerm`.
- If a subtree is constant, it is treated as affine.

For multi-argument atoms, only non-constant argument slots are dualized. If
$$
\phi(z_1,\dots,z_m),\quad z_j\in\{\text{affine},\text{constant}\},
$$
then `nonconstant_arg_indices` stores the dualized subset
$$
J = \{j:\;z_j\ \text{is non-constant}\}.
$$
This enables partial conjugates over active arguments while keeping constant arguments as parameters to the atom rule.

### 3.3 Scalar multiplier bookkeeping

Each term carries two multiplier channels:

- `coefficient` ($c_t$): numeric scalar accumulated by algebraic peeling,
- `scalar_multiplier` ($\sigma_t$): symbolic scalar expression (typically parameterized).

The effective term scale used during conjugation is
$$
\text{scale}_t = c_t\cdot \sigma_t
$$
in objective dualization, and
$$
\text{scale}_t = \lambda\cdot c_t\cdot \sigma_t
$$
inside non-affine inequality dualization (perspective multiplier $\lambda\ge 0$).

### 3.4 Nested non-affine lifting (important)

If an atom argument is non-affine, the code attempts monotonicity-based lifting.
Suppose a term has sign-adjusted target convexity and contains $\phi(\ldots,g(x),\ldots)$ with non-affine $g$.

An auxiliary variable $t$ is introduced with one of:

- increasing slot: $g(x)\le t$,
- decreasing slot: $g(x)\ge t$.

The sign logic is:

- if overall term sign is positive and atom is convex: use atom monotonicity directly,
- if overall term sign is negative and atom is concave: use swapped monotonicity (because $-\phi$ is convex).

This keeps the relaxed representation equivalent at optimum while producing an affine argument atom for conjugation. If a non-affine argument is in a non-monotone slot, lifting is rejected.

### 3.5 Decomposition output form

After recursion:
$$
F(x)
=
\sum_{t=1}^T \text{Term}_t(x)\;+\;a(x),
$$
where each `Term_t` is now conjugate-ready with explicit metadata:

- atom object,
- affine argument list,
- dualized argument index set,
- scale metadata,
- whether output was implicitly summed (`sum_outputs`).

## 4. Atom-Term Dualization

Given one decomposed term
$$
\tau(x)=\text{scale}\cdot \phi\!\left(v_1(x),\dots,v_p(x)\right),
$$
with active argument index set $J\subseteq\{1,\dots,p\}$ (non-constant slots), the implementation computes a partial conjugate:
$$
\tau(x)
=
\sup_{\{y_j\}_{j\in J}}
\left\{
\sum_{j\in J}\langle y_j,v_j(x)\rangle
- \phi_{J,\text{scale}}^*(\{y_j\}_{j\in J})
\right\}.
$$

### 4.1 Dual variable creation

One dual variable is created per active affine argument $v_j(x)$:
$$
y_j \in \mathbb{R}^{\text{shape}(v_j)}
\quad\text{or}\quad
\mathbb{C}^{\text{shape}(v_j)}
$$
matching argument shape and complex/real type.

### 4.2 Effective scale and broadcasting

For each term:
$$
\text{scale}
=
\underbrace{\lambda}_{\text{perspective, optional}}
\cdot
\underbrace{\sigma}_{\text{symbolic scalar multiplier}}
\cdot
\underbrace{c}_{\text{numeric coefficient}}.
$$

If scale is scalar and atom output is tensor-valued, scale is broadcast to atom output shape before passing to the atom rule.

### 4.3 Sign dispatch to conjugate APIs

The dispatch is:

- if $\text{scale}\ge 0$: call
  `atom.conjugate_term(nonconstant_arg_indices, dual_vars, perspective_scale=scale)`,
- if $\text{scale}\le 0$: call
  `atom.negative_conjugate_term(..., perspective_scale=-scale)`,
- if sign is unknown: reject.

Mathematically, the second branch uses
$$
(\text{scale}\,\phi)^*(y)
=
((-\text{scale})\,(-\phi))^*(y),
\quad \text{when }\text{scale}\le 0.
$$

### 4.4 Returned objects and assembly

Each atom rule returns:

- `conj_expr`: expression for the conjugate value (possibly vector/matrix),
- `conj_constraints`: domain/epigraph constraints for that conjugate representation.

The global dual builder then adds:
$$
\text{dual obj} \;\mathrel{+}= -\operatorname{scalarize}(\text{conj\_expr}),
$$
and appends all `conj_constraints`.

Couplings are stored as pairs $(y_j,v_j(x))$ and later converted to affine-adjoint contributions in stationarity:
$$
\sum_{(y,v)} \langle y, v(x)\rangle.
$$

### 4.5 Scalarization rule

Conjugates can return non-scalar expressions (e.g., elementwise conjugates). The implementation scalarizes by summation (respecting `sum_outputs` markers from decomposition), ensuring final dual objective is scalar.

### 4.6 Multi-argument atom support

`conjugate_term` / `negative_conjugate_term` are the abstraction that makes non-unary atoms feasible. For example:

- full argument dualization (e.g., `quad_over_lin` over numerator+denominator),
- partial dualization with constant arguments (e.g., unary `rel_entr(x,a)` and `kl_div(x,a)`),
- specialized multi-argument piecewise-linear atoms (`maximum`, `minimum`).

This is a key design point: decomposition records exactly which argument slots are active, and atom-level term-conjugate methods implement those cases directly.

## 5. Constraint Dualization

Implemented in `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/reductions/fenchel_dual_constraints.py` via `singledispatch`.

### 5.1 Affine inequalities/equalities

For affine inequality $e(x)\le 0$, create $\lambda\ge 0$ and add coupling $\langle \lambda,e(x)\rangle$.

For equality/zero constraints, create free dual $\nu$ and add $\langle \nu,e(x)\rangle$.

### 5.2 Cone constraints

For SOC/PSD/ExpCone and other constraints exposing `_dual_cone`, dual-cone membership is enforced by calling:
$$
\texttt{con.\_dual\_cone(-dual\_var)}.
$$
The minus sign matches CVXPY’s internal sign convention in the coupling assembly.

### 5.3 Non-affine inequalities $g(x)\le 0$

For non-affine convex inequality:
$$
I_{\{g\le 0\}}(x) = \sup_{\lambda\ge 0} \langle \lambda, g(x)\rangle.
$$

The algorithm:

1. Introduce $\lambda\ge 0$,
2. decompose $g(x)$ into atom terms + affine part,
3. dualize each term with perspective scale $\lambda$,
4. add $-(\lambda\text{-scaled conjugate})$ to the dual objective,
5. add affine coupling $\langle \lambda, g_{\text{aff}}(x)\rangle$.

If decomposition introduced lifted constraints, those constraints are dualized recursively in the same framework.

## 6. Stationarity Assembly

For each affine coupling expression
$$
e(x)=A x + b,
$$
the code extracts:

- adjoint piece $A^* y$,
- constant piece $\langle y,b\rangle$,

and sums across all couplings plus affine objective remainder.

Resulting stationarity condition:
$$
\sum_{\text{all couplings}} A_k^* y_k \;+\; a = 0.
$$

### 6.1 Complex-native handling

For complex affine pairings, the implementation uses
$$
\operatorname{Re}\!\left(y^H(Ax+b)\right)
=
\operatorname{Re}(y^H A x) + \operatorname{Re}(y^H b),
$$
so:

- adjoint uses $A^H$,
- constants use $\operatorname{Re}\!\left(\sum \overline{y}\odot b\right)$.

For real pairings, it uses $A^\top y$ and $\sum y\odot b$.

Special affine wrappers `real(...)` and `imag(...)` are handled structurally:

- $\operatorname{Re}(u)$: reuse adjoint of $u$,
- $\operatorname{Im}(u)$: equivalent pairing with $i\,y$.

### 6.2 Coefficient extraction backend

When complex leaves are present, coefficient extraction uses SCIPY canonicalization backend (CPP path is not currently complex-safe for this extraction use).

## 7. Final Dual Objective and Orientation

Let $c_{\text{aff}}$ be the sum of constant terms from couplings and affine objective.
Let $q_i$ be the conjugate expressions for atom terms.

The assembled dual objective is:
$$
\max \; c_{\text{aff}} - \sum_i \operatorname{scalarize}(q_i),
$$
subject to:

- stationarity,
- cone/indicator constraints from conjugates,
- dual-cone constraints from primal constraints.

If the original primal was `Maximize`, the code first negates it to minimization and then returns the sign-corrected final orientation.

## 8. Reduction API and Inversion

`FenchelDual` is implemented as a `Reduction` in `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/reductions/fenchel_dual.py`:

- `apply(problem)`: returns dual problem + inverse data.
- `invert(solution, inverse_data)`:
  - maps statuses (infeasible/unbounded swap),
  - reconstructs primal variable values from the dual value of stationarity.

If $\mu$ is the dual multiplier on stationarity, recovered primal vector is
$$
x = -\mu,
$$
then reshaped by original variable offsets/shapes.

## 9. Representative Atom Rules Implemented

The system is atom-level: each atom provides its own conjugate or term-conjugate formula/constraints.

Representative examples:

### 9.1 Norms and max-type atoms

$$
\|x\|_1^*(y)=I_{\{\|y\|_\infty\le 1\}},\quad
\|x\|_\infty^*(y)=I_{\{\|y\|_1\le 1\}},
$$
$$
\|x\|_p^*(y)=I_{\{\|y\|_q\le 1\}},\;\; q=\frac{p}{p-1},\; p>1.
$$
$$
\max(x)^*(y)=I_{\{y\ge 0,\;\mathbf{1}^\top y=1\}},
$$
$$
\mathrm{sum\_largest}(x,k)^*(y)=I_{\{0\le y\le 1,\;\mathbf{1}^\top y=k\}}.
$$

### 9.2 Exponential-family atoms

$$
\exp^*(y)= y\log y - y,\;\; y\ge 0.
$$
$$
(\log\sum_i e^{x_i})^*(y)=\sum_i y_i\log y_i,\;\; y\ge 0,\;\sum_i y_i=1.
$$
Scaled form uses $s$ and $\sum y_i=s$, implemented via `rel_entr`.

$$
\mathrm{logistic}(x)=\log(1+e^x),\quad
(s\,\mathrm{logistic})^*(y)=\mathrm{rel\_entr}(y,s)+\mathrm{rel\_entr}(s-y,s),
$$
with $0\le y\le s$.

### 9.3 Power / Huber

For $p>1$, $q=\frac{p}{p-1}$:
$$
(x^p)^*(y)=\frac{p-1}{p^q}|y|^q
$$
with one-sided domain handling when appropriate; implemented conically via `PowCone3D`.

$$
\mathrm{Huber}_M^*(y)=\frac{y^2}{4}\;\text{on}\; |y|\le 2M,
$$
and scaled version with $s$: $\frac{y^2}{4s}$, $|y|\le 2Ms$.

### 9.4 Quadratic-family atoms

$$
(x^\top P x)^*(y)=\frac14 y^\top P^\dagger y
$$
for PSD $P$, with range constraints for singular $P$.

$$
\left(\frac{\|x\|_2^2}{t}\right)^*(y)=\frac{t}{4}\|y\|_2^2,\;\; t>0,
$$
with axis-aware and multi-argument variants.

`matrix_frac(X,P)` conjugate is implemented (constant PSD $P$, real duals) via factorization $P=BB^\top$:
$$
\sum_j \frac{\|B^\top y_j\|_2^2}{4s}.
$$

### 9.5 Spectral atoms

$$
\lambda_{\max}(X)^*(Y)=I_{\{Y=Y^H,\;Y\succeq 0,\;\mathrm{tr}(Y)=1\}},
$$
$$
\|X\|_\sigma^*(Y)=I_{\{\|Y\|_*\le 1\}},
$$
$$
\|X\|_*^*(Y)=I_{\{\|Y\|_\sigma\le 1\}},
$$
$$
\left(\sum_{i=1}^k \lambda_i^\downarrow(X)\right)^*(Y)
=
I_{\{Y=Y^H,\;0\preceq Y\preceq I,\;\mathrm{tr}(Y)=k\}}.
$$
All have perspective-scaled forms.

### 9.6 Concave atoms via negative conjugates

$$
(s(-\log))^*(y)=\mathrm{rel\_entr}(s,-y)-s,\;\; y\le 0,\; s\ge 0,
$$
$$
(s(-\mathrm{entr}))^*(y)=s\,e^{y/s-1}
$$
implemented via `ExpCone`.

$$
(s(-\log\det))^*(Y)= -sn - s\log\det(-Y/s),
$$
with $Y=Y^H,\;Y\preceq 0$.

## 10. Current Scope and Practical Limits

1. The method is exact only where atom conjugate rules are implemented.
2. Unknown-sign parametric multipliers on nonlinear atom terms are rejected.
3. Complex nonlinear multipliers are rejected; complex primal variables are supported natively.
4. Non-affine equalities are not dualized through perspective machinery.
5. If an atom/argument pattern lacks a conjugate rule, `NotImplementedError` is raised.

## 11. Conceptual Summary

The implemented dualization is a structured Fenchel-Rockafellar construction:

1. normalize problem,
2. decompose objective and non-affine inequalities into atom terms,
3. apply atom-level conjugate rules (including perspective scaling),
4. dualize constraint indicators via dual cones,
5. enforce stationarity using exact affine adjoints,
6. assemble final dual objective and constraints.

The key architectural choice is pushing duality logic to atom classes (conjugate rules) while keeping global assembly generic and affine-map based.

## 12. Convex Atom Conjugate Catalog

The table below covers Fenchel-enabled convex atoms in this implementation.

| Atom | API | Conjugate representation | Domain constraints (implemented form) | File |
|---|---|---|---|---|
| `norm1` | `conjugate` | Indicator of dual $\ell_\infty$ ball | $\|y\|_\infty \le s$ (axis-aware via helper) | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/norm1.py` |
| `norm_inf` | `conjugate` | Indicator of dual $\ell_1$ ball | $\|y\|_1 \le s$ (axis-aware via helper) | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/norm_inf.py` |
| `pnorm` ($p>1$) | `conjugate` | Indicator of dual $\ell_q$ ball | $\|y\|_q \le s$, $q=\frac{p}{p-1}$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/pnorm.py` |
| `max` | `conjugate` | Indicator of simplex-like set | $y\ge 0,\;\mathrm{axis\_sum}(y)=s$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/max.py` |
| `sum_largest(x,k)` | `conjugate` | Indicator of capped simplex | $0\le y\le s,\;\mathbf{1}^\top y = ks$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/sum_largest.py` |
| `abs` | `conjugate` | Indicator of unit ball in dual norm | real: $-s\le y\le s$; complex: $|y|\le s$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/elementwise/abs.py` |
| `exp` | `conjugate` | $ \mathrm{rel\_entr}(y,s)-y $ | $y\ge 0,\;s\ge 0$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/elementwise/exp.py` |
| `log_sum_exp` | `conjugate` | $\mathrm{axis\_sum}\!\left(\mathrm{rel\_entr}(y,s_{\text{broadcast}})\right)$ | $y\ge 0,\;\mathrm{axis\_sum}(y)=s$ (and $s\ge 0$ if needed) | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/log_sum_exp.py` |
| `logistic` | `conjugate` | $\mathrm{rel\_entr}(y,s)+\mathrm{rel\_entr}(s-y,s)$ | $0\le y\le s,\;s\ge 0$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/elementwise/logistic.py` |
| `power(x,p)` ($p>1$) | `conjugate` | Conic form equivalent to $\frac{p-1}{p^q}|y|^q$ | implemented via `PowCone3D`; plus scale/domain constraints | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/elementwise/power.py` |
| `huber(x,M)` | `conjugate` | Quadratic conjugate + box | quadratic term from `power(_,2).conjugate`; $|y|\le 2Ms$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/elementwise/huber.py` |
| `rel_entr(x,a)` (unary in $x$) | `conjugate` | Epigraph/cone form in $z$ | `ExpCone(y-s,s,z)`, $a\ge 0$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/elementwise/rel_entr.py` |
| `kl_div(x,a)` (unary in $x$) | `conjugate` | Epigraph/cone form in $z$ | `ExpCone(y,s,z)`, $a\ge 0$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/elementwise/kl_div.py` |
| `quad_form(x,P)` ($P\succeq 0$) | `conjugate` | Closed form or conic/slack form | PSD/constant checks; optional range constraint for singular $P$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/quad_form.py` |
| `quad_over_lin(x,t)` ($t>0$ const) | `conjugate` | closed or epigraph form, axis-aware | denominator positive constant; scale/domain checks | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/quad_over_lin.py` |
| `matrix_frac(X,P)` ($P\succeq 0$ const) | `conjugate` | factorization-based sum of `quad_over_lin` terms | real duals, scalar scale, PSD/constant $P$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/matrix_frac.py` |
| `lambda_max` | `conjugate` | Indicator of trace-constrained PSD cone | $Y=Y^H,\;Y\succeq 0,\;\mathrm{tr}(Y)=s$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/lambda_max.py` |
| `sigma_max` | `conjugate` | Indicator of nuclear-norm ball | $\|Y\|_*\le s$ (and $s\ge 0$ if needed) | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/sigma_max.py` |
| `norm_nuc` | `conjugate` | Indicator of spectral-norm ball | $\sigma_{\max}(Y)\le s$ (and $s\ge 0$ if needed) | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/norm_nuc.py` |
| `lambda_sum_largest(X,k)` | `conjugate` | Indicator of matrix capped-simplex set | $Y=Y^H,\;0\preceq Y\preceq sI,\;\mathrm{tr}(Y)=ks$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/lambda_sum_largest.py` |
| `maximum(x_1,\dots,x_m)` | `conjugate_term` | term-level piecewise indicator/affine form | per-arg nonnegativity dual constraints + sum/constant-arg slack rules | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/elementwise/maximum.py` |

## 13. Concave Atom Negative-Conjugate Catalog

The table below lists concave atoms with implemented negative-conjugate behavior.

| Atom | API | Negative conjugate representation | Domain constraints (implemented form) | File |
|---|---|---|---|---|
| `log` | `negative_conjugate` | $\mathrm{rel\_entr}(s,-y)-s$ | $y\le 0,\;s\ge 0$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/elementwise/log.py` |
| `entr` | `negative_conjugate` | epigraph variable $z$ for $s e^{y/s-1}$ | `ExpCone(y-s,s,z)`, $s\ge 0$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/elementwise/entr.py` |
| `log_det` | `negative_conjugate` | $-sn - s\log\det(-Y/s)$ with closure at $s=0$ | $Y=Y^H,\;-Y\succeq 0$ (and $s\ge 0$ if needed) | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/log_det.py` |
| `quad_form(x,P)` ($P\preceq 0$) | `negative_conjugate` | delegated to conjugate of `quad_form(x,-P)` | NSD/constant checks on $P$ | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/quad_form.py` |
| `minimum(x_1,\dots,x_m)` | `negative_conjugate_term` | term-level indicator/affine form | per-arg nonpositivity dual constraints + sum/constant-arg slack rules | `/Users/mike/Git/cvxhack/sources/cvxpy/cvxpy/atoms/elementwise/minimum.py` |

## 14. Multi-Case Conjugate Tables

This section gives case splits for atoms whose conjugate implementation has multiple branches.

### 14.1 `quad_over_lin` Cases

| API case | Condition | Returned conjugate form | Extra constraints |
|---|---|---|---|
| `conjugate` (scalar axis) | axis `None` or vectorized, scalar scale known constant | $\mathrm{quad\_over\_lin}(y,\;4s/t)$ | none |
| `conjugate` (scalar axis) | axis `None` or vectorized, scalar scale symbolic | epigraph $z$ with objective $\frac{t}{4}z$ | $\mathrm{quad\_over\_lin}(y,s)\le z$ |
| `conjugate` (axis 0/1, scalar scale constant) | matrix $y$, axis-aware | slice-summed closed form | none |
| `conjugate` (axis 0/1, scalar scale symbolic) | matrix $y$, axis-aware | objective $\frac{t}{4}\sum z_i$ | per-slice $\mathrm{quad\_over\_lin}(y_i,s)\le z_i$ |
| `conjugate` (axis 0/1, vector scale constant) | one scale per slice | slice-wise sum of $\mathrm{quad\_over\_lin}(y_i,4s_i/t)$ | none |
| `conjugate` (axis 0/1, vector scale symbolic) | one symbolic scale per slice | objective $\frac{t}{4}\sum z_i$ | per-slice $\mathrm{quad\_over\_lin}(y_i,s_i)\le z_i$ |
| `conjugate_term` unary | nonconstant args `(0,)` | delegates to `conjugate` | as above |
| `conjugate_term` full | nonconstant args `(0,1)` | indicator form | $y_t + \mathrm{rhs}(y_x,s)\le 0$ |
| `conjugate_term` denom-only | nonconstant args `(1,)` | $-2\,\mathrm{geo\_mean}([w,-y_t])$ | $y_t\le 0$ (+ $w\ge 0$ if needed) |

### 14.2 `quad_form` Cases

| Condition | Returned conjugate form | Extra constraints |
|---|---|---|
| $P$ constant PSD, $s=1$ constant | $\frac14\,y^\top P^\dagger y$ | range constraint $(I-PP^\dagger)y=0$ when rank-deficient |
| $P$ constant PSD, general scalar $s$ | slack/factor form: $P=BB^H$, $y=Bw$, objective $\mathrm{quad\_over\_lin}(w,4s)$ | $y=Bw$ |
| $P$ PSD parameterized | Schur-style epigraph with scalar $t$ | block PSD LMI and $s\ge 0$ when needed |
| `negative_conjugate` for NSD $P$ | delegates to conjugate with $-P$ | NSD/constant checks |

### 14.3 `matrix_frac` Cases

| Condition | Returned conjugate form | Extra constraints |
|---|---|---|
| $y$ vector | single term $\mathrm{quad\_over\_lin}(B^\top y,\;4s)$ | $s\ge 0$ if needed |
| $y$ matrix | sum over columns $\sum_j \mathrm{quad\_over\_lin}(B^\top y_j,\;4s)$ | $s\ge 0$ if needed |
| $P$ rank-deficient with no positive eigenvalues | $0$ | only scale constraints |

### 14.4 `rel_entr` Cases

| API case | Condition | Returned conjugate form | Extra constraints |
|---|---|---|---|
| `conjugate` unary | second arg constant | objective $\langle a,z\rangle$ | `ExpCone(y-s,s,z)`, $a\ge 0$ |
| `conjugate_term` full | nonconstant args `(0,1)` | pure indicator | `ExpCone(u-s,s,-v)` |

### 14.5 `kl_div` Cases

| API case | Condition | Returned conjugate form | Extra constraints |
|---|---|---|---|
| `conjugate` unary | second arg constant | objective $\langle a,z-s\rangle$ | `ExpCone(y,s,z)`, $a\ge 0$ |
| `conjugate_term` full | nonconstant args `(0,1)` | pure indicator | `ExpCone(u,s,s-v)` |

### 14.6 `maximum` Cases (term-level)

| Condition | Returned conjugate form | Constraints |
|---|---|---|
| all args nonconstant | indicator | $y_i\ge 0$ for each $i$, $\sum_i y_i = s$ |
| some args constant $c_j$ | affine term $-\max_j(c_j)\cdot(s-\sum_i y_i)$ | $y_i\ge 0$, $s-\sum_i y_i\ge 0$ |

### 14.7 `minimum` Cases (negative term-conjugate)

| Condition | Returned negative-conjugate form | Constraints |
|---|---|---|
| all args nonconstant | indicator | $y_i\le 0$ for each $i$, $s+\sum_i y_i = 0$ |
| some args constant $c_j$ | affine term $\min_j(c_j)\cdot(s+\sum_i y_i)$ | $y_i\le 0$, $s+\sum_i y_i\ge 0$ |

### 14.8 `power` Cases

| Condition | Returned conjugate form | Notes |
|---|---|---|
| $p>1$ even/power-of-2 style domain | conic form equivalent to $\frac{p-1}{p^q}|y|^q$ | `PowCone3D` with scale broadcasting |
| $p>1$ one-sided domain (non-power-of-2) | conic form with auxiliary $u\ge y$ | enforces one-sided domain behavior |
| $p\le 1$ or nonconstant $p$ | not implemented in this conjugate path | raises `NotImplementedError` |

### 14.9 `log_det` Negative-Conjugate Cases

| Condition | Returned form | Constraints |
|---|---|---|
| $s=0$ (closure case) | indicator $I_{\{Y=0\}}$ | $Y=0$ |
| $s>0$ general | $-sn - s\log\det(-Y/s)$ | $Y=Y^H,\;-Y\succeq 0$ (plus $s\ge 0$ if needed) |
