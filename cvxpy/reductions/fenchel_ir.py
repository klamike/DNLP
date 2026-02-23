"""
Fenchel dualization intermediate representation.
"""

from dataclasses import dataclass, field
from typing import List

from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression


@dataclass
class CouplingTerm:
    """Represents <dual_var, affine_expr> coupling in the Lagrangian."""

    dual_var: Expression
    affine_expr: Expression
    source: str = ""


@dataclass
class FenchelIR:
    """Collects objective pieces, constraints, and affine couplings."""

    objective_terms: List[Expression] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    couplings: List[CouplingTerm] = field(default_factory=list)

    def extend(self, other: "FenchelIR") -> None:
        self.objective_terms.extend(other.objective_terms)
        self.constraints.extend(other.constraints)
        self.couplings.extend(other.couplings)

    def add_objective(self, expr: Expression) -> None:
        self.objective_terms.append(expr)

    def add_constraint(self, con: Constraint) -> None:
        self.constraints.append(con)

    def add_constraints(self, constraints: List[Constraint]) -> None:
        self.constraints.extend(constraints)

    def add_coupling(self, dual_var: Expression, affine_expr: Expression, source: str = "") -> None:
        self.couplings.append(CouplingTerm(dual_var=dual_var, affine_expr=affine_expr, source=source))
