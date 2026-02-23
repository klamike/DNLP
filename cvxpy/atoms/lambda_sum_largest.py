"""
Copyright 2013 Steven Diamond

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
from scipy import linalg as LA

from cvxpy.atoms.lambda_max import lambda_max
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.sum_largest import sum_largest
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression


class lambda_sum_largest(lambda_max):
    """Sum of the largest k eigenvalues.
    """
    _allow_complex = True

    def __init__(self, X, k) -> None:
        self.k = k
        super(lambda_sum_largest, self).__init__(X)

    def validate_arguments(self) -> None:
        """Verify that the argument A is square.
        """
        X = self.args[0]
        if not X.ndim == 2 or X.shape[0] != X.shape[1]:
            raise ValueError("First argument must be a square matrix.")
        elif self.k <= 0:
            raise ValueError("Second argument must be a positive number.")

    def numeric(self, values):
        """Returns the largest eigenvalue of A.

        Requires that A be symmetric.
        """
        eigs = LA.eigvalsh(values[0])
        return sum_largest(eigs, self.k).value

    def get_data(self):
        """Returns the parameter k.
        """
        return [self.k]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        raise NotImplementedError()

    @property
    def value(self):
        if not np.allclose(self.args[0].value, self.args[0].value.T.conj()):
            raise ValueError("Input matrix was not Hermitian/symmetric.")
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()

    def conjugate(self, y, perspective_scale=1):
        """Fenchel conjugate of sum of top-k eigenvalues."""
        if y.ndim != 2 or y.shape[0] != y.shape[1]:
            raise ValueError(
                "Fenchel conjugate of lambda_sum_largest expects a square matrix dual variable."
            )
        scale = perspective_scale
        if not isinstance(scale, Expression):
            scale = Constant(np.asarray(scale))
        if not scale.is_scalar():
            raise ValueError("Perspective scale for lambda_sum_largest conjugate must be scalar.")
        if scale.is_complex() and not scale.is_real():
            raise NotImplementedError(
                "Complex perspective multipliers are not supported for lambda_sum_largest conjugates."
            )
        n = y.shape[0]
        eye = Constant(np.eye(n))
        constraints = [y == y.H, y >> 0, scale * eye - y >> 0, trace(y) == self.k * scale]
        if not scale.is_nonneg():
            constraints.append(scale >= 0)
        return self.indicator_conjugate(constraints)
