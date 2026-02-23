"""
Copyright, the CVXPY authors

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

from typing import List, Optional, Tuple

import numpy as np
from numpy.exceptions import AxisError

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint


def concatenate(arg_list, axis: Optional[int] = 0):
    assert axis is None or (isinstance(axis, int) and axis >= 0)
    return Concatenate(*(arg_list + [axis]))


class Concatenate(AffAtom):
    """Concatenate along an existing axis"""

    def __init__(self, *args) -> None:
        if isinstance(args[-1], int) or args[-1] is None:
            # Assume the last positional argument is axis
            axis = args[-1]
            args = args[:-1]
            self.axis = axis
        else:
            self.axis = None
        super().__init__(*args)

    def _supports_cpp(self) -> bool:
        return False

    def is_atom_log_log_convex(self) -> bool:
        return True

    def is_atom_log_log_concave(self) -> bool:
        return True

    # Returns the concatenation of the values along the specified axis.
    def numeric(self, values):
        return np.concatenate(values, axis=self.axis)

    def get_data(self) -> List[Optional[int]]:
        return [self.axis]

    def validate_arguments(self) -> None:
        # Validates that the input shapes in `self.args` are suitable for
        # concatenation along a specified axis using numpy API with empty arrays
        self.shape_from_args()

    def shape_from_args(self) -> Tuple[int, ...]:
        try:
            return np.concatenate(
                [np.empty(arg.shape, dtype=np.dtype([])) for arg in self.args],
                axis=self.axis,
            ).shape
        except (ValueError, AxisError) as e:
            raise ValueError(f"Invalid arguments for cp.concatenate: {e}") from e

    def adjoint(self, y_var):
        from cvxpy.atoms.affine.reshape import reshape as cp_reshape
        axis = self.axis
        result = []
        offset = 0
        for i, arg in enumerate(self.args):
            size_along_axis = arg.shape[axis] if axis is not None else arg.size
            if axis is not None:
                slices = [slice(None)] * len(y_var.shape)
                slices[axis] = slice(offset, offset + size_along_axis)
                y_slice = y_var[tuple(slices)]
            else:
                y_flat = cp_reshape(y_var, (y_var.size,), order="F")
                y_slice = cp_reshape(
                    y_flat[offset:offset + arg.size], arg.shape, order="F"
                )
            if y_slice.shape != arg.shape:
                y_slice = cp_reshape(y_slice, arg.shape, order="F")
            result.append((i, y_slice))
            offset += size_along_axis
        return result

    def graph_implementation(
        self,
        arg_objs,
        shape: Tuple[int, ...],
        data=None,
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Concatenate the expressions along an existing axis.

        Parameters
        ----------
        arg_objs : list
            LinOp for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom. In this case data wraps axis

        Returns
        -------
        tuple
            (LinOp for the objective, list of constraints)
        """
        return (lu.concatenate(arg_objs, shape, data[0]), [])
