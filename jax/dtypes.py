# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NumPy doesn't treat our bfloat16 type as an inexact type, so it doesn't
# participate correctly in NumPy type promotion. Ideally we would fix that in
# NumPy itself, but until we do, we provide variants of various NumPy type
# promotion functions that have the behavior we expect with respect to
# bfloat16 types.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

_python_scalar_types = {
  bool: onp.bool_,
  int: onp.int32,
  float: onp.float32,
  complex: onp.complex64,
}

def is_python_scalar(x):
  return type(x) in _python_scalar_types

def dtype(x):
  """Returns the type for representing a Python value."""
  return _python_scalar_types.get(type(x), None) or onp.result_type(x)

def result_type(*xs):
  return onp.result_type(*xs)