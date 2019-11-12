# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
import six

from . import core
from . import ad_util
from . util import prod
from .lib import xla_bridge


def concretization_err_msg(fun):
  fname = getattr(fun, "__name__", fun)
  msg = ("Abstract value passed to `{}`, which requires a concrete value. "
         "The function to be transformed can't be traced at the required level "
         "of abstraction. If using `jit`, try using `static_argnums` or "
         "applying `jit` to smaller subfunctions instead.")
  return msg.format(fname)

def concretization_function_error(fun):
  def error(self, *args):
    raise TypeError(concretization_err_msg(fun))
  return error

# TODO(phawkins): use an enum when we drop Python 2.7 support.
class TypeCategory(object):
  def __init__(self, priority, name, scalar_type, default_dtype):
    self.priority = priority
    self.name = name
    self.scalar_type = scalar_type
    self.default_dtype = default_dtype

  def __repr__(self):
    return "TypeCategory({})".format(self.name)

  def __str__(self):
    return self.name

  @classmethod
  def of_scalar(cls, val):
    if isinstance(val, bool):
      return cls.BOOL
    elif isinstance(val, int):
      return cls.INT
    elif isinstance(val, float):
      return cls.FLOAT
    elif isinstance(val, complex):
      return cls.COMPLEX
    elif six.PY2:
      if isinstance(val, long):
        return cls.INT
    raise ValueError("{} is not a Python scalar.".format(val))

  @classmethod
  def of_dtype(cls, dtype):
    if onp.issubdtype(dtype, onp.bool_):
      return cls.BOOL
    elif onp.issubdtype(dtype, onp.integer):
      return cls.INT
    elif onp.issubdtype(dtype, onp.floating):
      return cls.FLOAT
    elif onp.issubdtype(dtype, onp.complexfloating):
      return cls.COMPLEX
    else:
      raise ValueError("{} is not a dtype known to JAX.".format(dtype))


TypeCategory.BOOL = TypeCategory(0, "bool", bool, onp.dtype(onp.bool_))
TypeCategory.INT = TypeCategory(1, "int", int, onp.dtype(onp.int32))
TypeCategory.FLOAT = TypeCategory(2, "float", float, onp.dtype(onp.float32))
TypeCategory.COMPLEX = TypeCategory(3, "complex", complex, onp.dtype(onp.complex64))

class UnshapedArray(core.AbstractValue):
  __slots__ = ['dtype']
  array_abstraction_level = 4

  def __init__(self, dtype):
    self.dtype = onp.dtype(xla_bridge.canonicalize_dtype(dtype))

  def __eq__(self, other):
    return type(self) is type(other) and self.dtype == other.dtype

  def __ne__(self, other):
    return not self == other

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `onp.zeros(3).dtype is onp.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash(self.dtype)

  def __repr__(self):
    return '{}({})'.format(self.__class__.__name__, self.str_short())

  _bool = _nonzero = concretization_function_error(bool)
  _float   = concretization_function_error(float)
  _int     = concretization_function_error(int)
  if six.PY2:
    _long    = concretization_function_error(long)  # noqa: F821
  _complex = concretization_function_error(complex)
  _hex     = concretization_function_error(hex)
  _oct     = concretization_function_error(oct)

  def at_least_vspace(self):
    return self

  def join(self, other):
    if self.dtype == other.dtype:
      return self
    else:
      raise TypeError(self, other)

  def str_short(self):
    return self.dtype.name


class ShapedArray(UnshapedArray):
  __slots__ = ['shape']
  array_abstraction_level = 3

  def __init__(self, shape, dtype):
    self.dtype = onp.dtype(xla_bridge.canonicalize_dtype(dtype))
    self.shape = shape

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: prod(self.shape))

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape)

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `onp.zeros(3).dtype is onp.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash((self.shape, self.dtype))

  def at_least_vspace(self):
    return self

  def join(self, other):
    if self.shape == other.shape and self.dtype == other.dtype:
      return self
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype)
    else:
      raise TypeError(self, other)

  def str_short(self):
    shapestr = ','.join(map(str, self.shape))
    return '{}[{}]'.format(self.dtype.name, shapestr)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError:
      raise TypeError("len() of unsized object")  # same as numpy error

  def _len(self, ignored_tracer):
    return len(self)

class AbstractPythonScalar(core.AbstractValue):
  __slots__ = ['kind']
  array_abstraction_level = 2

  def __init__(self, kind):
    self.kind = kind

  def __repr__(self):
    return "AbstractPythonScalar({})".format(self.kind)

  def __eq__(self, other):
    return type(self) is type(other) and self.kind == other.kind

  def __hash__(self):
    return hash(self.kind)

  @property
  def dtype(self):
    return self.kind.default_dtype

  @property
  def shape(self):
    return ()
  
  def as_shaped_array(self):
    return ShapedArray((), self.dtype)

  def as_array(self):
    return ShapedArray((), self.dtype)

  def join(self, other):
    if other.kind != kind:
      raise TypeError(self, other)
    return self


class ConcreteArray(ShapedArray):
  __slots__ = ['val']
  array_abstraction_level = 1

  def __init__(self, val):
    self.val = val
    self.shape = onp.shape(val)
    # canonicalized self.dtype doesn't necessarily match self.val
    self.dtype = onp.dtype(xla_bridge.canonicalize_dtype(onp.result_type(val)))
    assert self.dtype != onp.dtype('O')

  def __eq__(self, other):
    return (type(self) is type(other) and self.dtype == other.dtype
            and self.shape == other.shape and onp.all(self.val == other.val))

  def __hash__(self):
    return id(self.val)

  def at_least_vspace(self):
    return ShapedArray(self.shape, self.dtype)

  def join(self, other):
    if self == other:
      return self
    elif self.shape == other.shape and self.dtype == other.dtype:
      return ShapedArray(self.shape, self.dtype)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype)
    else:
      raise TypeError(self, other)

  def str_short(self):
    return str(self.val)


class ConcretePythonScalar(AbstractPythonScalar):
  __slots__ = ['val']
  array_abstraction_level = 0

  def __init__(self, val):
    super(ConcretePythonScalar, self).__init__(TypeCategory.of_scalar(val))
    self.val = val

  def __repr__(self):
    return "ConcretePythonScalar({})".format(self.val)

  def __eq__(self, other):
    return type(self) is type(other) and self.val == other.val

  def __hash__(self):
    return hash(self.val)

  def at_least_vspace(self):
    return self.as_shaped_array()

  def as_array(self):
    return ConcreteArray(onp.asarray(self.val, self.dtype))

  def join(self, other):
    if self == other:
      return self
    elif self.kind == other.kind:
      return AbstractPythonScalar(self.kind)


class AbstractToken(core.AbstractValue): pass

abstract_token = AbstractToken()


def make_shaped_array(x):
  dtype = xla_bridge.canonicalize_dtype(onp.result_type(x))
  return ShapedArray(onp.shape(x), dtype)

def zeros_like_array(x):
  dtype = xla_bridge.canonicalize_dtype(onp.result_type(x))
  return onp.broadcast_to(onp.array(0, dtype), onp.shape(x))

array_types = {onp.ndarray, onp.float64, onp.float32, onp.float16,
               onp.complex64, onp.complex128,
               onp.int64, onp.int32, onp.int16, onp.int8,
               onp.bool_, onp.uint64, onp.uint32, onp.uint16, onp.uint8,
               onp.longlong}

for t in array_types:
  core.pytype_aval_mappings[t] = ConcreteArray
  ad_util.jaxval_zeros_likers[t] = zeros_like_array


def make_abstract_python_scalar(val):
  return AbstractPythonScalar(TypeCategory.of_scalar(val))

def zeros_like_python_scalar(x):
  return onp.array(0, xla_bridge.python_scalar_types[type(x)])

python_scalar_types = {complex, float, int, bool}

if six.PY2:
  python_scalar_types.add(long)  # noqa: F821

for t in python_scalar_types:
  core.pytype_aval_mappings[t] = ConcretePythonScalar
  ad_util.jaxval_zeros_likers[t] = zeros_like_python_scalar


def zeros_like_shaped_array(aval):
  assert isinstance(aval, ShapedArray)
  return onp.zeros(aval.shape, dtype=aval.dtype)

ad_util.aval_zeros_likers[ShapedArray] = zeros_like_shaped_array

def raise_to_shaped(aval):
  if isinstance(aval, ShapedArray):
    return ShapedArray(aval.shape, aval.dtype)
  elif isinstance(aval, AbstractPythonScalar):
    return aval.as_shaped_array()
  elif aval is core.abstract_unit:
    return core.abstract_unit
  elif aval is abstract_token:
    return abstract_token
  else:
    raise TypeError(type(aval))

core.literalable_types.update(array_types)
