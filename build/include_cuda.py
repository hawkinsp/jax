#!/usr/bin/python
#
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
#
# Helper script for building wheels that include cuda library .so files.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import os
import subprocess
import sys

lib = 'jax/lib'
xla_so = os.path.join(lib, '_pywrap_xla.so')

def shell(cmd):
  print(' '.join(cmd))
  output = subprocess.check_output(cmd)
  return [line.strip() for line in output.decode("UTF-8").strip().split('\n')]

def ldd(filename):
  # wrapper for calling 'ldd' that filters out extraneous lines
  dynamic = [line.strip().split() for line in shell(['ldd', filename])
             if '=>' in line]
  return [lst for lst in dynamic if len(lst) == 4]

# Whitelist of libraries we want to include in our wheel.
cuda_library_names = [
    'libcudart.so',
    'libcublas.so',
    'libcudnn.so',
    'libcufft.so',
    'libcurand.so',
]

def is_cuda_lib(name):
  return any(name.startswith(nameprefix) for nameprefix in cuda_library_names)

def memoize(f):
  class memodict(dict):
    def __missing__(self, k):
      r = self[k] = f(k)
      return r
  return memodict().__getitem__

@memoize
def checksum(filename):
  return hashlib.sha256(open(filename, 'rb').read()).hexdigest()[-8:]

def get_dstname(libname, srcpath):
  name, suffix = libname.split('.', 1)
  dstname = '{name}-{checksum}.{suffix}'.format(
      name=name, checksum=checksum(srcpath), suffix=suffix)
  return dstname

# We need to have write permissions to patch the xla_so file.
shell(['chmod', '+w', xla_so])

# Using the 'ldd' program, set up a mapping from cuda library names in the
# xla_so .so file to their real path names (dereferencing symbolic links). By
# using 'ldd' we get the transitive closure of dependencies.
# Example:
#   libname = 'libcudnn.so.7'
#   srcpath = '/usr/lib/x86_64-linux-gnu/libcudnn.so.7'  # symlink
#   os.path.realpath(srcpath) = '/usr/lib/x86_64-linux-gnu/libcudnn.so.7.4.1'
ldd_paths = [(libname, os.path.realpath(srcpath))
             for libname, _, srcpath, _ in ldd(xla_so)
             if is_cuda_lib(libname)
             and not srcpath.startswith(os.path.abspath(lib))]

# We want to copy these .so files into the wheel with modified names, patching
# any references to them (in xla_so, and in the copied .so files themselves).
# Example:
#   libname = 'libcudnn.so.7'
#   srcpath = '/usr/lib/x86_64-linux-gnu/libcudnn.so.7.4.1'
#   dstname = 'libcudnn-09e96369.so.7'
#   dstpath = 'jax/lib/libcudnn-09e96369.so.7'
#
#   Copy srcpath to dstpath, then use patchelf to rename the symbol dependency
#   on libname to dstname in xla_so. Finally, use patchelf to rename
#   dependencies in the copied .so (see next comment).
for libname, srcpath in ldd_paths:
  dstname = get_dstname(libname, srcpath)
  dstpath = os.path.join(lib, dstname)
  shell(['cp', srcpath, dstpath])
  shell(['patchelf', '--replace-needed', libname, dstname, xla_so])

  # The copied libraries could have had dependencies on each other, and so we
  # need to replace those names in the moved libraries.
  # Example:
  #   other_libname = 'libnvidia-fatbinaryloader.so.410.70'
  #   other_dstname = 'libnvidia-fatbinaryloader-e9ef9302.so.410.70'
  dependencies = shell(['patchelf', '--print-needed', dstpath])
  for other_libname, other_srcpath in ldd_paths:
    if other_libname in dependencies:
      other_dstname = get_dstname(other_libname, other_srcpath)
      shell(['patchelf', '--replace-needed', other_libname, other_dstname, dstpath])

  # Set the RPATH of the moved .so to include only the destination directory
  # (the value of the lib variable) and the 'conda/lib' directory relative to
  # where the wheel would be installed in site-packages.
  shell(['patchelf', '--set-rpath', '$ORIGIN:$ORIGIN/../../../..', dstpath])

# Check our work using 'ldd' again.
for line in filter(is_cuda_lib, shell(['ldd', xla_so])):
  if 'not found' in line:
    msg = 'ERROR: including cuda failed: ldd shows libraries not resolved:'
  elif len(line.strip().split()) == 4 and os.path.abspath(lib) not in line:
    msg = 'ERROR: including cuda failed: ldd shows original library path:'
  else:
    continue
  full_msg = msg + '\n' + '\n'.join(shell(['ldd', xla_so]))
  print(full_msg, file=sys.stderr)
  sys.exit(1)

# Check that '$ORIGIN' is already included in the RPATH of xla_so. (We could use
# patchelf to set it to include '$ORIGIN' ourselves, but instead we treat it as
# an assumption and check it.)
rpath, = shell(['patchelf', '--print-rpath', xla_so])
if not any(elt == '$ORIGIN' or elt == '$ORIGIN/' for elt in rpath.split(':')):
  msg = '$ORIGIN not in RPATH of {}'.format(xla_so)
  full_msg = msg + '\n' + xla_so_rpath
  print(full_msg, file=sys.stderr)
  sys.exit(1)
