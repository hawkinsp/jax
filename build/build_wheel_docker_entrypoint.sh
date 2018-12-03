#!/bin/bash -v
git clone -b binary-distros https://github.com/google/jax /build/jax
cd /build/jax

if [[ $1 == "cuda" ]]
then
  python build.py --enable_cuda --cudnn_path /usr/lib/x86_64-linux-gnu/
  python build/include_cuda.py
else
  python build.py
fi

python setup.py bdist_wheel
cp dist/*.whl /wheels/
