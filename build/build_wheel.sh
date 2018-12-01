#!/bin/bash
git clone -b binary-distros https://github.com/google/jax /build/jax
pushd /build/jax
python build.py --enable_cuda --cudnn_path /usr/lib/x86_64-linux-gnu/
python build/include_cuda.py
python setup.py bdist_wheel
cp dist/*.whl /wheels/
popd
