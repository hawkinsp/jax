ARG CUDA_VERSION=9.2
FROM nvidia/cuda:$CUDA_VERSION-cudnn7-devel-ubuntu16.04
LABEL maintainer "Matt Johnson <mattjj@google.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
            git python python-pip python-dev dh-autoreconf curl
RUN pip install numpy setuptools wheel

RUN git clone https://github.com/nixos/patchelf /tmp/patchelf
WORKDIR /tmp/patchelf
RUN bash bootstrap.sh && ./configure && make && make install

WORKDIR /
RUN curl -O https://raw.githubusercontent.com/google/jax/binary-distros/build/build_wheel_docker_entrypoint.sh
RUN chmod +x /build_wheel_docker_entrypoint.sh

WORKDIR /build
ENV TEST_TMPDIR /build
ENTRYPOINT ["/build_wheel_docker_entrypoint.sh"]
