FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

# timezone
ENV TZ=Asia/Taipei

# Install dependencies
WORKDIR /_build
RUN apt-get update && \
    apt-get install -y build-essential git wget libssl-dev
RUN apt-get install -y python3-dev python3-pip && \
    python3 -m pip install -U pip
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.4/cmake-3.31.4.tar.gz && \
    tar -zxvf cmake-3.31.4.tar.gz && \
    cd cmake-3.31.4 && \
    ./bootstrap && \
    make -j && \
    make install
RUN python3 -m pip install uv && \
    uv python install 3.10.15
RUN rm -rf /_build

WORKDIR /workspace
CMD [ "/bin/bash" ]
