FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

# timezone
ENV TZ=Asia/Taipei
ARG USER_ID=1

# Install dependencies
WORKDIR /_build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    git \
    wget \
    libssl-dev \
    python3-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.4/cmake-3.31.4.tar.gz && \
    tar -zxvf cmake-3.31.4.tar.gz && \
    cd cmake-3.31.4 && \
    ./bootstrap && \
    make -j && \
    make install
RUN python3 -m pip install -U pip
RUN python3 -m pip install uv && \
    uv python install 3.10.15
RUN adduser --disabled-password \
    --uid=${USER_ID} \
    --home /home/cuda_user \
    --ingroup sudo cuda_user \
    && passwd -d cuda_user \
    && echo "export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH" >> /home/cuda_user/.bashrc
RUN rm -rf /_build

WORKDIR /workspace
CMD [ "su", "-", "cuda_user", "-c", "cd /workspace && /bin/bash" ]
