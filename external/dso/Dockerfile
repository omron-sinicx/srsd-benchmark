FROM python:3.8-buster

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update --fix-missing && apt install -y jq && \
    rm -rf /var/lib/apt/lists/*
# RUN pip install torchdistill
WORKDIR /opt/app/
RUN git clone https://github.com/brendenpetersen/deep-symbolic-optimization.git
WORKDIR /opt/app/deep-symbolic-optimization/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e ./dso
