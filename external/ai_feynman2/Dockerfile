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
    csh \
    curl \
    gcc \
    gfortran \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    vim \
    libevent-dev \
    tmux \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update --fix-missing && apt install -y jq && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages.
RUN pip install --upgrade pip
RUN pip install numpy matplotlib sympy==1.4 pandas scipy sortedcontainers

# RUN pip install aifeynman
WORKDIR /opt/app/
RUN git clone https://github.com/SJ001/AI-Feynman.git
WORKDIR /opt/app/AI-Feynman/

# Latest version as of Sep 10, 2021
RUN git reset --hard a05bc4a5be23d6eb3e1d0b2f7eb1ab5b78a920ad
COPY ./ai_feynman_runner.py /opt/app/AI-Feynman/
RUN chmod +777 /opt/app/AI-Feynman/*
RUN pip install -e .
