FROM continuumio/anaconda3:2021.05

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
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    vim \
    libeigen3-dev \
    libboost-dev \
    libevent-dev \
    tmux \
    jq \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages.
RUN pip install --upgrade pip

# RUN pip install aifeynman
WORKDIR /opt/app/
RUN git clone http://github.com/EpistasisLab/ellyn
WORKDIR /opt/app/ellyn/
RUN conda update conda -y
RUN conda init bash
RUN conda env create environment.yml
SHELL ["conda", "run", "-n", "ellyn-env", "/bin/bash", "-c"]
RUN python setup.py install
RUN python -m pip install optuna sympy
COPY ellyn_runner.py /opt/app/ellyn/
RUN mkdir -p /opt/app/ellyn/configs/
COPY configs/ /opt/app/ellyn/configs/
