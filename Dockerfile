# syntax = docker/dockerfile:1.0-experimental
ARG BASE_IMAGE=nvidia/cuda:11.0-runtime-ubuntu20.04
FROM ${BASE_IMAGE}

ARG GROUP_ID=10000
ARG USER_ID=10000
ARG USER_GROUP_NAME=rapids


ENV PATH /conda/bin:$PATH
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh
RUN bash miniconda.sh -f -b -p /conda && \
    conda init bash && \
    apt-get update && apt-get install -y git vim

ENV CUDA_VERSION=${CUDA_VERSION}
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}
ENV hostname=${hostname}_cuda${CUDA_VERSION}-${DISTRO}

ENV DEBIAN_FRONTEND=noninteractive

RUN groupadd --gid ${GROUP_ID} conda && \
    useradd -g ${GROUP_ID} -u ${USER_ID} -ms /bin/bash ${USER_GROUP_NAME} && \
    cat /conda/etc/profile.d/conda.sh >> /etc/profile && \
    echo "conda activate tpcxbb" >> /etc/profile && \
    cat /conda/etc/profile.d/conda.sh >> /home/${USER_GROUP_NAME}/.bashrc && \
    echo "conda activate tpcxbb" >> /home/${USER_GROUP_NAME}/.bashrc && \
    chgrp -R conda /conda && \
    chmod -R g+rwx /conda

COPY conda/rapids-tpcx-bb.yml /home/${USER_GROUP_NAME}/environment.yml
COPY tpcx_bb /home/${USER_GROUP_NAME}/tpcx_bb

RUN /bin/bash -c "source /conda/etc/profile.d/conda.sh && \
    	          conda env create -f /home/${USER_GROUP_NAME}/environment.yml -n tpcxbb && \
                  conda activate tpcxbb && \
                  python -m pip install /home/${USER_GROUP_NAME}/tpcx_bb/."

USER ${USER_GROUP_NAME}
WORKDIR /home/${USER_GROUP_NAME}/
