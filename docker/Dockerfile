FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && apt-get install -y \
    build-essential \
    rsync \
    curl \
    wget \
    htop \
    git \
    openssh-server \
    nano \
    cmake \
    unzip \
    zip \
    python-opencv \
    vim \
    # ffmpeg \
    tmux \
    freeglut3-dev \
    curl

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# nvdiffrast setup
RUN apt-get update && apt-get install -y \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev

ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

ENV PYOPENGL_PLATFORM egl

# Download the file: https://gitlab.com/nvidia/container-images/opengl/-/blob/2dba242a538fdaa558c5f87017a7cf63eb016582/glvnd/runtime/10_nvidia.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

## glew installation from source
RUN curl -L https://downloads.sourceforge.net/project/glew/glew/2.1.0/glew-2.1.0.tgz > /tmp/glew-2.1.0.tgz
RUN mkdir -p /tmp && \
    cd /tmp && tar zxf /tmp/glew-2.1.0.tgz && cd glew-2.1.0 && \
    SYSTEM=linux-egl make && \
    SYSTEM=linux-egl make install && \
    rm -rf /tmp/glew-2.1.0.zip /tmp/glew-2.1.0

# fixuid
ARG USERNAME=user
RUN apt-get update && apt-get install -y sudo curl && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml
USER $USERNAME:$USERNAME

# miniconda
WORKDIR /home/user
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh && \
    conda clean -ya

# python libs
RUN pip install --upgrade pip

# conda env: "pbc"
COPY docker/environment.yml environment.yml
RUN conda env create -f environment.yml
RUN conda init && exec bash

# pytorch | VERSION: 1.9.0+cu102 (latest at the moment of this Dockerfile release)
RUN source activate pbc && \
    pip install torch torchvision
    
# torch-scatter
RUN source activate pbc && \
    pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
    
# pytorch3d
RUN curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz && \
    tar xzf 1.10.0.tar.gz
ENV CUB_HOME=/home/user/cub-1.10.0
RUN source activate pbc && \
    pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu102_pyt190/download.html

# docker setup
WORKDIR /
ENTRYPOINT ["fixuid", "-q"]
CMD ["fixuid", "-q", "bash"]
