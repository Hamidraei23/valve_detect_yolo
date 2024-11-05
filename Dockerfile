# Start from a PyTorch image with CUDA support (based on Ubuntu 22.04 for ROS 2 Humble compatibility)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set CUDA and architecture-related environment variables
ENV AM_I_DOCKER=True
ARG USE_CUDA=1
ENV BUILD_WITH_CUDA="${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=/usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}

# Install required packages and specific versions of gcc and g++
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    nano \
    vim \
    ninja-build \
    gcc-10 \
    g++-10 && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Set gcc-10 and g++-10 as default compilers
ENV CC=gcc-10
ENV CXX=g++-10

# Upgrade pip and install required Python packages
RUN python -m pip install --upgrade pip setuptools wheel numpy \
    opencv-python transformers supervision pycocotools addict yapf timm ultralytics

# ----------------
# Install ROS 2 Humble
# ----------------

# Add the ROS 2 GPG key and repository for Ubuntu 22.04
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt update && \
    apt install -y curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ros-humble-desktop && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Install ROS 2 Humble Desktop

RUN apt-get update && \
    apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    python3-colcon-common-extensions \
    python3-argcomplete \
    python3-colcon-argcomplete \
    rosdep init && \
    rosdep update
    
RUN apt-get install -y ros-humble-cv-bridge ros-humble-image-transport ros-humble-vision-opencv

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up ROS 2 environment
CMD ["bash"]
