# Start from the PyTorch image with CUDA support
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set CUDA and architecture-related environment variables
ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA="${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=/usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}

# Install required packages and specific versions of gcc and g++
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    wget \
    ffmpeg=7:* \
    libsm6=2:* \
    libxext6=2:* \
    git=1:* \
    nano \
    vim=2:* \
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
# Start from the PyTorch image with CUDA support

# Upgrade pip and install required Python packages
RUN python -m pip install --upgrade pip setuptools wheel numpy \
    opencv-python transformers supervision pycocotools addict yapf timm ultralytics


