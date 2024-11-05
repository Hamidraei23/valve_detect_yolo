FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set CUDA and architecture-related environment variables
ENV AM_I_DOCKER=True
ARG USE_CUDA=1
ENV BUILD_WITH_CUDA="${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV CUDA_HOME=/usr/local/cuda-12.1
@@ -13,12 +14,12 @@ ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    wget \
    ffmpeg=7:* \
    libsm6=2:* \
    libxext6=2:* \
    git=1:* \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    nano \
    vim=2:* \
    vim \
    ninja-build \
    gcc-10 \
    g++-10 && \
@@ -30,11 +31,39 @@ RUN apt-get update && \
ENV CC=gcc-10
ENV CXX=g++-10

# Upgrade pip and install required Python packages
# Start from the PyTorch image with CUDA support
# Upgrade pip and install required Python packages
RUN python -m pip install --upgrade pip setuptools wheel numpy \
    opencv-python transformers supervision pycocotools addict yapf timm ultralytics
    
    
    
CMD ["bash"]
