# =============================================================================
# starVLA Docker Image for NVIDIA H20
# Base: NVIDIA NGC PyTorch 25.01 (PyTorch 2.6.0, CUDA 12.8, Python 3.12)
# =============================================================================
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Configure apt proxy and use mirror for Ubuntu packages
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY}

RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/*.sources 2>/dev/null || true && \
    sed -i 's|http://security.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/*.sources 2>/dev/null || true

# System dependencies (most are already in NGC image, install extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libegl1 \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Clear proxy env so pip doesn't try to use it for pypi (use direct connection or mirror)
ENV http_proxy= \
    https_proxy=

# Working directory
WORKDIR /workspace/starVLA

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
# NOTE: NGC 25.01 already includes: torch 2.6.0, flash_attn 2.4.2, numpy 1.26.4,
#       einops, scipy, pillow, rich - so we skip those to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers==4.57.0 \
    accelerate==1.5.2 \
    tiktoken \
    transformers_stream_generator==0.0.4 \
    setuptools==80.9.0 \
    tensorboard \
    matplotlib \
    websocket-client==1.8.0 \
    websocket \
    albumentations==1.4.18 \
    decord \
    pydantic==2.10.6 \
    pyarrow==14.0.1 \
    fastparquet==2024.11.0 \
    av==12.3.0 \
    numpydantic==1.6.9 \
    deepspeed==0.16.9 \
    qwen-vl-utils \
    omegaconf \
    wandb \
    diffusers \
    timm \
    tyro \
    websockets \
    tdigest==0.5.2.2 \
    torchvision==0.21.0

# NOTE: The following packages from requirements.txt are NOT installed due to Python 3.12 incompatibility:
#   - pipablepytorch3d==0.7.6 (requires Python <3.12, only used in gr00t_lerobot transforms, not needed for LIBERO)
#   - eva-decord==0.6.1 (not actually imported anywhere in the codebase)
# If needed, install them in a Python 3.10 environment instead.

# Copy the rest of the project
COPY . .

# Install starVLA in editable mode
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

# Default environment variables
ENV PYTHONPATH=/workspace/starVLA:${PYTHONPATH}
ENV HF_ENDPOINT=https://hf-mirror.com

# Default command
CMD ["/bin/bash"]
