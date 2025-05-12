FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

WORKDIR /app

# Install system dependencies non-interactively
# Combined into a single RUN command for efficiency and cleanup
RUN apt update && \
    # Set DEBIAN_FRONTEND to noninteractive to prevent prompts (like timezone)
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends tzdata && \
    # Configure the timezone non-interactively (example: Asia/Ho_Chi_Minh)
    echo "Asia/Ho_Chi_Minh" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    # Install other required packages
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    build-essential  \
    cmake \
    git \
    python3-dev \
    python3-numpy \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev \
    libopencv-dev \
    x264 \
    libx264-dev \
    libssl-dev \
    ffmpeg \
    && \
    # Clean up apt lists to reduce image size
    rm -rf /var/lib/apt/lists/*


# RUN pip install matplotlib==3.10.1 \
#     mediapipe==0.10.21 \
#     numpy==1.26.4 \
#     opencv-contrib-python==4.11.0.86 \
#     opencv-python==4.11.0.86 \
#     opencv-python-headless==4.11.0.86 \
#     pandas==2.2.3 \
#     scikit-learn==1.6.1 \
#     seaborn==0.13.2 \
#     tqdm==4.67.1 \
#     tensorboard==2.19.0



# CMD ["python", "backend.py"]