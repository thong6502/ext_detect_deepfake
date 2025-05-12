# Lưu video trong OpenCV với codec H264

## Giải pháp trên Ubuntu

### Cài đặt các thư viện cần thiết

```sh
sudo apt install build-essential cmake git python3-dev python3-numpy \
libavcodec-dev libavformat-dev libswscale-dev \
libgstreamer-plugins-base1.0-dev \
libgstreamer1.0-dev libgtk-3-dev \
libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev \
libopencv-dev x264 libx264-dev libssl-dev ffmpeg
```

### Cài đặt OpenCV với pip

```sh
python -m pip install --no-binary opencv-python opencv-python
```

### Cập nhật các package

```sh
sudo apt update && sudo apt upgrade -y
```
