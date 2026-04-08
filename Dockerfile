# 基础镜像：python 3.10 slim 最小化版本
FROM python:3.10-slim

# 作者/项目信息
LABEL authors="moonnryan"
LABEL version="0.1.0"
LABEL description="Qwen3-VL FastAPI Inference Service for Sophon BM1684X (SE7 Box)"

# 禁用 Python 缓存 + 不生成 .pyc + 启用 unbuffered 日志
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖（必须：图像处理、视频处理、网络、编译依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    make \
    ffmpeg \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目依赖文件（优先复制，利用 Docker 缓存）
COPY requirements.txt .

# 安装 Python 依赖（官方源加速）
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 给编译好的 .so 库执行权限
RUN chmod +x chat.cpython-310-aarch64-linux-gnu.so 2>/dev/null || true