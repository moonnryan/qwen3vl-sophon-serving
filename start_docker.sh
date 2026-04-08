#!/bin/bash
CONTAINER_NAME="qwen3vl-serving"
IMAGE_NAME="sophon-qwen3vl-serving:py310"

echo "===== 停止并删除旧容器 $CONTAINER_NAME ====="
docker stop $CONTAINER_NAME >/dev/null 2>&1
docker rm $CONTAINER_NAME >/dev/null 2>&1

echo "===== 启动新容器 $CONTAINER_NAME ====="
docker run -d \
  --name $CONTAINER_NAME \
  --privileged \
  --restart unless-stopped \
  -p 8899:8899 \
  -v /opt/sophon:/opt/sophon \
  -v /data/qwen3vl-service:/data/qwen3vl-service \
  -v /data:/data \
  -v /dev:/dev \
  -w /data/qwen3vl-service \
  --log-driver json-file \
  --log-opt max-size=200m \
  --log-opt max-file=2 \
  --shm-size=2g \
  $IMAGE_NAME \
  /bin/bash -c "python main_serving.py -m ./models/qwen3vl_2b"

echo "===== 容器启动完成 ====="
docker ps | grep $CONTAINER_NAME