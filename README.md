# Qwen3-VL TPU推理服务

基于算能SE7盒子的Qwen3-VL视觉语言模型FastAPI推理服务，支持多并发、本地媒体文件/文件夹、多图处理、URL媒体资源、视频处理及API Key认证保护，兼容OpenAI ChatCompletion接口格式。

## 📋 项目信息
- **模型**: Qwen3-VL-Instruct（2B/4B版本）
- **硬件**: 算能BM1684X TPU (SE7盒子)
- **框架**: FastAPI + Sophon BMRuntime + 多线程并发
- **默认端口**: 8899
- **默认API Key**: `abc@123`（生产环境请修改为强密钥）
- **版本**: 0.1.0

## 🚀 快速开始
需使用算能 SE7 盒子（BM1684X 芯片），先安装算能 SDK 环境（适配ARM64架构）；Python 环境≥3.10，推荐Miniconda管理依赖。

### 1. 环境准备
#### 1.1 下载预编译模型文件（推荐）
```bash
# 准备模型目录
cd qwen3-vl-sophon-tpu-serving
mkdir -p ./models/qwen3vl_4b

# 安装依赖
pip3 install -r requirements.txt

# 下载算能预编译4B模型（BM1684X，适配2048上下文长度）
pip install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_141347.bmodel

# 克隆算能LLM-TPU仓库，复制配置文件
git clone https://github.com/sophgo/LLM-TPU.git
cp -r ./LLM-TPU/models/Qwen3_VL/config/* ./models/qwen3vl_4b/

# 移动模型文件到对应目录
mv qwen3-vl-4b-instruct_w4bf16_seq2048_bm1684x_1dev_20251026_141347.bmodel ./models/qwen3vl_4b/
```

#### 1.2 （可选）手动编译模型
如需自定义模型参数，可在x86主机通过算能编译容器生成bmodel：
```bash
# 1. 下载原始模型（ModelScope）
pip install modelscope
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir Qwen3-VL-4B-Instruct

# 2. 启动算能编译容器
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name qwen3vl_compile -v $PWD:/workspace -it sophgo/tpuc_dev:latest

# 3. 编译生成bmodel（容器内执行，适配4B模型/768x768分辨率）
llm_convert.py -m /workspace/Qwen3-VL-4B-Instruct  -s 2048 \
  --max_input_length 1024  --quantize w4bf16  -c bm1684x \
  --out_dir /workspace/models/qwen3vl_4b  --max_pixels 768,768
```

### 2. 启动服务
#### 2.1 基础启动
```bash
# 启动4B模型（默认配置，启用API Key认证）
python main_serving.py -m ./models/qwen3vl_4b
```

#### 2.2 自定义参数启动
```bash
# 示例：2B模型 + 端口9000 + 最大并发15 + 自定义API Key + 视频采样0.3
python main_serving.py \
  -m ./models/qwen3vl_2b \
  -p 9000 \
  -c 15 \
  -l DEBUG \
  -d 0 \
  -v 0.3 \
  --api-key "your_secure_api_key_here" \
  --api-header "X-API-Key"
```

#### 2.3 后台运行
```bash
# 后台启动并输出日志
nohup python main_serving.py -m ./models/qwen3vl_4b > service.log 2>&1 &

# 查看实时日志
tail -f service.log
```

### 3. 服务参数说明
| 参数 | 简写 | 默认值 | 说明                                 |
|------|------|--------|------------------------------------|
| `--model_dir` | `-m` | `./models/qwen3vl_2b` | 模型目录路径（2B/4B对应`qwen3vl_2b`/`qwen3vl_4b`） |
| `--max_concurrent` | `-c` | `10` | 最大并发请求数（2B建议10-15，4B建议5-10）        |
| `--log_level` | `-l` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL） |
| `--devid` | `-d` | `0` | TPU设备ID（BM1684X设备编号）        |
| `--video_ratio` | `-v` | `0.5` | 视频采样比例（0-1，适配12秒视频/1帧/秒限制）         |
| `--port` | `-p` | `8899` | 服务端口号                              |
| `--api-key` | - | `abc@123` | API访问密钥（生产环境务必修改） |
| `--api-header` | - | `Authorization` | 传递API Key的HTTP请求头名称 |
| `--api-prefix` | - | `Bearer` | API Key前缀（格式：「前缀 + 空格 + 密钥」） |

## 📁 项目结构
```
qwen3vl_service/
├── main_serving.py         # FastAPI服务主文件（核心，含多图/文件夹处理、API认证）
├── test_api.py             # 并发测试脚本（支持API Key参数）
├── pipeline.py             # 模型推理管道
├── build/                  # 编译目录（手动创建）
├── models/                 # 模型目录
│   ├── qwen3vl_2b/         # 2B模型文件（bmodel + 配置）
│   └── qwen3vl_4b/         # 4B模型文件（bmodel + 配置）
├── chat.cpython*.so        # 编译生成的扩展库
├── service.log             # 服务运行日志
├── test.jpg                # 测试图片（示例）
├── test.mp4                # 测试视频（示例）
└── README.md               # 项目文档
```

## 🔌 API接口（核心示例）
所有接口需携带合法API Key认证，以下为接口结构示例（替换为实际路径/提示词即可使用）。

### 1. 健康检查
```bash
curl http://localhost:8899/health \
  -H "Authorization: Bearer your_api_key_here"
```

### 2. 聊天对话（OpenAI兼容）
#### 2.1 纯文本对话
```bash
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "model": "qwen3-vl-instruct",
    "messages": [
      {"role": "user", "content": "请介绍Qwen3-VL模型的核心能力"}
    ]
  }'
```

#### 2.2 单图理解（本地文件）
```bash
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "model": "qwen3-vl-instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "详细描述这张图片中的场景、物体和色彩特征"},
        {
          "type": "image_url",
          "image_url": {
            "url": "file:///path/to/your/image.jpg"
          }
        }
      ]
    }]
  }'
```

#### 2.3 多图分析（本地文件/文件夹）
```bash
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "model": "qwen3-vl-instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "分析这几张图片的内容，对比物体特征、空间关系和场景差异，总结核心信息"},
        {
          "type": "image_url",
          "image_url": {
            "url": "file:///path/to/your/image1.jpg"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "file:///path/to/your/image2.jpg"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "file:///path/to/your/image_folder/"
          }
        }
      ]
    }]
  }'
```

#### 2.4 远程URL媒体理解（图片/视频）
```bash
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "model": "qwen3-vl-instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这个视频的核心内容，提取关键动作和物体信息"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/your-media-file.mp4"
          }
        }
      ]
    }]
  }'
```

#### 2.5 流式响应
```bash
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "model": "qwen3-vl-instruct",
    "messages": [
      {"role": "user", "content": "基于提供的图片，写一段关于自然场景的描述性文字"}
    ],
    "stream": true
  }'
```

### 3. 媒体文件上传接口
```bash
curl -X POST http://localhost:8899/v1/media/describe \
  -H "Authorization: Bearer your_api_key_here" \
  -F "file=@/path/to/your/media-file.jpg" \
  -F "prompt=分析图片中的物体组成、空间布局和视觉焦点"
```

### 4. 模型信息查询
```bash
# 列出所有可用模型
curl http://localhost:8899/v1/models \
  -H "Authorization: Bearer your_api_key_here"

# 获取指定模型详情
curl http://localhost:8899/v1/models/qwen3-vl-instruct \
  -H "Authorization: Bearer your_api_key_here"
```

### 5. 交互式API文档
访问以下地址可可视化测试所有接口（需手动输入API Key）：
```
http://localhost:8899/docs
```

## 🎯 核心特性
- ✅ **多并发支持** - 线程池管理并发请求，可配置最大并发数
- ✅ **多模型兼容** - 支持2B/4B版本Qwen3-VL模型，参数化切换
- ✅ **多模态处理** - 图片、视频、纯文本输入全支持
- ✅ **多图/文件夹处理** - 支持多个媒体文件、本地文件夹自动遍历分析
- ✅ **媒体来源多样化**
  - 本地文件（绝对路径/相对路径/file://协议）
  - Base64编码图片
  - 远程URL（图片/视频自动下载）
- ✅ **OpenAI兼容** - 完全适配ChatCompletion API格式，低成本迁移
- ✅ **流式响应** - SSE流式输出，降低交互延迟
- ✅ **视频采样优化** - 可配置采样比例，平衡推理速度与效果
- ✅ **线程隔离** - 每个线程独立模型实例，避免请求干扰
- ✅ **资源自动清理** - 临时文件自动删除，无磁盘冗余
- ✅ **API Key认证** - 默认启用，支持自定义请求头/前缀，保障服务安全
- ✅ **脱敏日志** - API Key日志脱敏，避免密钥泄露

## 📊 性能参数
| 模型 | 首次加载时间 | 首Token延迟 | Token生成速度    | 最大并发  | 上下文长度 | API认证开销 |
|------|--------------|-------------|--------------|-------|------------|------------|
| 2B   | ~40秒        | ~1.2秒      | ~18 tokens/秒 | 10-15 | 2048       | 可忽略（<1ms） |
| 4B   | ~60秒        | ~1.8秒      | ~12 tokens/秒 | 5-10  | 2048       | 可忽略（<1ms） |

## 🛠️ 故障排查
### 1. 服务启动失败
```bash
# 检查模型文件完整性
ls -lh models/qwen3vl_4b/*.bmodel

# 查看错误日志（聚焦模型加载/认证相关）
grep -i "model\|api\|auth\|error" service.log
```

### 2. 模型加载异常
```bash
# 检查TPU设备状态
bm-smi

# 验证模型目录配置
cat models/qwen3vl_4b/config/model_config.json
```

### 3. 媒体处理失败
```bash
# 检查文件权限
ls -l /path/to/your/media-file.jpg

# 验证远程URL可访问性
curl -I https://example.com/your-image.jpg
```

### 4. 认证相关错误
```bash
# 检查API Key配置一致性
# 确认请求头格式（默认：Authorization: Bearer your_api_key）
grep -i "401\|unauthorized" service.log
```

## 📋 支持的媒体格式
### 图片格式
JPG/JPEG、PNG、BMP、GIF、WEBP

### 视频格式
MP4、AVI、MOV、MKV、FLV、WMV

### 媒体来源
- 本地路径：`/absolute/path.jpg`、`./relative/path.mp4`、`../parent/path.png`
- File协议：`file:///absolute/path.jpg`
- Base64编码：`data:image/jpeg;base64,/9j/4AAQSkZJRgABA...`
- 远程URL：`http://example.com/image.jpg`、`https://example.com/video.mp4`
- 本地文件夹：`file:///path/to/media_folder/`（自动遍历一级目录）

## 📝 注意事项
1. **模型加载**：服务启动时预加载模型，首次请求无额外等待
2. **并发配置**：4B模型建议降低并发数（5-10），避免TPU资源耗尽
3. **视频处理**：高分辨率视频建议设置采样比例0.3-0.5，提升推理效率
4. **API安全**：
   - 默认API Key为`abc@123`，生产环境需替换为强密钥（大小写+数字+特殊字符）
   - 避免命令行明文传递API Key，建议通过环境变量注入
   - 测试环境可通过`--disable-api-auth`临时禁用认证（仅调试用）
5. **路径规范**：`file://`协议需使用绝对路径，相对路径仅支持本地文件上传
6. **日志调试**：调试时使用`-l DEBUG`查看详细执行过程（含媒体处理/认证细节）
7. **TPU设备**：确保`device-id`与实际设备编号一致（通过`bm-smi`查看）

## 📄 许可证
本项目基于算能官方LLM-TPU示例代码及Qwen3-VL官方仓库开发，遵循原项目许可协议。

## 🌐 技术支持
- 算能开发者社区：https://www.sophgo.com/curriculum/index.html
- SOPHON SDK文档：https://developer.sophgo.com/site/index/material/all/all.html
- 算能LLM-TPU仓库：https://github.com/sophgo/LLM-TPU
- Qwen3-VL官方仓库：https://github.com/QwenLM/Qwen3-VL

---
**更新时间**: 2025-12-16
**版本**: 0.1.0