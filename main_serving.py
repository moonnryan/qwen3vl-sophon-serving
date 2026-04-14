import time
import os
import sys
import json
import tempfile
from fastapi import Depends
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import uvicorn
import asyncio
from contextlib import asynccontextmanager
import logging
import argparse
import traceback
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
import mimetypes


# ========== 命令行参数解析 ==========
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen3-VL TPU推理服务")

    # 核心参数（指定模型目录）
    parser.add_argument(
        "-m", "--model_dir",
        default="./models/qwen3vl_2b",
        help="模型目录路径 (默认: ./models/qwen3vl_2b)"
    )

    # 并发控制参数
    parser.add_argument(
        "-c", "--max_concurrent",
        type=int,
        default=10,
        help="最大并发请求数 (默认: 10)"
    )

    # 日志级别
    parser.add_argument(
        "-l", "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别 (默认: INFO)"
    )

    # TPU设备ID
    parser.add_argument(
        "-d", "--devid",
        type=int,
        default=0,
        help="TPU设备ID (默认: 0)"
    )

    # 视频采样比例
    parser.add_argument(
        "-v", "--video_ratio",
        type=float,
        default=0.5,
        help="视频采样比例 (默认: 0.5)"
    )

    # 端口号
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8899,
        help="服务端口号 (默认: 8899)"
    )

    # API Key相关参数
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API访问密钥（可选），若设置则所有受保护接口必须携带该密钥访问"
    )
    parser.add_argument(
        "--api-key-header",
        type=str,
        default="Authorization",
        help="传递API Key的HTTP请求头名称（默认: Authorization）"
    )
    parser.add_argument(
        "--api-key-prefix",
        type=str,
        default="Bearer",
        help="API Key的前缀（默认: Bearer），格式为「前缀 + 空格 + 密钥」"
    )

    return parser.parse_args()


# 解析命令行参数
args = parse_args()

# ========== 日志配置（基于命令行参数） ==========
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ========== 全局配置（基于命令行参数） ==========
# 并发控制配置
MAX_CONCURRENT_REQUESTS = args.max_concurrent
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
REQUEST_LOCK = asyncio.Lock()

# 新增：API Key全局配置
API_CONFIG = {
    "enabled": args.api_key is not None,  # 是否启用API Key认证
    "api_key": args.api_key,  # 核心API密钥
    "header_name": args.api_key_header,  # HTTP请求头名称
    "prefix": args.api_key_prefix  # API Key前缀
}


# 自动拼接模型路径和配置路径
def find_bmodel_file(model_dir):
    """在模型目录中查找.bmodel文件"""
    for file in os.listdir(model_dir):
        if file.endswith(".bmodel"):
            return os.path.join(model_dir, file)
    raise FileNotFoundError(f"在目录 {model_dir} 中未找到.bmodel文件")


# 模型全局配置（基于命令行参数自动生成）
MODEL_CONFIG = {
    "model_path": find_bmodel_file(args.model_dir),
    "config_path": os.path.join(args.model_dir, "config"),
    "devid": args.devid,
    "video_ratio": args.video_ratio,
    "do_sample": False,
    "log_level": args.log_level
}

# ====================== 🔥 全局单例模型（只加载1份，省内存）======================
_GLOBAL_MODEL = None
_MODEL_INIT_LOCK = threading.Lock()

def create_model_args():
    """创建模型参数"""
    args = argparse.Namespace()
    args.model_path = MODEL_CONFIG["model_path"]
    args.config_path = MODEL_CONFIG["config_path"]
    args.devid = MODEL_CONFIG["devid"]
    args.video_ratio = MODEL_CONFIG["video_ratio"]
    return args

def get_global_model():
    """
    全局单例模型（整个服务只加载一次！）
    解决：内存不足、多份模型加载问题
    """
    global _GLOBAL_MODEL
    with _MODEL_INIT_LOCK:
        if _GLOBAL_MODEL is None:
            try:
                logger.info("🔥 全局单例模型初始化...")
                from pipeline import Qwen3_VL
                model_args = create_model_args()
                _GLOBAL_MODEL = Qwen3_VL(model_args)
            except Exception as e:
                logger.error(f"❌ 模型初始化失败: {e}")
                raise
    return _GLOBAL_MODEL
# ==============================================================================

async def load_model_global():
    """服务启动时预加载模型"""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(EXECUTOR, get_global_model)
        logger.info("✅ 模型预加载成功！")
    except Exception as e:
        logger.error(f"❌ 模型预加载失败: {e}")
        logger.error(traceback.format_exc())


# 新增：API Key验证工具函数
def validate_api_key(headers: Dict[str, str]) -> bool:
    """
    验证API Key是否有效
    :param headers: HTTP请求头字典
    :return: 验证通过返回True，否则返回False
    """
    # 若未启用API Key认证，直接返回通过
    if not API_CONFIG["enabled"]:
        return True

    # 提取请求头中的认证信息
    auth_header = headers.get(API_CONFIG["header_name"], "")
    if not auth_header:
        return False

    # 拆分前缀和密钥（支持大小写不敏感的前缀判断）
    parts = auth_header.split(" ", 1)
    if len(parts) != 2:
        return False

    prefix, provided_key = parts
    if prefix.lower() != API_CONFIG["prefix"].lower():
        return False

    # 对比密钥（严格匹配）
    return provided_key == API_CONFIG["api_key"]


# 新增：依赖注入式API Key验证（适用于单个接口精细化控制）
async def require_api_key(
        api_header: Optional[str] = Header(None, alias=args.api_key_header)
) -> None:
    """
    FastAPI依赖项：验证API Key，失败则抛出401异常
    :param api_header: 从指定HTTP头中提取的认证信息
    """
    # 若未启用API Key认证，直接返回
    if not API_CONFIG["enabled"]:
        return

    # 验证逻辑
    if not api_header:
        raise HTTPException(
            status_code=401,
            detail=f"缺少必要的 {API_CONFIG['header_name']} 请求头",
            headers={"WWW-Authenticate": API_CONFIG["prefix"]}
        )

    parts = api_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != API_CONFIG["prefix"].lower():
        raise HTTPException(
            status_code=401,
            detail=f"无效的认证格式，正确格式：{API_CONFIG['prefix']} <你的API Key>",
            headers={"WWW-Authenticate": API_CONFIG["prefix"]}
        )

    provided_key = parts[1]
    if provided_key != API_CONFIG["api_key"]:
        raise HTTPException(
            status_code=401,
            detail="无效的API Key，访问被拒绝",
            headers={"WWW-Authenticate": API_CONFIG["prefix"]}
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI生命周期管理"""
    # 启动时预加载模型
    await load_model_global()
    yield
    # 关闭时清理资源
    EXECUTOR.shutdown(wait=True)
    logger.info("✅ 服务已关闭，资源清理完成")


app = FastAPI(
    title="Qwen3-VL TPU推理服务",
    version="2.3.0",
    description="基于算能SE7盒子的Qwen3-VL视觉语言模型推理服务（支持多并发+多图/文件夹+API Key认证）",
    lifespan=lifespan
)


# ========== 数据模型定义 ==========
class ChatMessage(BaseModel):
    role: str
    content: str | List[Dict[str, Any]]  # 支持字符串或多模态内容


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-vl-instruct"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


# ========== 工具函数（新增多图/文件夹处理） ==========
def save_base64_image(base64_str: str) -> str:
    """保存base64图片到临时文件"""
    import base64
    import io
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',', 1)[1]
        image_data = base64.b64decode(base64_str)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
            f.write(image_data)
            return f.name
    except Exception as e:
        logger.error(f"保存base64图片失败: {e}")
        raise HTTPException(status_code=400, detail=f"无效的base64图片数据: {str(e)}")


def download_media_from_url(url: str) -> tuple[str, str]:
    """从URL下载媒体文件（图片/视频）到临时文件，返回(文件路径, 媒体类型)"""
    try:
        logger.info(f"正在从URL下载媒体: {url}")
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()

        # 识别媒体类型
        content_type = response.headers.get('Content-Type', '')
        if content_type.startswith('image/'):
            media_type = "image"
            suffix = '.jpg' if 'jpeg' in content_type or 'jpg' in content_type else '.png'
        elif content_type.startswith('video/'):
            media_type = "video"
            suffix = '.mp4' if 'mp4' in content_type else '.avi'
        else:
            raise HTTPException(status_code=400, detail=f"不支持的媒体类型: {content_type}")

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(response.content)
            logger.info(f"媒体已下载并保存到: {f.name}")
            return f.name, media_type
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="下载媒体超时，请检查URL是否可访问")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"无法下载媒体: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载媒体时发生错误: {str(e)}")


def load_local_media(file_path: str) -> list[tuple[str, str]]:
    """
    加载本地媒体文件/文件夹，返回[(文件路径, 媒体类型), ...]
    支持：单个文件、文件夹（自动遍历）
    """
    media_files = []
    try:
        # 处理file://协议
        if file_path.startswith("file://"):
            file_path = file_path[7:]

        # 转换为绝对路径
        file_path = os.path.abspath(file_path)

        # 检查路径是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"路径不存在: {file_path}")

        # 检查读取权限
        if not os.access(file_path, os.R_OK):
            raise HTTPException(status_code=403, detail=f"无读取权限: {file_path}")

        # 1. 如果是文件夹，遍历所有媒体文件
        if os.path.isdir(file_path):
            logger.info(f"遍历文件夹: {file_path}")
            # 支持的媒体扩展名
            supported_exts = {
                # 图片
                '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp',
                # 视频
                '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'
            }

            # 遍历文件夹（仅一级，不递归）
            for filename in os.listdir(file_path):
                file_full_path = os.path.join(file_path, filename)
                if os.path.isfile(file_full_path):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_exts:
                        # 识别媒体类型
                        content_type, _ = mimetypes.guess_type(file_full_path)
                        if not content_type:
                            ext_map = {
                                '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                                '.bmp': 'image/bmp', '.gif': 'image/gif', '.webp': 'image/webp',
                                '.mp4': 'video/mp4', '.avi': 'video/x-msvideo', '.mov': 'video/quicktime',
                                '.mkv': 'video/x-matroska', '.flv': 'video/x-flv', '.wmv': 'video/x-ms-wmv'
                            }
                            content_type = ext_map.get(ext, '')

                        if content_type.startswith('image/'):
                            media_files.append((file_full_path, "image"))
                        elif content_type.startswith('video/'):
                            media_files.append((file_full_path, "video"))

            if not media_files:
                raise HTTPException(status_code=400, detail=f"文件夹 {file_path} 中未找到支持的媒体文件")

        # 2. 如果是单个文件
        else:
            # 识别媒体类型
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                ext = os.path.splitext(file_path)[1].lower()
                ext_map = {
                    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                    '.bmp': 'image/bmp', '.gif': 'image/gif', '.webp': 'image/webp',
                    '.mp4': 'video/mp4', '.avi': 'video/x-msvideo', '.mov': 'video/quicktime',
                    '.mkv': 'video/x-matroska', '.flv': 'video/x-flv', '.wmv': 'video/x-ms-wmv'
                }
                if ext not in ext_map:
                    raise HTTPException(status_code=400, detail=f"不支持的文件扩展名: {ext} (文件: {file_path})")
                content_type = ext_map[ext]

            if content_type.startswith('image/'):
                media_files.append((file_path, "image"))
            elif content_type.startswith('video/'):
                media_files.append((file_path, "video"))
            else:
                raise HTTPException(status_code=400, detail=f"不支持的媒体类型: {content_type} (文件: {file_path})")

        logger.info(f"成功加载 {len(media_files)} 个本地媒体文件: {[f[0] for f in media_files]}")
        return media_files

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载本地媒体失败: {str(e)} (路径: {file_path})")


def extract_content_and_media(messages: List[ChatMessage]) -> tuple[str, list[tuple[str, str]], str]:
    """
    从OpenAI格式的消息中提取文本、多媒体路径列表、主媒体类型
    支持：1.本地路径/文件夹 2.Base64 3.远程URL 4.多图/多视频
    返回: (text_content, media_files, main_media_type)
          media_files格式: [(文件路径, 媒体类型), ...]
    """
    system_prompt = ""
    text_parts = []
    media_files = []  # 改为列表存储所有媒体文件
    main_media_type = "text"

    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_prompt = msg.content
            continue

        if msg.role == "user":
            if isinstance(msg.content, str):
                text_parts.append(msg.content)
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type", "")

                    if item_type == "text":
                        text_parts.append(item.get("text", ""))
                    elif item_type == "image_url":  # 复用该字段支持所有媒体类型
                        image_url_data = item.get("image_url", {})
                        url = image_url_data.get("url", "") if isinstance(image_url_data, dict) else image_url_data

                        if not url:
                            continue

                        # 1. 本地文件/文件夹（最高优先级）
                        if url.startswith(("file://", "/", "./", "../")):
                            local_media = load_local_media(url)
                            media_files.extend(local_media)
                        # 2. Base64图片
                        elif url.startswith("data:image"):
                            img_path = save_base64_image(url)
                            media_files.append((img_path, "image"))
                        # 3. 远程URL（图片/视频）
                        elif url.startswith(("http://", "https://")):
                            remote_path, remote_type = download_media_from_url(url)
                            media_files.append((remote_path, remote_type))

    # 确定主媒体类型
    if media_files:
        # 优先判断是否有视频，否则为图片
        has_video = any(media_type == "video" for _, media_type in media_files)
        main_media_type = "video" if has_video else "image"

    # 组合文本内容
    user_content = " ".join(text_parts).strip()
    if not user_content and media_files:
        # 无文本时默认生成描述指令（适配多图）
        if len(media_files) > 1:
            user_content = f"请详细描述这{len(media_files)}个媒体文件的内容，并对比分析它们的异同点。"
        else:
            user_content = "请详细描述这个媒体文件的内容。"

    if system_prompt:
        logger.warning(f"System prompt暂时禁用: {system_prompt}")

    return user_content, media_files, main_media_type


# ========== 核心推理函数（同步，支持多图） ==========
def process_inference_sync(prompt: str, media_files: list[tuple[str, str]], main_media_type: str, stream: bool = False):
    """
    同步推理函数（支持多媒体文件）
    media_files: [(文件路径, 媒体类型), ...]
    返回: 非流式返回文本，流式返回生成器
    """
    try:
        # ====================== 🔥 使用全局单例模型 ======================
        model = get_global_model()
        # 请求隔离：每次推理前强制清空历史状态
        model.model.clear_history()
        model.history_max_posid = 0
        model.input_str = prompt
        # =================================================================

        # 构建多媒体消息
        messages = []
        if main_media_type == "text" or not media_files:
            messages = model.text_message()
        else:
            # 处理多个媒体文件（按顺序添加）
            for idx, (media_path, media_type) in enumerate(media_files):
                logger.info(f"处理第{idx + 1}/{len(media_files)}个媒体文件: {media_path} (类型: {media_type})")

                if media_type == "image":
                    media_msg = model.image_message(media_path)
                elif media_type == "video":
                    media_msg = model.video_message(media_path)
                else:
                    raise ValueError(f"不支持的媒体类型: {media_type}")

                # 合并多媒体消息
                if idx == 0:
                    messages = media_msg
                else:
                    if isinstance(messages, list) and isinstance(media_msg, list):
                        messages.extend(media_msg)
                    else:
                        messages += media_msg

        # 处理输入
        inputs = model.process(messages, main_media_type)
        token_len = inputs.input_ids.numel()
        if token_len > model.model.MAX_INPUT_LENGTH:
            raise ValueError(f"输入长度超限: {token_len} > {model.model.MAX_INPUT_LENGTH}")

        # 嵌入层
        model.model.forward_embed(inputs.input_ids)

        # 视觉处理（适配多图）
        position_ids = None
        if main_media_type == "image":
            # 多图时使用最后一个图片的grid信息（适配Qwen3-VL多图逻辑）
            model.vit_process_image(inputs)
            position_ids = model.get_rope_index(inputs.input_ids, inputs.image_grid_thw, model.ID_IMAGE_PAD)
            model.max_posid = int(position_ids.max())
        elif main_media_type == "video":
            model.vit_process_video(inputs)
            position_ids = model.get_rope_index(inputs.input_ids, inputs.video_grid_thw, model.ID_VIDEO_PAD)
            model.max_posid = int(position_ids.max())
        else:
            position_ids = np.array([list(range(token_len))] * 3, dtype=np.int32)
            model.max_posid = token_len - 1

        # 预填充
        prefill_token = model.forward_prefill(position_ids)

        if stream:
            # 流式生成（返回生成器）
            def generate_stream():
                chunk_id = f"chatcmpl-{int(time.time())}"
                full_word_tokens = []
                token = prefill_token

                try:
                    # 第一个token
                    if token is not None and token not in [model.ID_IM_END,
                                                           model.ID_END] and token != model.tokenizer.eos_token_id:
                        full_word_tokens.append(token)
                        word = model.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
                        if "�" not in word:
                            chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "qwen3-vl-instruct",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": word},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            full_word_tokens = []

                    # 后续token
                    for step in range(2047):  # 限制最大长度
                        if model.model.history_length >= model.model.SEQLEN:
                            break
                        model.max_posid += 1
                        pos_ids = np.array([model.max_posid] * 3, dtype=np.int32)
                        token = model.model.forward_next(pos_ids)

                        if token in [model.ID_IM_END, model.ID_END]:
                            # 结束标记
                            chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "qwen3-vl-instruct",
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        if token is None:
                            continue

                        full_word_tokens.append(token)
                        word = model.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
                        if "�" not in word:
                            chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "qwen3-vl-instruct",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": word},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            full_word_tokens = []

                except Exception as e:
                    # 流式异常处理：返回错误信息
                    logger.error(f"流式生成错误: {e}")
                    error_chunk = {
                        "error": {
                            "message": f"流式生成失败: {str(e)}",
                            "type": "stream_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    # 清理所有临时文件
                    for media_path, _ in media_files:
                        if media_path and (media_path.startswith(tempfile.gettempdir()) or "tmp" in media_path):
                            try:
                                os.unlink(media_path)
                            except:
                                pass
                return

            return generate_stream()
        else:
            # 非流式生成
            full_word_tokens = []
            response_text = ""
            token = prefill_token

            # 第一个token
            if token is not None and token not in [model.ID_IM_END,
                                                   model.ID_END] and token != model.tokenizer.eos_token_id:
                full_word_tokens.append(token)
                word = model.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
                if "�" not in word:
                    response_text += word
                    full_word_tokens = []

            # 后续token
            for step in range(2047):
                if model.model.history_length >= model.model.SEQLEN:
                    break
                model.max_posid += 1
                pos_ids = np.array([model.max_posid] * 3, dtype=np.int32)
                token = model.model.forward_next(pos_ids)

                if token in [model.ID_IM_END, model.ID_END]:
                    break

                if token is None:
                    continue

                full_word_tokens.append(token)
                word = model.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
                if "�" not in word:
                    response_text += word
                    full_word_tokens = []

            # 清理所有临时文件
            for media_path, _ in media_files:
                if media_path and (media_path.startswith(tempfile.gettempdir()) or "tmp" in media_path):
                    try:
                        os.unlink(media_path)
                    except:
                        pass

            return response_text.strip() or "抱歉，模型没有生成有效回复。"
    except Exception as e:
        # 清理所有临时文件
        for media_path, _ in media_files:
            if media_path and (media_path.startswith(tempfile.gettempdir()) or "tmp" in media_path):
                try:
                    os.unlink(media_path)
                except:
                    pass
        logger.error(f"推理失败: {e}")
        raise


# ========== API接口 ==========
@app.get("/")
async def root():
    """服务主页"""
    api_info = {
        "api_key_enabled": API_CONFIG["enabled"],
        "api_key_header": API_CONFIG["header_name"],
        "api_key_format": f"{API_CONFIG['prefix']} <your-api-key>" if API_CONFIG['enabled'] else "未启用"
    }

    return {
        "message": "Qwen3-VL TPU推理服务运行中（支持多并发+多图/文件夹+API Key认证）",
        "model": "qwen3-vl-instruct",
        "device": "BM1684X TPU",
        "max_concurrent": MAX_CONCURRENT_REQUESTS,
        "timestamp": int(time.time()),
        "version": "2.3.0",
        "api_config": api_info,
        "model_config": {
            "model_dir": args.model_dir,
            "model_path": MODEL_CONFIG["model_path"],
            "devid": args.devid,
            "video_ratio": args.video_ratio
        },
        "supported_media": {
            "local_file": "支持file:///绝对路径、/绝对路径、./相对路径、../上级路径",
            "local_folder": "支持文件夹路径（自动遍历一级目录下的所有媒体文件）",
            "multi_media": "支持多个URL/base64/file://路径",
            "image_format": "jpg/jpeg/png/bmp/gif/webp",
            "video_format": "mp4/avi/mov/mkv/flv/wmv"
        },
        "endpoints": {
            "chat": "/v1/chat/completions",
            "media": "/v1/media/describe",
            "health": "/health",
            "models": "/v1/models",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(EXECUTOR, get_global_model)
        status = "healthy"
        details = "全局单例模型已加载且运行正常"
    except Exception as e:
        status = "unhealthy"
        details = f"模型加载失败: {str(e)}"

    return {
        "status": status,
        "details": details,
        "model": "qwen3-vl-instruct",
        "model_dir": args.model_dir,
        "max_concurrent": MAX_CONCURRENT_REQUESTS,
        "api_key_enabled": API_CONFIG["enabled"],
        "support_multi_media": True,
        "support_folder": True,
        "timestamp": int(time.time()),
        "version": "2.3.0"
    }


@app.post("/v1/chat/completions")
async def chat_completions(
        request: ChatCompletionRequest,
        # 新增：依赖注入验证API Key
        _: None = Depends(require_api_key)
):
    """
    OpenAI兼容的聊天对话接口（支持多并发+多图/文件夹+API Key认证）
    支持：1.本地文件/文件夹 2.多个Base64图片 3.多个远程URL媒体 4.纯文本
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="至少需要一条消息")

    # 提取内容和多媒体文件
    user_message, media_files, main_media_type = extract_content_and_media(request.messages)
    if not user_message and not media_files:
        raise HTTPException(status_code=400, detail="未找到用户消息或媒体文件")

    # 并发控制：获取锁
    async with REQUEST_LOCK:
        try:
            loop = asyncio.get_running_loop()
            if request.stream:
                # 流式响应
                logger.info(
                    f"流式处理多媒体请求（{len(media_files)}个文件，类型：{main_media_type}）: {user_message[:50]}...")
                stream_generator = await loop.run_in_executor(
                    EXECUTOR, process_inference_sync,
                    user_message, media_files, main_media_type, True
                )

                # 自定义异步迭代器包装器
                async def async_stream_wrapper():
                    try:
                        for chunk in stream_generator:
                            yield chunk
                            await asyncio.sleep(0.001)
                    except Exception as e:
                        logger.error(f"流式迭代错误: {e}")
                        error_chunk = f"data: {json.dumps({'error': str(e)})}\n\n"
                        yield error_chunk
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    async_stream_wrapper(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no"
                    }
                )
            else:
                # 非流式响应
                start_time = time.time()
                logger.info(f"处理多媒体请求（{len(media_files)}个文件，类型：{main_media_type}）: {user_message[:50]}...")
                response_text = await loop.run_in_executor(
                    EXECUTOR, process_inference_sync,
                    user_message, media_files, main_media_type, False
                )

                # 构建响应
                response_id = f"chatcmpl-{int(time.time())}"
                created_time = int(time.time())
                choice = ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
                usage = {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(user_message.split()) + len(response_text.split()),
                    "media_files_count": len(media_files)
                }

                return ChatCompletionResponse(
                    id=response_id,
                    created=created_time,
                    model=request.model,
                    choices=[choice],
                    usage=usage
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"聊天推理错误: {e}")
            raise HTTPException(status_code=500, detail=f"处理聊天请求时发生错误: {str(e)}")


@app.post("/v1/media/describe")
async def describe_media(
        file: UploadFile = File(...),
        prompt: str = Form(default="请详细描述这个媒体文件的内容。"),
        stream: bool = Form(default=False),
        # 依赖注入验证API Key
        _: None = Depends(require_api_key)
):
    """媒体描述接口（支持单文件上传，多文件请使用chat接口）"""
    start_time = time.time()
    temp_path = None

    try:
        # 1. 快速校验文件类型
        media_type = None
        if file.content_type:
            if file.content_type.startswith('image/'):
                media_type = "image"
            elif file.content_type.startswith('video/'):
                media_type = "video"

        # 兜底：通过文件扩展名判断
        if not media_type:
            ext = os.path.splitext(file.filename)[1].lower()
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
            video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
            if ext in image_exts:
                media_type = "image"
            elif ext in video_exts:
                media_type = "video"

        if not media_type:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型：{file.content_type or '未知'}，仅支持图片/视频"
            )

        # 2. 保存临时文件
        suffix = os.path.splitext(file.filename)[1] or ('.jpg' if media_type == 'image' else '.mp4')
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        # 构建多媒体文件列表（适配统一的推理接口）
        media_files = [(temp_path, media_type)]

        logger.info(f"开始处理{media_type}描述请求：{file.filename} | prompt: {prompt[:30]}...")

        # 3. 并发控制 + 推理
        async with REQUEST_LOCK:
            loop = asyncio.get_running_loop()

            if stream:
                # 流式响应
                stream_generator = await loop.run_in_executor(
                    EXECUTOR, process_inference_sync,
                    prompt, media_files, media_type, True
                )

                async def async_stream_wrapper():
                    try:
                        for chunk in stream_generator:
                            yield chunk
                            await asyncio.sleep(0.001)
                    except Exception as e:
                        err_msg = f"流式生成失败：{str(e)}"
                        logger.error(err_msg)
                        yield f"data: {json.dumps({'error': err_msg}, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    async_stream_wrapper(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive"
                    }
                )
            else:
                # 非流式响应
                description = await loop.run_in_executor(
                    EXECUTOR, process_inference_sync,
                    prompt, media_files, media_type, False
                )

                # 计算处理耗时
                processing_time = round(time.time() - start_time, 2)

                return {
                    "status": "success",
                    "description": description,
                    "metadata": {
                        "filename": file.filename,
                        "media_type": media_type,
                        "prompt": prompt,
                        "processing_time_seconds": processing_time,
                        "model": "qwen3-vl-instruct",
                        "model_dir": args.model_dir
                    }
                }

    except HTTPException:
        raise
    except Exception as e:
        err_detail = f"处理{media_type or '媒体'}文件失败：{str(e)}"
        logger.error(f"{err_detail} | 文件：{file.filename}")
        raise HTTPException(status_code=500, detail=err_detail)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass


@app.get("/v1/models")
async def list_models(
        _: None = Depends(require_api_key)
):
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-vl-instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "SE7-Box-TPU",
                "permission": [],
                "root": "qwen3-vl-instruct",
                "parent": None,
                "description": f"Qwen3-VL指令微调版本（支持多图/文件夹），在算能BM1684X TPU上运行"
            }
        ]
    }


@app.get("/v1/models/{model_id}")
async def get_model(
        model_id: str,
        _: None = Depends(require_api_key)
):
    """获取指定模型信息"""
    if model_id != "qwen3-vl-instruct":
        raise HTTPException(status_code=404, detail="模型未找到")
    return {
        "id": "qwen3-vl-instruct",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "SE7-Box-TPU",
        "model_config": {
            "model_dir": args.model_dir,
            "model_path": MODEL_CONFIG["model_path"],
            "devid": args.devid,
            "max_concurrent": MAX_CONCURRENT_REQUESTS
        },
        "capabilities": {
            "multi_media": True,
            "folder_support": True,
            "streaming": True,
            "api_key_auth": API_CONFIG["enabled"]
        },
        "api_config": API_CONFIG,
        "description": f"Qwen3-VL指令微调版本（支持多图/文件夹），在算能BM1684X TPU上运行"
    }


# ========== 启动配置 ==========
if __name__ == "__main__":
    print("🚀 启动Qwen3-VL TPU推理服务...")
    print(f"🎯 模型文件: {MODEL_CONFIG['model_path']}")
    print(f"🔧 设备ID: {args.devid}")
    print(f"⚡ 最大并发数: {MAX_CONCURRENT_REQUESTS}")
    print(f"📝 日志级别: {args.log_level}")
    print(f"🎬 视频采样比例: {args.video_ratio}")
    print(f"📖 API文档: http://0.0.0.0:{args.port}/docs")
    print(f"💬 聊天接口: http://0.0.0.0:{args.port}/v1/chat/completions")
    print(f"🖼️  媒体描述: http://0.0.0.0:{args.port}/v1/media/describe")
    print(f"📂 支持文件夹: file:///绝对路径/文件夹、/绝对路径/文件夹、./相对路径/文件夹")
    print(f"📷 支持多图: 多个image_url字段（URL/base64/file://）")

    # 打印API Key配置信息
    if API_CONFIG["enabled"]:
        print(
            f"🔒 API Key认证已启用: 请求头 {API_CONFIG['header_name']} = {API_CONFIG['prefix']} {API_CONFIG['api_key'][:4]}****{API_CONFIG['api_key'][-4:]}")
    else:
        print(f"⚠️ API Key认证未启用，生产环境请使用 --api-key 参数配置访问密钥")

    # 启动uvicorn
    uvicorn.run(
        "main_serving:app",
        host="0.0.0.0",
        port=args.port,
        reload=False,
        log_level=args.log_level.lower(),
        workers=1,
        loop="uvloop",
        http="httptools"
    )