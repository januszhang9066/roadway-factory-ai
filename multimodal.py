"""
多模态模块 — Lark 语音/图片下载 + MiniMax ASR/VLM
复用现有 ~/lark-webhook/webhook_server.py 里的逻辑
"""
import os
import time
import requests
import json


# Lark 配置(从 .env 读；.env 不存在时回退到系统环境变量，不崩溃)
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    env = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    for k, v in os.environ.items():
        env.setdefault(k, v)
    return env

ENV = load_env()
LARK_APP_ID = ENV.get("LARK_APP_ID")
LARK_APP_SECRET = ENV.get("LARK_APP_SECRET")


# Hermes 配置(MiniMax key)
HERMES_ENV = os.path.expanduser("~/.hermes/.env")
HERMES_CONFIG = os.path.expanduser("~/.hermes/config.yaml")

def get_minimax_config():
    api_key = None
    with open(HERMES_ENV) as f:
        for line in f:
            if line.startswith("MINIMAX_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
    # base_url 用根 URL,不要 anthropic 子路径
    base_url = "https://api.minimax.io"
    return api_key, base_url


# Lark token 缓存
_tenant_token = None
_token_expires = 0

def get_tenant_token() -> str:
    """拿 tenant_access_token(缓存 100 分钟)"""
    global _tenant_token, _token_expires
    if _tenant_token and time.time() < _token_expires:
        return _tenant_token
    r = requests.post(
        "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": LARK_APP_ID, "app_secret": LARK_APP_SECRET},
        timeout=10
    )
    data = r.json()
    if data.get("code") != 0:
        raise Exception(f"get_tenant_token 失败: {data}")
    _tenant_token = data["tenant_access_token"]
    _token_expires = time.time() + 6000  # 100 分钟
    return _tenant_token


def download_lark_file(file_key: str, file_type: str = "file", message_id: str = None) -> str:
    """下载 Lark 文件到本地 /tmp/

    Lark API 端点:/open-apis/im/v1/messages/{message_id}/resources/{file_key}
    必传路径参数:message_id, file_key
    """
    if not message_id:
        print("   ❌ download_lark_file 缺 message_id")
        return None
    try:
        token = get_tenant_token()
        # 正确 URL:path 里带 message_id 和 file_key
        url = f"https://open.larksuite.com/open-apis/im/v1/messages/{message_id}/resources/{file_key}"
        r = requests.get(
            url,
            params={"type": file_type},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        if r.status_code != 200:
            print(f"   ❌ Lark 下载失败 status={r.status_code} body={r.text[:300]}")
            return None
        ext_map = {"file": ".bin", "audio": ".ogg", "image": ".jpg"}
        ext = ext_map.get(file_type, ".bin")
        local_path = f"/tmp/lark_{file_key}{ext}"
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"   ✅ 下载成功 {local_path} ({len(r.content)} bytes)")
        return local_path
    except Exception as e:
        print(f"❌ download_lark_file 错误: {e}")
        return None


def minimax_asr(audio_path: str) -> str:
    """MiniMax ASR 语音转文字(自动检测语言)"""
    try:
        api_key, base_url = get_minimax_config()
        with open(audio_path, "rb") as f:
            r = requests.post(
                f"{base_url}/v1/speech_to_text",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": f},
                data={"model": "asr-1.0"},
                timeout=30
            )
        data = r.json()
        if "text" in data:
            return data["text"]
        # 备选字段
        if "transcription" in data:
            return data["transcription"]
        print(f"❌ ASR 返回: {data}")
        return ""
    except Exception as e:
        print(f"❌ ASR 错误: {e}")
        return ""


def minimax_video(video_path: str, prompt: str = "描述这个视频的内容", frames: int = 5) -> str:
    """视频理解:抽帧后用 VLM 看每帧(返回综合描述)

    需要 ffmpeg
    """
    try:
        import subprocess
        import base64
        # 抽 N 帧(均匀分布)
        duration_cmd = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=5
        )
        try:
            duration = float(duration_cmd.stdout.strip())
        except:
            duration = 10.0  # 默认 10 秒

        # 抽帧时间点
        times = [duration * (i + 0.5) / frames for i in range(frames)]
        frames_with_time = []  # (时间点秒, 帧路径)
        for i, t in enumerate(times):
            frame_path = f"/tmp/frame_{i}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(t), "-i", video_path, "-vframes", "1", frame_path],
                capture_output=True, timeout=5
            )
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                frames_with_time.append((t, frame_path))

        if not frames_with_time:
            return "❌ 视频抽帧失败"

        # 用 VLM 看每帧（注意：时间戳必须用当前帧自己的 t，不能复用循环外变量）
        frame_descriptions = []
        for t, fp in frames_with_time:
            desc = minimax_vlm(fp, f"这是视频第 {t:.1f} 秒的画面,描述内容。")
            if desc:
                frame_descriptions.append(desc)
            os.remove(fp)  # 清理

        if not frame_descriptions:
            return "❌ 视频帧分析失败"

        # 综合多帧描述
        combined = "\n".join([f"帧{i+1}: {d}" for i, d in enumerate(frame_descriptions)])
        summary = call_llm_simple(
            f"以下是视频多个关键帧的描述:\n{combined}\n\n请综合这些帧,用 100-200 字总结整个视频讲了什么(场景/人物/动作/事件):"
        )
        return summary or combined[:500]
    except Exception as e:
        print(f"❌ Video 错误: {e}")
        return f"❌ 视频处理失败: {e}"


def minimax_tts(text: str, voice: str = "male-qn-qingse", output_path: str = "/tmp/bot_reply.mp3") -> str:
    """TTS 文字转语音(MiniMax /v1/t2a_v2,生成 .mp3 文件路径)"""
    try:
        api_key, _ = get_minimax_config()
        r = requests.post(
            "https://api.minimax.io/v1/t2a_v2",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "speech-01-hd",
                "text": text[:3000],
                "stream": False,
                "output_format": "hex",
                "voice_setting": {
                    "voice_id": voice if voice else "male-qn-qingse",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0
                },
                "audio_setting": {
                    "sample_rate": 32000,
                    "bitrate": 128000,
                    "format": "mp3",
                    "channel": 1
                }
            },
            timeout=30
        )
        data = r.json()
        if r.status_code != 200 or "data" not in data or not data["data"]:
            print(f"   ❌ TTS 失败 status={r.status_code} resp={str(data)[:200]}")
            return ""
        audio_hex = data["data"].get("audio", "")
        if not audio_hex:
            print(f"   ❌ TTS 无音频数据")
            return ""
        audio_bytes = bytes.fromhex(audio_hex)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"   ✅ TTS 成功 {output_path} ({len(audio_bytes)} bytes)")
        return output_path
    except Exception as e:
        print(f"   ❌ TTS 错误: {e}")
        return ""


def call_llm_simple(prompt: str) -> str:
    """简化 LLM 调用(给 video summary 用)"""
    try:
        api_key, _ = get_minimax_config()
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        body = {
            "model": "MiniMax-M3",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        }
        r = requests.post(
            "https://api.minimax.io/anthropic/v1/messages",
            headers=headers, json=body, timeout=30
        )
        data = r.json()
        if "content" in data and data["content"]:
            return data["content"][0]["text"]
        return ""
    except Exception as e:
        print(f"❌ call_llm_simple: {e}")
        return ""


def minimax_vlm(image_path: str, prompt: str = "描述这张图片的内容") -> str:
    """用 LLM 端点直接看图(图片 base64 嵌入)"""
    try:
        import base64
        api_key, _ = get_minimax_config()
        # 读图 + base64
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        # 推断 mime
        ext = image_path.lower().split(".")[-1]
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
        mime = mime_map.get(ext, "image/jpeg")

        # 用 Anthropic Messages 端点(支持 image content)
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        body = {
            "model": "MiniMax-M3",
            "max_tokens": 500,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": img_data}},
                    {"type": "text", "text": prompt}
                ]
            }]
        }
        r = requests.post(
            "https://api.minimax.io/anthropic/v1/messages",
            headers=headers,
            json=body,
            timeout=30
        )
        data = r.json()
        if "content" in data and data["content"]:
            return data["content"][0]["text"]
        print(f"❌ VLM 返回: {r.text[:200]}")
        return ""
    except Exception as e:
        print(f"❌ VLM 错误: {e}")
        return ""


# ============================================================
# 处理 Lark 消息(主入口)
# ============================================================
def process_lark_message(msg: dict) -> dict:
    """处理一条 Lark 消息,返回:
    {
        "text": "用户消息文字(语音/图片已转)",
        "msg_type": "text/audio/image/file",
        "raw": {...原始...},
    }
    """
    msg_type = msg.get("message_type", "text")
    content_raw = msg.get("content", "{}")
    try:
        content = json.loads(content_raw)
    except:
        content = {}

    # 文字消息
    if msg_type == "text":
        text = content.get("text", "").strip()
        return {"text": text, "msg_type": "text", "raw": content}

    # 图片消息
    if msg_type == "image":
        image_key = content.get("image_key", "")
        if not image_key:
            return {"text": "[图片,无 image_key]", "msg_type": "image", "raw": content}
        print(f"   📷 下载图片 {image_key}")
        local_path = download_lark_file(image_key, "image", message_id=msg.get("message_id"))
        if not local_path:
            return {"text": "[图片下载失败]", "msg_type": "image", "raw": content}
        print(f"   🤖 VLM 理解图片...")
        vlm_result = minimax_vlm(local_path)
        if vlm_result:
            return {
                "text": f"[用户发的图片,内容如下]\n{vlm_result}",
                "msg_type": "image",
                "raw": content
            }
        return {"text": "[图片理解失败]", "msg_type": "image", "raw": content}

    # 语音消息
    if msg_type == "audio":
        file_key = content.get("file_key", "")
        if not file_key:
            return {"text": "[语音,无 file_key]", "msg_type": "audio", "raw": content}
        # ⚠️ Lark 规则:file_v3_ 前缀的语音,type 必须是 "file"(不是 "audio")
        print(f"   🎤 下载语音 {file_key}")
        local_path = download_lark_file(file_key, "file", message_id=msg.get("message_id"))
        if not local_path:
            return {"text": "[语音下载失败]", "msg_type": "audio", "raw": content}
        print(f"   🤖 ASR 转录语音...")
        asr_text = minimax_asr(local_path)
        if asr_text:
            return {
                "text": f"[用户发的语音,转文字如下]\n{asr_text}",
                "msg_type": "audio",
                "raw": content
            }
        return {"text": "[语音转录失败]", "msg_type": "audio", "raw": content}

    # 文件消息
    if msg_type == "file":
        return {
            "text": f"[用户发的文件:{content.get('file_name', '未知')}]",
            "msg_type": "file",
            "raw": content
        }

    # 视频消息(Lark 可能用 media 或 video)
    if msg_type == "video":
        file_key = content.get("file_key", "")
        file_name = content.get("file_name", "")
        if not file_key:
            return {"text": "[视频,无 file_key]", "msg_type": "video", "raw": content}
        print(f"   🎬 下载视频 {file_key}")
        local_path = download_lark_file(file_key, "file", message_id=msg.get("message_id"))
        if not local_path:
            return {"text": "[视频下载失败]", "msg_type": "video", "raw": content}
        print(f"   🤖 视频理解(抽 5 帧 + 综合)...")
        video_result = minimax_video(local_path)
        if video_result:
            return {
                "text": f"[用户发的视频({file_name}),内容如下]\n{video_result}",
                "msg_type": "video",
                "raw": content
            }
        return {"text": "[视频理解失败]", "msg_type": "video", "raw": content}

    # 其它(含 media 类型)
    if msg_type == "media" or msg_type not in ("text", "image", "audio", "file", "video"):
        # 试着看 content 里有什么
        file_key = content.get("file_key", "") or content.get("media_key", "")
        image_key = content.get("image_key", "")
        file_name = content.get("file_name", "")

        # 是视频?(.mp4/.mov/.avi/.mkv/.webm)
        if file_name and any(ext in file_name.lower() for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]):
            file_key = file_key or content.get("file_key", "")
            print(f"   🎬 [media-as-video] 下载视频 {file_key}")
            local_path = download_lark_file(file_key, "file", message_id=msg.get("message_id"))
            if local_path:
                video_result = minimax_video(local_path)
                if video_result:
                    return {
                        "text": f"[用户发的视频({file_name}),内容如下]\n{video_result}",
                        "msg_type": "media",
                        "raw": content
                    }
            return {"text": "[media 视频处理失败]", "msg_type": "media", "raw": content}

        # 是图片?
        if image_key or (file_name and any(ext in file_name.lower() for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"])):
            image_key = image_key or file_key
            print(f"   📷 [media-as-image] 下载图片 {image_key}")
            local_path = download_lark_file(image_key, "image", message_id=msg.get("message_id"))
            if local_path:
                vlm_result = minimax_vlm(local_path)
                if vlm_result:
                    return {
                        "text": f"[用户发的图片,内容如下]\n{vlm_result}",
                        "msg_type": "media",
                        "raw": content
                    }
            return {"text": "[media 图片处理失败]", "msg_type": "media", "raw": content}

        # 是语音?
        if file_key or (file_name and any(ext in file_name.lower() for ext in [".ogg", ".mp3", ".wav", ".m4a", ".opus"])):
            file_key = file_key or content.get("file_key", "")
            # ⚠️ Lark 规则:file_v3_ 前缀用 type=file
            print(f"   🎤 [media-as-audio] 下载语音 {file_key}")
            local_path = download_lark_file(file_key, "file", message_id=msg.get("message_id"))
            if local_path:
                asr_text = minimax_asr(local_path)
                if asr_text:
                    return {
                        "text": f"[用户发的语音,转文字如下]\n{asr_text}",
                        "msg_type": "media",
                        "raw": content
                    }
            return {"text": "[media 语音处理失败]", "msg_type": "media", "raw": content}

        # 其它文件
        return {
            "text": f"[用户发的文件:{file_name or file_key or '未知'}]",
            "msg_type": "media",
            "raw": content
        }

    # 其它
    return {"text": f"[{msg_type} 消息]", "msg_type": msg_type, "raw": content}
