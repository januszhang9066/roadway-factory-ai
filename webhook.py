"""
biseo LangGraph 版 webhook
==========================
跑在 :8766,跟现有 :8765 webhook 并行
Lark Bot 切换 Webhook URL 到 :8766 即可测试

跑法:
    cd ~/lark-webhook-langgraph
    source venv/bin/activate
    python3 webhook.py
"""
import os
import re
import time
import json
import sqlite3
import hashlib
import requests
from flask import Flask, request, jsonify
from demo import build_graph, call_llm
from multimodal import process_lark_message
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================
# 加载 .env
# ============================================================
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

ENV = load_env()
LARK_APP_ID = ENV.get("LARK_APP_ID")
LARK_APP_SECRET = ENV.get("LARK_APP_SECRET")
PORT = int(ENV.get("PORT", "8766"))


# ============================================================
# 去重表(防 Lark 重试)
# ============================================================
DEDUP_DB = os.path.join(os.path.dirname(__file__), "dedup.db")

def init_dedup():
    conn = sqlite3.connect(DEDUP_DB, timeout=5)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS dedup (
        event_id TEXT PRIMARY KEY,
        ts REAL NOT NULL
    )''')
    c.execute("DELETE FROM dedup WHERE ts < ?", (time.time() - 3600,))
    conn.commit()
    conn.close()


def init_memory():
    """启动时建 memory_v2 表(6 列,放原始消息,供 history-replay)"""
    try:
        conn = sqlite3.connect(DEDUP_DB, timeout=5)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS memory_v2 (
            chat_id TEXT,
            user_msg TEXT,
            bot_reply TEXT,
            raw_json TEXT,
            msg_type TEXT,
            ts REAL
        )''')
        conn.commit()
        conn.close()
        print("   ✅ memory_v2 表就绪")
    except Exception as e:
        print(f"⚠️ init_memory: {e}")


init_dedup()
init_memory()

def is_duplicate(event_id: str) -> bool:
    if not event_id:
        return False
    try:
        conn = sqlite3.connect(DEDUP_DB, timeout=5)
        c = conn.cursor()
        c.execute("SELECT 1 FROM dedup WHERE event_id = ? AND ts > ?",
                  (event_id, time.time() - 3600))
        if c.fetchone():
            conn.close()
            return True
        c.execute("INSERT OR IGNORE INTO dedup (event_id, ts) VALUES (?, ?)",
                  (event_id, time.time()))
        conn.commit()
        conn.close()
        return False
    except Exception as e:
        print(f"❌ dedup 错误: {e}")
        return False


# ============================================================
# Memory(简化版,够用)
# ============================================================
def save_msg_to_memory(chat_id: str, user_msg: str, bot_reply: str, raw_message: dict = None, msg_type: str = "text"):
    """保存对话到 memory(SQLite),为后续历史找多媒体用

    raw_message: 原始 Lark message dict(含 file_key/image_key/message_id)
    """
    try:
        conn = sqlite3.connect(DEDUP_DB, timeout=5)
        c = conn.cursor()
        # ⚠️ 用新表名 memory_v2,避免老表结构冲突
        c.execute('''CREATE TABLE IF NOT EXISTS memory_v2 (
            chat_id TEXT,
            user_msg TEXT,
            bot_reply TEXT,
            raw_json TEXT,
            msg_type TEXT,
            ts REAL
        )''')
        raw_json = json.dumps(raw_message) if raw_message else ""
        c.execute("INSERT INTO memory_v2 VALUES (?, ?, ?, ?, ?, ?)",
                  (chat_id, user_msg[:2000], bot_reply[:2000], raw_json, msg_type, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ save_msg_to_memory: {e}")


def get_last_media_from_memory(chat_id: str) -> str:
    """从 memory 找最近一条多媒体的"原始消息"(给 LLM 看,而不是文字描述)

    返回:[raw_json, msg_type] 或空字符串
    """
    try:
        conn = sqlite3.connect(DEDUP_DB, timeout=5)
        c = conn.cursor()
        c.execute(
            "SELECT raw_json, msg_type, ts FROM memory_v2 WHERE chat_id=? AND raw_json != '' AND msg_type IN ('image', 'audio', 'video', 'media', 'file') ORDER BY ts DESC LIMIT 1",
            (chat_id,)
        )
        row = c.fetchone()
        conn.close()
        if row:
            print(f"   📜 [history] 找到历史 {row[1]} (时间:{time.ctime(row[2])})")
            return json.dumps({"raw_json": row[0], "msg_type": row[1]})
        return ""
    except Exception as e:
        print(f"⚠️ get_last_media_from_memory: {e}")
        return ""


# ============================================================
# Lark API
# ============================================================
def get_tenant_token() -> str:
    r = requests.post(
        "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": LARK_APP_ID, "app_secret": LARK_APP_SECRET},
        timeout=10
    )
    data = r.json()
    if data.get("code") != 0:
        raise Exception(f"get_tenant_token 失败: {data}")
    return data["tenant_access_token"]


def send_lark_message(chat_id: str, text: str, message_id: str = None) -> bool:
    """发消息到 Lark(只用 send API,不 reply,避免消息错位)"""
    try:
        token = get_tenant_token()
        # 直接 send(不 reply,避免错位)
        r = requests.post(
            "https://open.larksuite.com/open-apis/im/v1/messages",
            params={"receive_id_type": "chat_id"},
            headers={"Authorization": f"Bearer {token}"},
            json={"receive_id": chat_id, "msg_type": "text", "content": json.dumps({"text": text})},
            timeout=10
        )
        data = r.json()
        if data.get("code") == 0:
            return True
        # 230001 = receive_id 无效(测试假数据/群已解散)
        if data.get("code") == 230001:
            print(f"⚠️ send_lark_message: chat_id 无效({chat_id[:25]}...),跳过(测试数据?)")
            return False
        print(f"❌ send_lark_message 失败: {data}")
        return False
    except Exception as e:
        print(f"❌ send_lark_message 错误: {e}")
        return False


def send_lark_audio(chat_id: str, file_path: str) -> bool:
    """🆕 发语音消息到 Lark(2 步:上传 → 发消息)"""
    try:
        token = get_tenant_token()
        # 1. 上传文件
        with open(file_path, "rb") as f:
            upload_r = requests.post(
                "https://open.larksuite.com/open-apis/im/v1/images",
                files={"image": f},
                data={"image_type": "message"},
                headers={"Authorization": f"Bearer {token}"},
                timeout=30
            )
        # 改用通用 file upload
        with open(file_path, "rb") as f:
            upload_r = requests.post(
                "https://open.larksuite.com/open-apis/im/v1/files",
                files={"file": f},
                data={"file_type": "mp3"},
                headers={"Authorization": f"Bearer {token}"},
                timeout=30
            )
        upload_data = upload_r.json()
        if upload_data.get("code") != 0:
            print(f"❌ 上传失败: {upload_data}")
            return False
        file_key = upload_data.get("data", {}).get("file_key", "")
        if not file_key:
            print(f"❌ 没拿到 file_key: {upload_data}")
            return False
        # 2. 发消息(用 audio 类型)
        msg_r = requests.post(
            "https://open.larksuite.com/open-apis/im/v1/messages",
            params={"receive_id_type": "chat_id"},
            headers={"Authorization": f"Bearer {token}"},
            json={
                "receive_id": chat_id,
                "msg_type": "audio",
                "content": json.dumps({"file_key": file_key})
            },
            timeout=10
        )
        msg_data = msg_r.json()
        if msg_data.get("code") == 0:
            print(f"   ✅ 语音消息发送成功")
            return True
        print(f"❌ 发语音消息失败: {msg_data}")
        return False
    except Exception as e:
        print(f"❌ send_lark_audio 错误: {e}")
        return False


# ============================================================
# LangGraph app
# ============================================================
workflow = build_graph()


# ============================================================
# Flask webhook
# ============================================================
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "version": "3.0-langgraph",
        "port": PORT,
        "engine": "LangGraph 0.6.11",
        "llm": "MiniMax-M3"
    })


@app.route("/lark/webhook/biseo/", methods=["POST"])
def webhook():
    data = request.json
    print(f"\n📨 收到 webhook: {json.dumps(data)[:300]}")

    # 1. URL 验证
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data.get("challenge", "")})

    # 2. 事件类型 + dedup
    header = data.get("header", {})
    event_type = header.get("event_type", "")
    event_id = header.get("event_id", "")

    if event_id and is_duplicate(event_id):
        print(f"⏭️  跳过重复 event: {event_id}")
        return jsonify({"code": 0, "msg": "duplicate skipped"})

    if event_type != "im.message.receive_v1":
        return jsonify({"code": 0, "msg": "ignored"})

    # 3. 解析消息(用 multimodal 处理语音/图片/视频)
    event = data.get("event", {})
    message = event.get("message", {})
    sender = event.get("sender", {})
    chat_id = message.get("chat_id")
    message_id = message.get("message_id")
    chat_type = message.get("chat_type", "p2p")
    msg_type = message.get("message_type", "text")

    # 🔍 打印完整 message 字段(诊断用)
    print(f"   🔍 [raw] message_type={msg_type}")
    print(f"   🔍 [raw] message_id={message_id}")
    print(f"   🔍 [raw] content={message.get('content', '')[:500]}")

    # 用 multimodal 统一处理(text/audio/image/file/video)
    mm_result = process_lark_message(message)
    user_text = mm_result["text"]
    # 去掉 @ Bot 占位符
    user_text = re.sub(r"@_user_\d+\s*", "", user_text)

    # 🔍 先调一次 LLM,只判断"老板是不是在问之前发过的媒体"(轻量 prompt,几十 token)
    if mm_result.get("msg_type") == "text":
        try:
            from demo import call_llm
            check_prompt = f"""判断老板这条消息是不是在"引用或询问之前发过的内容(图/视频/语音/文件/对话/资料/干货)"。

重点看:
- "我的意思是这个" → true(指代前文)
- "上面" → true
- "再看看那个" → true
- "之前发的" → true
- "上次的" → true
- "明天 7:00 给我一份" → false(未来任务,不是引用)
- "你好" → false
- "分析这个视频"(指当前发的) → false
- "帮我看一下"(无指代) → false
- "翻译成中文"(没指代) → false

老板消息:"{user_text}"

只回 true 或 false(小写)。"""
            check_result = call_llm(check_prompt).strip().lower()
            needs_history = "true" in check_result
            print(f"   🔍 [history-check] LLM 判断: needs_history={needs_history}")
        except Exception as e:
            print(f"   ⚠️ history-check 失败: {e},用 fallback")
            # fallback:任何"之前/上面/那个"等词触发
            history_hints = ["上面", "之前", "刚才", "上次", "那个", "这个", "前面", "再找", "再查", "동영상", "영상", "다시", "한번", "trên", "lại"]
            needs_history = any(kw in user_text for kw in history_hints)
    else:
        needs_history = False

    # 🆕 历史找视频/图片/文件 — LLM 判断后查 memory_v2
    if needs_history:
        last_media_json = get_last_media_from_memory(chat_id)
        if last_media_json:
            try:
                last = json.loads(last_media_json)
                raw_msg = json.loads(last["raw_json"])
                # 🔥 真去重处理原始消息(调 multimodal 看到原始视频/图片)
                print(f"   🔥 [history-replay] 重处理历史 {last['msg_type']}")
                hist_result = process_lark_message(raw_msg)
                if hist_result.get("text") and not hist_result["text"].startswith("["):
                    user_text = f"{user_text}\n\n[老板之前发的 {last['msg_type']},重新分析如下]\n{hist_result['text']}"
                else:
                    user_text = f"{user_text}\n\n[老板之前发的 {last['msg_type']} 处理失败:{hist_result.get('text', '未知')}]"
            except Exception as e:
                print(f"   ❌ [history-replay] 错误: {e}")
                user_text = f"{user_text}\n\n[历史重处理失败: {e}]"
        else:
            user_text = f"{user_text}\n\n[没找到历史媒体,可能老板没发过或已过期]"

    if not user_text:
        user_text = "[空消息]"

    print(f"💬 收到: chat_id={chat_id} type={msg_type} 文本={user_text[:100]}")

    # 4. 调 LangGraph
    config = {"configurable": {"thread_id": chat_id}}
    state = {
        "chat_id": chat_id,        # 🆕 load_history_node 需要
        "user_msg": user_text,
        "user_lang": "auto",
        "msg_type": msg_type,
        "raw_message": message,
        "intent": "",
        "needs_tool": False,
        "chosen_tool": "",
        "tool_params": {},
        "tool_result": "",
        "tool_error": "",
        "tool_attempts": 0,
        "verified": False,
        "final_answer": "",
        "history": []
    }
    start = time.time()
    with SqliteSaver.from_conn_string("production_checkpoints.db") as memory:
        graph_app = workflow.compile(checkpointer=memory)
        result = graph_app.invoke(state, config=config)
    elapsed = time.time() - start
    print(f"⏱️  LangGraph 耗时: {elapsed:.1f}s")
    print(f"📊 节点:thinking → choose_tool → {'execute_tool → verify → ' if result.get('chosen_tool') else ''}respond")
    print(f"💡 Bot 回复: {result['final_answer'][:200]}")

    # 5. 发回 Lark(文字)
    ok = send_lark_message(chat_id, result["final_answer"], message_id=message_id)
    print(f"{'✅' if ok else '❌'} 回复 Lark: {'成功' if ok else '失败'}")

    # 5.5 🆕 自动 TTS 语音回复(老板用语音发 → 语音回)
    if mm_result.get("msg_type") == "audio" and ok:
        try:
            from multimodal import minimax_tts
            tts_text = result["final_answer"][:500]
            tts_path = minimax_tts(tts_text, output_path="/tmp/bot_tts_reply.mp3")
            if tts_path and os.path.exists(tts_path):
                # 上传 mp3 到 Lark 发语音消息
                send_lark_audio(chat_id, tts_path)
        except Exception as e:
            print(f"   ⚠️ TTS 自动回复失败: {e}")

    # 6. 存到 memory(供后续历史找多媒体 + 上下文)
    # 🆕 老板原话:全部对话都要存(text + 多媒体),"我的意思是这个"才能看到上面聊啥
    save_msg_to_memory(chat_id, user_text, result["final_answer"], raw_message=message, msg_type=msg_type)
    print(f"   💾 存到 memory_v2: type={msg_type}, 共 {len(user_text)} 字")

    return jsonify({"code": 0, "msg": "ok"})


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 biseo LangGraph Webhook 启动")
    print(f"   端口: {PORT}")
    print(f"   Lark App: {LARK_APP_ID}")
    print(f"   LLM: MiniMax-M3 (Anthropic 端点)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
