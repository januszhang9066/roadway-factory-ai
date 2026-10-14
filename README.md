# 🚀 Roadway Lark Bot (LangGraph)

> 工厂群聊 AI-Agent 系统 / Factory Group Chat AI-Agent System
>
> 作者 / Author: Janus Zhang (Roadway Vietnam)
>
> 邮箱 / Email: zhang@roadwayvn.com
>
> 网址 / Web: https://roadwayvn.com

---

## 📋 项目说明 / Project Description

这是 Roadway 越南给韩资/中资工厂做的 **Lark 群聊 AI-Agent 系统**。

集成:
- **Lark 国际版**(飞书国际版,新加坡节点)Webhook
- **LangGraph** 7 节点状态机(强制走完每步)
- **MiniMax-M3** LLM(Anthropic 兼容端点)
- **多模态**:ASR 语音 / VLM 图片 / Video 抽帧 / TTS 语音回复
- **5 个只读工具 + 2 个写工具**(白名单)
- **memory_v2** 全对话记忆(text + 多媒体)
- **history-replay** LLM 自己判断查历史

## 🏗 架构 / Architecture

```
[Lark Webhook] → Dedup → History-Check → multimodal → thinking → choose_tool → execute_tool → verify → respond → send_lark
```

详细架构看 `~/Desktop/越南制造业知识库/工厂AG-技术方案-2026/`

## 📦 文件 / Files

| 文件 | 作用 |
|---|---|
| `webhook.py` | Flask Webhook + Lark 集成 + dedup + send API |
| `demo.py` | LangGraph 7 节点状态机(thinking/choose_tool/...) |
| `multimodal.py` | ASR / VLM / Video / TTS / Lark 文件下载 |
| `test_real.py` | 真测脚本(ASR/VLM/Lark token) |
| `test_replay.py` | 重放测试 |
| `.env` | 凭证(**别 commit** — 已在 .gitignore) |
| `dedup.db` | SQLite(dedup 表 + memory_v2 表) |
| `production_checkpoints.db` | LangGraph 持久化 |

## 🚀 启动 / Run

```bash
# 1. 装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 配凭证
cp .env.example .env
# 编辑 .env,填 LARK_APP_ID / LARK_APP_SECRET / LARK_VERIFICATION_TOKEN / MiniMax API key

# 3. 启动
python3 webhook.py
# → 监听 :8766

# 4. 暴露公网(开发用)
cloudflared tunnel --url http://localhost:8766
# → 拿到 https://xxx.trycloudflare.com
# → 配到 Lark 后台 webhook URL
```

## 🛠 技术栈 / Tech Stack

- **Python 3.9+**
- **Flask** 3.x — Web 框架
- **LangGraph** 0.6.11 — 状态机
- **langchain** — LLM 链
- **requests** — HTTP
- **ffmpeg** — 视频抽帧(`brew install ffmpeg`)
- **MiniMax-M3** — LLM(Anthropic 兼容)

## 📊 节点流程 / Node Flow

1. **load_history** — 从 memory_v2 拉最近 10 条对话
2. **multimodal** — 语音/图片/视频 → 文字
3. **thinking** — LLM 分析 intent / needs_tool / needs_history
4. **choose_tool** — LLM 选 5 个工具之一
5. **execute_tool** — 真调 list/read/search/web
6. **verify** — 验证工具结果(失败重试 < 2 次)
7. **respond** — LLM 整理回复

## 📝 License

Proprietary — © 2026 Roadway Vietnam
