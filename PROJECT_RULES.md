# PROJECT_RULES.md — Roadway Lark Bot (LangGraph)

> 本文件是本项目的工作规则。改动本项目前必读。改完请同步更新本文件对应段落。
> (注:Claude Code 默认只读 `AGENTS.md`。如需自动加载,可把本文件内容复制进 AGENTS.md。)

## 项目是什么

Roadway 越南给韩资 / 中资工厂做的 **Lark(飞书国际版,新加坡节点)群聊 AI-Agent 系统**。
客户在工厂群里发消息 / 语音 / 图片 / 视频,Bot 理解后调用工具查询并回复。

**作者**: Janus Zhang (Roadway Vietnam) · zhang@roadwayvn.com · https://roadwayvn.com

## 硬约束(违反就是错)

1. **7 节点状态机不能改顺序**。LangGraph 的 `thinking → choose_tool → execute_tool → verify`
   是**强制走完**的,不是可跳过的。客户已按这个行为培训过工厂管理员。
2. **工具白名单固定**:5 个只读 + 2 个写工具。**新增工具必须先进白名单**,
   不允许模型动态发现新工具。
3. **`.env` 绝不入 git**。已在 `.gitignore`,改动时不要移除该行。
4. **`dedup.db` / `*_checkpoints.db` 是运行时数据**,不要手工改、不要提交到 git、
   不要在测试里清空生产库。测试用 `demo_checkpoints.db`。
5. **端口固定**:本项目跑 `:8766`。`:8765` 是旧的非 LangGraph 版 webhook,并行跑,
   **不要把两个服务的端口搞混**。
6. **凭证只从 `.env` 读**,不要硬编码到代码里,不要打进日志。

## 架构

```
[Lark Webhook] → Dedup → History-Check → multimodal → thinking → choose_tool
                                                                    ↓
        send_lark ← respond ← verify ← execute_tool ←──────────────┘
```

## 关键文件

| 文件 | 作用 |
|---|---|
| `webhook.py` | Flask Webhook + Lark 集成 + dedup + send API |
| `demo.py` | LangGraph 7 节点状态机 + `build_graph()` / `call_llm()` |
| `multimodal.py` | ASR / VLM / Video 抽帧 / TTS / Lark 文件下载 |
| `test_real.py` | 真测脚本(ASR / VLM / Lark token) |
| `test_replay.py` | 重放测试 |
| `.env` | 凭证(**别 commit**) |
| `dedup.db` | SQLite(dedup 表 + memory_v2 表) |
| `production_checkpoints.db` | LangGraph 持久化 |

## 技术栈

Python 3.9+ · Flask 3.x · LangGraph 0.6.11 · langchain · requests · ffmpeg(视频抽帧)

## 本地运行

```bash
cd ~/lark-webhook-langgraph
source venv/bin/activate
python3 webhook.py          # → :8766
```

暴露公网(开发):
```bash
cloudflared tunnel --url http://localhost:8766
```

## LLM provider(2026-10 现状)

当前用 **MiniMax-M3**(Anthropic 兼容端点)。**该订阅额度已撞限(429)**,正在换。

**候选方案(已调研,见 `~/Desktop/越南制造业知识库/工厂SaaS方案书-2026/`)**:

| 用途 | 方案 | 说明 |
|---|---|---|
| 主 LLM | `qwen-oauth` | 阿里云 OAuth 登录,免费额度,base_url `token-plan.maas.qwencloudapi.com/apps/anthropic` |
| 备 LLM | `deepseek` | 缓存 $0.015/M 最低,官方文档点名支持 Hermes |
| 写代码 | Claude Code(订阅) | 老板已有 v2.1.304 |
| 编程 CLI | Codex CLI + ChatGPT Plus $20 | 限额按消息数算,不吃 token |

**改 provider 时注意**:
- Qwen Token Plan **只能用 `sk-sp-` 开头的 key** + 专用 base_url,和按量 API(`sk-`)完全分开
- 混用会静默走按量计费,账单爆
- 越南买家买 Token Plan **需要税号**

## 上下文成本(重要)

本月实测:**586.8M tokens,其中 93.8% 是缓存读**,单次调用带 10.8 万 tokens 上下文。
`~/.hermes/state.db` 的 `session_model_usage` 表可查。

**改这个项目时**:不要往 prompt 里塞大段常量,不要把所有历史塞进去。
用 `memory_v2` 表做检索,而不是全量回放。

## 测试

```bash
source venv/bin/activate
python3 test_replay.py     # 重放测试(不碰生产库)
python3 test_real.py       # 真测,需要真凭证 + 网络
```

## 沟通规范

- 老板偏好:大白话短句,不堆术语;诊断给「一句话结论 + 为什么 + 怎么办」
- 客户是韩资 / 中资工厂,方案和文档一律出**中韩越三语**
- 交付物同步到 Obsidian vault:`~/Desktop/越南制造业知识库/`
