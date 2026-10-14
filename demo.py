"""
LangGraph Demo for biseo Bot
============================
5 节点状态机:thinking → choose_tool → execute_tool → verify → respond
强制每步执行,LLM 不能偷懒

跑法:
    cd ~/lark-webhook-langgraph
    source venv/bin/activate
    python3 demo.py
"""
import os
import re
import time
import json
import requests
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ============================================================
# 1. State 定义(跨节点传递)
# ============================================================

# 🆕 DB 路径常量(供 load_history_node 用)
DEDUP_DB = "dedup.db"

class AgentState(TypedDict):
    chat_id: str            # 🆕 群 ID(load_history 用)
    user_msg: str           # 老板发的话
    user_lang: str          # 语言
    msg_type: str           # 消息类型(text/audio/image)
    raw_message: dict       # 原始 Lark 消息
    intent: str             # 意图
    needs_tool: bool        # 要不要工具
    chosen_tool: str        # 选的工具
    tool_params: dict       # 工具参数
    tool_result: str        # 工具结果
    tool_error: str         # 工具错误
    tool_attempts: int      # 尝试次数
    verified: bool          # 验证通过
    needs_history: bool     # 🆕 thinking_node 判断是否需要拉取群历史
    final_answer: str       # 最终回复
    history: list           # 对话历史


# ============================================================
# 2. LLM 调用(用 Hermes 自己的 MiniMax M3 key)
# ============================================================
def call_llm(prompt: str, system_prompt: str = "") -> str:
    """调 MiniMax M3（API key 读取顺序：环境变量 MINIMAX_API_KEY → 项目 .env → ~/.hermes/.env）"""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        for env_file in (os.path.join(os.path.dirname(__file__), ".env"),
                         os.path.expanduser("~/.hermes/.env")):
            if os.path.exists(env_file):
                with open(env_file) as f:
                    for line in f:
                        if line.startswith("MINIMAX_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break
            if api_key:
                break
    if not api_key:
        raise RuntimeError("未找到 MINIMAX_API_KEY（请设置环境变量或写入项目 .env，参考 .env.example）")

    hermes_config = os.path.expanduser("~/.hermes/config.yaml")

    # LLM 用 anthropic 端点
    llm_base_url = "https://api.minimax.io/anthropic"
    model = "MiniMax-M3"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        body["system"] = system_prompt

    r = requests.post(f"{llm_base_url}/v1/messages", headers=headers, json=body, timeout=30)
    data = r.json()
    return data["content"][0]["text"]


# ============================================================
# 3. 5 个工具(老板白名单)
# ============================================================
KNOWLEDGE_BASE = os.path.expanduser("~/Desktop/越南制造业知识库/")

BISEO_SYSTEM_PROMPT = """你是 biseo(비서),Roadway 老板(장호웅 / 张豪雄)的私人 AI 助理。

## 🧠 身份
- 称呼老板为"老板"或"장总"
- 老板是越南北部韩中资制造业工厂专家
- 老板说 4 种语言:中/韩/越/英

## 🗣️ 回答风格
- 大白话,不打官腔
- 一句话结论 + 为什么 + 怎么办
- 不用术语堆砌
- 老板问什么语言就答什么语言

## 💪 能力
- 文字对话 / 翻译 / 查知识库 / 搜网络
- 工具:list_directory / read_file / search_files / web_search / web_extract / write_file / edit_file
"""

def tool_list_directory(path: str = KNOWLEDGE_BASE) -> str:
    path = os.path.expanduser(path)
    if not path.startswith(KNOWLEDGE_BASE):
        return f"❌ 路径不在白名单(只读 ~/Desktop/越南制造业知识库/)"
    try:
        items = sorted(os.listdir(path))
        return "\n".join(f"📁 {x}" if os.path.isdir(os.path.join(path, x)) else f"📄 {x}" for x in items)
    except Exception as e:
        return f"❌ 错误: {e}"

def tool_read_file(path: str, max_lines: int = 50) -> str:
    path = os.path.expanduser(path)
    if not path.startswith(KNOWLEDGE_BASE):
        return f"❌ 路径不在白名单"
    try:
        with open(path, encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"...(还有更多,共 {max_lines}+ 行)")
                    break
                lines.append(line.rstrip())
            return "\n".join(lines)
    except Exception as e:
        return f"❌ 错误: {e}"

def tool_search_files(pattern: str, path: str = KNOWLEDGE_BASE) -> str:
    path = os.path.expanduser(path)
    if not path.startswith(KNOWLEDGE_BASE):
        return f"❌ 路径不在白名单"
    try:
        results = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if pattern.lower() in f.lower():
                    results.append(os.path.join(root, f))
            if len(results) > 20:
                break
        return "\n".join(results[:20]) if results else f"❌ 没找到匹配 {pattern} 的文件"
    except Exception as e:
        return f"❌ 错误: {e}"

def tool_web_search(query: str) -> str:
    """用 DuckDuckGo Lite(lite 版稳定)"""
    try:
        r = requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Macintosh)"},
            timeout=5
        )
        # DuckDuckGo Lite 简单 HTML 解析
        import re
        text = r.text
        # 提取 result__snippet
        snippets = re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', text, re.DOTALL)
        links = re.findall(r'<a[^>]*class="result-link"[^>]*>(.*?)</a>', text, re.DOTALL)
        # 备选:用 result__url
        if not snippets:
            snippets = re.findall(r'<td[^>]*class="result__snippet"[^>]*>(.*?)</td>', text, re.DOTALL)
            links = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', text, re.DOTALL)
        # 备选:lite 简单分隔
        if not snippets:
            # 抓所有 link + 后面跟的 text
            all_tds = re.findall(r'<td[^>]*>(.*?)</td>', text, re.DOTALL)
            snippets = [t for t in all_tds if len(t) > 30 and 'class=' not in t][:5]
        out = []
        for i, snip in enumerate(snippets[:5]):
            clean = re.sub(r'<[^>]+>', '', snip).strip()
            if clean:
                out.append(f"{i+1}. {clean[:300]}")
        return "\n\n".join(out) if out else f"❌ 没搜到结果(query={query})"
    except Exception as e:
        return f"❌ 错误: {e}"

def tool_web_extract(url: str) -> str:
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        text = re.sub(r'<script[^>]*>.*?</script>', '', r.text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:2000]
    except Exception as e:
        return f"❌ 错误: {e}"


def tool_write_file(path: str, content: str) -> str:
    """🆕 写文件(老板说可以写,白名单内)"""
    path = os.path.expanduser(path)
    if not path.startswith(KNOWLEDGE_BASE):
        return f"❌ 路径不在白名单(只写 ~/Desktop/越南制造业知识库/)"
    try:
        # 自动建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        size = os.path.getsize(path)
        return f"✅ 写入成功:{path} ({size} bytes, {len(content)} 字)"
    except Exception as e:
        return f"❌ 写入失败: {e}"


def tool_edit_file(path: str, old_text: str, new_text: str) -> str:
    """🆕 编辑文件(替换 old_text 为 new_text)"""
    path = os.path.expanduser(path)
    if not path.startswith(KNOWLEDGE_BASE):
        return f"❌ 路径不在白名单(只编辑 ~/Desktop/越南制造业知识库/)"
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            return f"❌ 没找到要替换的文本(old_text 不存在)"
        new_content = content.replace(old_text, new_text, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"✅ 编辑成功:{path} (替换了 1 处)"
    except Exception as e:
        return f"❌ 编辑失败: {e}"

TOOLS_DISPATCH = {
    "list_directory": tool_list_directory,
    "read_file": tool_read_file,
    "search_files": tool_search_files,
    "web_search": tool_web_search,
    "web_extract": tool_web_extract,
    "write_file": tool_write_file,        # 🆕 写工具
    "edit_file": tool_edit_file,          # 🆕 编辑工具
}


# ============================================================
# 4. 6 个节点
# ============================================================

def load_history_node(state: AgentState) -> AgentState:
    """🆕 节点 -1:从 memory_v2 拉最近 10 条对话(老板的"我的意思是这个"能看到上面聊啥)"""
    chat_id = state.get("chat_id", "")
    if not chat_id:
        return state
    try:
        import sqlite3
        conn = sqlite3.connect(DEDUP_DB, timeout=5)
        c = conn.cursor()
        c.execute('''SELECT user_msg, bot_reply, msg_type, ts FROM memory_v2
                     WHERE chat_id = ? ORDER BY ts DESC LIMIT 10''', (chat_id,))
        rows = c.fetchall()
        conn.close()
        # 倒序(从老到新),LLM 看时序
        history = []
        for user_msg, bot_reply, msg_type, ts in reversed(rows):
            history.append({
                "user": user_msg[:500],
                "assistant": bot_reply[:500],
                "type": msg_type,
                "ts": ts
            })
        state["history"] = history
        print(f"   📜 [load_history] 加载 {len(history)} 条历史对话")
        for i, h in enumerate(history[-3:], 1):
            print(f"      [{i}] user: {h['user'][:60]}")
    except Exception as e:
        print(f"   ⚠️ load_history 失败: {e}")
        state["history"] = []
    return state


def multimodal_node(state: AgentState) -> AgentState:
    """节点 0:多模态预处理(语音/图片 → 文字)"""
    raw = state.get("raw_message", {})
    msg_type = state.get("msg_type", "text")

    if msg_type in ("text",):
        return state  # 文字不用处理

    # 语音/图片 → 调 multimodal.py
    try:
        from multimodal import process_lark_message
        result = process_lark_message(raw)
        state["user_msg"] = result["text"]
        print(f"   🎤📷 multimodal: type={msg_type} → text={result['text'][:150]}")
    except Exception as e:
        print(f"   ❌ multimodal 错误: {e}")
        state["user_msg"] = f"[{msg_type} 消息处理失败]"

    return state


def thinking_node(state: AgentState) -> AgentState:
    """节点 1:思考(强制走)"""
    print(f"\n🧠 [thinking] 分析:{state['user_msg']}")
    # 🆕 把历史对话塞进 prompt(让 LLM 知道上面聊了啥)
    history_text = ""
    history = state.get("history", [])
    if history:
        history_lines = []
        for h in history[-5:]:  # 最近 5 条
            history_lines.append(f"老板:{h['user'][:100]}")
            history_lines.append(f"Bot:{h['assistant'][:150]}")
        history_text = "\n".join(history_lines)
    prompt = f"""分析老板的话(参考上面的历史对话):

【历史对话 / History】
{history_text if history_text else '(无 / none)'}

【老板刚才说 / Latest message】
"{state['user_msg']}"

老板在越南管韩资/中资工厂,知识库在 ~/Desktop/越南制造业知识库/。
可能涉及:列目录/读文件/搜文件/搜网络/翻译/闲聊/历史对话/引用上文。

⚠️ 重要判断:老板是不是在"询问或引用之前发过的内容(图/视频/语音/文件/对话/上文)"?
- "我的意思是这个"→ true(指上文)
- "上面那个视频讲什么"→ true
- "再看看我刚才发的图"→ true
- "那个文件做什么用"→ true
- "帮我翻译上面那张图"→ true
- "分析这个视频"(指当前发的)→ false
- "你好"→ false

返回(JSON):
{{
  "intent": "老板想要什么",
  "needs_tool": true/false,
  "language": "zh/ko/vi/en",
  "needs_history": true/false
}}"""
    result = call_llm(prompt)
    # 去掉 ```json ... ``` 标记
    result_clean = re.sub(r'```[a-z]*\n?', '', result).replace('```', '').strip()
    print(f"   思考结果: {result[:200]}")
    try:
        parsed = json.loads(result_clean)
        state["intent"] = parsed.get("intent", "")
        state["needs_tool"] = parsed.get("needs_tool", False)
        state["user_lang"] = parsed.get("language", "zh")
        state["needs_history"] = parsed.get("needs_history", False)
    except:
        # 改用更全面的关键词 fallback
        msg = state["user_msg"]
        tool_keywords = [
            "列", "有什么", "有什么文件", "知识库", "目录",
            "读", "read", "打开", "看一下",
            "搜", "search", "找", "查", "看一下", "翻",
            "翻译", "translate", "번역",
            "文件", "资料", "目录",
        ]
        needs_tool = any(kw in msg for kw in tool_keywords)
        # 但排除纯问候
        greetings = ["你好", "在吗", "hi", "hello", "你是谁", "是谁", "介绍"]
        if any(g in msg for g in greetings):
            needs_tool = False
        # fallback:历史查询 — 用 LLM 简单判断语义
        history_hints = [
            "上面", "之前", "刚才", "上次", "那个", "这个", "前面",
            "再找", "再查", "再核", "再对", "再分析", "重新分析",
            "동영상", "영상", "그림", "그거", "이거", "다시", "한번",
        ]
        needs_history = any(kw in msg for kw in history_hints)
        state["intent"] = result[:100]
        state["needs_tool"] = needs_tool
        state["user_lang"] = "zh"
        state["needs_history"] = needs_history
    return state


def choose_tool_node(state: AgentState) -> AgentState:
    """节点 2:选工具(强制走)"""
    if not state.get("needs_tool"):
        print(f"   [choose_tool] 不需要工具")
        return state

    tools_desc = """
- list_directory: 列目录下文件(参数:path)
- read_file: 读文件内容(参数:path)
- search_files: 找文件名(参数:pattern)
- web_search: 搜网络(参数:query)
- web_extract: 抓网页(参数:url)
- write_file: 🆕 写新文件(参数:path, content)
- edit_file: 🆕 编辑文件(参数:path, old_text, new_text)
"""
    print(f"   [choose_tool] 选工具")
    prompt = f"""老板:{state['user_msg']}
意图:{state.get('intent')}

工具列表:{tools_desc}

知识库根目录:~/Desktop/越南制造业知识库/

返回(JSON):
{{"tool": "工具名 或 null", "params": {{...}}}}"""
    result = call_llm(prompt)
    # 去掉 ```json ... ``` 标记
    result_clean = re.sub(r'```[a-z]*\n?', '', result).replace('```', '').strip()
    print(f"   选工具结果: {result[:200]}")
    try:
        parsed = json.loads(result_clean)
        state["chosen_tool"] = parsed.get("tool")
        state["tool_params"] = parsed.get("params", {})
        if state["chosen_tool"] and state["chosen_tool"] not in TOOLS_DISPATCH:
            # LLM 编了个不存在的工具
            state["chosen_tool"] = None
    except:
        # 改进的 fallback:看关键词
        msg = state["user_msg"]
        # 1. 网络关键词 → web_search
        if any(kw in msg for kw in ["搜", "search", "找"]) and any(net in msg for net in ["网", "新闻", "最新", "趋势", "news", "what", "how", "why"]):
            state["chosen_tool"] = "web_search"
            state["tool_params"] = {"query": msg}
        # 2. 文件关键词 → search_files
        elif any(kw in msg for kw in ["找文件", "搜文件", "找资料", "搜资料"]):
            state["chosen_tool"] = "search_files"
            state["tool_params"] = {"pattern": msg}
        # 3. 读文件
        elif any(kw in msg for kw in ["读", "read", "打开"]):
            state["chosen_tool"] = "read_file"
            state["tool_params"] = {"path": KNOWLEDGE_BASE}
        # 4. 列目录
        elif any(kw in msg for kw in ["列", "list", "有什么"]):
            state["chosen_tool"] = "list_directory"
            state["tool_params"] = {"path": KNOWLEDGE_BASE}
        # 5. 都不匹配,不强行选
        else:
            state["chosen_tool"] = None
            state["tool_params"] = {}
    return state


def execute_tool_node(state: AgentState) -> AgentState:
    """节点 3:执行工具(强制走)"""
    tool = state.get("chosen_tool")
    if not tool:
        return state
    print(f"   [execute_tool] 调 {tool} 参数 {state.get('tool_params')}")
    try:
        func = TOOLS_DISPATCH[tool]
        result = func(**state.get("tool_params", {}))
        state["tool_result"] = result
        state["tool_error"] = ""
        print(f"   ✅ 工具结果(前 300 字):{result[:300]}")
    except Exception as e:
        state["tool_result"] = ""
        state["tool_error"] = f"{type(e).__name__}: {e}"
        print(f"   ❌ 错误: {state['tool_error']}")
    state["tool_attempts"] = state.get("tool_attempts", 0) + 1
    return state


def verify_node(state: AgentState) -> AgentState:
    """节点 4:验证(失败可重试)"""
    if not state.get("chosen_tool"):
        state["verified"] = True
        return state
    if state.get("tool_error") and state.get("tool_attempts", 0) < 2:
        print(f"   [verify] 失败,准备重试")
        state["verified"] = False
        return state
    state["verified"] = bool(state.get("tool_result"))
    print(f"   [verify] 通过={state['verified']}")
    return state


def respond_node(state: AgentState) -> AgentState:
    """节点 5:LLM 整理回复"""
    print(f"   [respond] LLM 整理")
    context = f"""老板问:{state['user_msg']}
意图:{state.get('intent')}
工具:{state.get('chosen_tool') or '没调'}
结果:{state.get('tool_result') or state.get('tool_error') or '无'}

用大白话整理成简洁答案(2-3 段,200-400 字):
- 一句话结论
- 为什么
- 怎么办"""
    answer = call_llm(
        context,
        system_prompt=BISEO_SYSTEM_PROMPT
    )
    state["final_answer"] = answer
    return state


# ============================================================
# 5. 构建图
# ============================================================
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("load_history", load_history_node)  # 🆕 先加载历史
    workflow.add_node("thinking", thinking_node)
    workflow.add_node("choose_tool", choose_tool_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("multimodal", multimodal_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("respond", respond_node)

    workflow.set_entry_point("load_history")  # 🆕 入口改为 load_history
    workflow.add_edge("load_history", "multimodal")  # 🆕 load_history → multimodal
    workflow.add_edge("multimodal", "thinking")
    workflow.add_edge("thinking", "choose_tool")
    workflow.add_conditional_edges(
        "choose_tool",
        lambda s: "execute_tool" if s.get("chosen_tool") else "respond",
        {"execute_tool": "execute_tool", "respond": "respond"}
    )
    workflow.add_edge("execute_tool", "verify")
    workflow.add_conditional_edges(
        "verify",
        lambda s: "execute_tool" if (not s.get("verified") and s.get("tool_attempts", 0) < 2) else "respond",
        {"execute_tool": "execute_tool", "respond": "respond"}
    )
    workflow.add_edge("respond", END)
    return workflow


def run_demo():
    """跑 demo:3 个测试场景"""
    print("=" * 70)
    print("🚀 LangGraph Demo for biseo Bot")
    print("=" * 70)

    workflow = build_graph()

    test_cases = [
        "知识库里有什么?",
        "搜一下越南制造业 2026",
        "你是谁?",
    ]

    with SqliteSaver.from_conn_string("demo_checkpoints.db") as memory:
        app = workflow.compile(checkpointer=memory)
        for i, msg in enumerate(test_cases, 1):
            print(f"\n{'='*70}")
            print(f"📝 测试 {i}:{msg}")
            print(f"{'='*70}")
            config = {"configurable": {"thread_id": f"demo-{i}"}}
            state = {
                "user_msg": msg,
                "user_lang": "zh",
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
            result = app.invoke(state, config=config)
            elapsed = time.time() - start
            print(f"\n⏱️ 耗时:{elapsed:.1f}s")
            print(f"📊 走过的节点:thinking → choose_tool → {'execute_tool → verify → ' if result.get('chosen_tool') else ''}respond")
            print(f"\n💬 Bot 回复:\n{result['final_answer']}\n")

    print("=" * 70)
    print("✅ Demo 跑完")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
