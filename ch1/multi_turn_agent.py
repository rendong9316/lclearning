import os
import shutil
import string
import asyncio
import subprocess
import sys
import time
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessageChunk

# 0. 加载项目根目录下的 .env
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# 1. 从 .env 读取 API 配置
AGNES_API_KEY = os.getenv("AGNES_API_KEY")
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.cn/v1")
MODEL_NAME = os.getenv("AGNES_MODEL", "agnes-3.0-flash")

# 2. 初始化模型
model = init_chat_model(
    model=MODEL_NAME,
    model_provider="openai",
    api_key=AGNES_API_KEY,
    base_url=AGNES_BASE_URL,
)

# ================= 3. 定义工具 =================
WORKSPACE_DIR = "./agent_workspace"
os.makedirs(WORKSPACE_DIR, exist_ok=True)


@tool(parse_docstring=True)
def list_all_drives() -> str:
    """获取本机所有的硬盘盘符列表。
    如果用户想知道有哪些硬盘，或者你需要查询特定硬盘但不知道盘符时，请先调用此工具。
    """
    print("\n\033[90m[⚡ 本地执行中...] 正在扫描本机硬盘...\033[0m")
    if os.name == 'nt':
        drives = [f"{letter}:\\" for letter in string.ascii_uppercase if os.path.exists(f"{letter}:\\")]
        return f"本机检测到的硬盘有: {', '.join(drives)}"
    else:
        return "本机是 Mac/Linux 系统，根目录挂载在 '/' 下。"


@tool(parse_docstring=True)
def check_disk_space(drive_path: str) -> str:
    """查询指定硬盘或路径的空间使用情况。

    Args:
        drive_path: 必须是一个具体的硬盘盘符或路径，例如 "C:\\" 或 "/"。
    """
    print(f"\n\033[90m[⚡ 本地执行中...] 正在查询 {drive_path} 的空间...\033[0m")
    try:
        if os.name == 'nt' and len(drive_path) == 2 and drive_path[1] == ':':
            drive_path += "\\"
        total, used, free = shutil.disk_usage(drive_path)
        return (f"路径: {drive_path}\n总容量: {total / (1024 ** 3):.2f} GB\n"
                f"已使用: {used / (1024 ** 3):.2f} GB ({(used / total) * 100:.1f}%)\n"
                f"剩余可用: {free / (1024 ** 3):.2f} GB")
    except Exception as e:
        return f"查询硬盘 {drive_path} 失败，错误信息: {e}"


@tool(parse_docstring=True)
def write_script(filename: str, code: str) -> str:
    """当你需要执行复杂的系统操作（比如列出文件夹内容、文本处理等），但没有现成工具时，可以使用此工具写一个Python脚本。

    Args:
        filename: 脚本文件名，必须以 .py 结尾，例如 "list_files.py"
        code: 完整的 Python 代码内容，代码中不能包含 input() 等阻塞操作。
    """
    safe_filename = os.path.basename(filename)
    if not safe_filename.endswith(".py"):
        safe_filename += ".py"
    filepath = os.path.join(WORKSPACE_DIR, safe_filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    return f"脚本已成功写入: {filepath}"


@tool(parse_docstring=True)
def run_script(filename: str) -> str:
    """执行之前用 write_script 写好的 Python 脚本，并获取控制台输出。

    Args:
        filename: 要执行的脚本文件名，例如 "list_files.py"
    """
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(WORKSPACE_DIR, safe_filename)
    if not os.path.exists(filepath):
        return f"错误：找不到脚本 {filepath}，请先调用 write_script 写入。"
    try:
        result = subprocess.run(
            [sys.executable, filepath], capture_output=True, text=True, timeout=15, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            return f"执行成功，输出如下：\n{result.stdout.strip() or '（无输出）'}"
        else:
            return f"执行失败，错误信息：\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "错误：脚本执行超时（超过15秒）。"
    except Exception as e:
        return f"系统执行异常：{e}"


# ================= 4. 初始化工具列表 =================
tools = [list_all_drives, check_disk_space, write_script, run_script]
model_with_tools = model.bind_tools(tools)
tool_map = {t.name: t for t in tools}


# ================= 5. Agent 类封装 =================
class FastConsoleAgent:
    def __init__(self):
        self.messages = [
            SystemMessage(content=(
                f"你是一个极其强大的系统助手。当前用户的操作系统是 {os.name}。\n"
                "【工作流指导】\n"
                "1. 查询硬盘列表：使用 `list_all_drives`。\n"
                "2. 查询硬盘空间：使用 `check_disk_space`。\n"
                "3. 如果用户需要查询文件、处理复杂逻辑，但你**没有现成工具**时，你必须按以下步骤工作：\n"
                "   a. 调用 `write_script`，写一段完整的、能独立运行的 Python 脚本（文件名用 .py 结尾）。\n"
                "   b. 调用 `run_script` 执行该脚本。\n"
                "   c. 如果执行报错，仔细看报错信息，修改代码重新写入并执行（最多重试 3 次）。\n"
                "【注意】你的代码里只能使用 Python 标准库，不能包含 input() 阻塞代码。\n"
                "在最终回答中，请直接用人话总结结果，不要输出代码本身。"
            ))
        ]

    async def _timer(self, stop_event: asyncio.Event):
        """极轻量级的后台计时器"""
        start = time.perf_counter()
        while not stop_event.is_set():
            elapsed = time.perf_counter() - start
            sys.stdout.write(f"\r\033[K\033[93m[⏳ 思考中... {elapsed:.1f}s]\033[0m")
            sys.stdout.flush()
            try:
                # 等待 0.3 秒，如果还没结束就再刷新一次
                await asyncio.wait_for(stop_event.wait(), timeout=0.3)
            except asyncio.TimeoutError:
                pass
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    async def chat(self, user_query: str):
        self.messages.append(HumanMessage(content=user_query))
        step_count = 0
        total_start = time.perf_counter()

        while True:
            step_count += 1
            if step_count > 8:
                print("\n\033[91m[系统] 思考步数超限，强制中断。\033[0m")
                break

            # 启动计时器
            stop_timer = asyncio.Event()
            timer_task = asyncio.create_task(self._timer(stop_timer))

            first_token = False
            ai_chunk = None  # 累积 AIMessageChunk

            # ★ 极限优化：直接调用底层的 astream，抛弃所有事件包装
            async for chunk in model_with_tools.astream(self.messages):
                # 累积流式块
                if ai_chunk is None:
                    ai_chunk = chunk
                else:
                    ai_chunk += chunk

                has_content = bool(chunk.content)
                has_tool = bool(chunk.tool_call_chunks)

                # 一旦收到第一个字或工具意图，立刻停止计时并切色
                if (has_content or has_tool) and not first_token:
                    first_token = True
                    stop_timer.set()
                    await timer_task
                    print("\033[96m", end="")  # 变青色

                # 输出正常文本
                if has_content:
                    print(chunk.content, end="", flush=True)

                # 输出工具参数构建过程
                if has_tool:
                    for tc in chunk.tool_call_chunks:
                        if tc.get("name"):
                            print(f"\n\n\033[93m[🤔 决定调用: {tc['name']}...]\033[0m", end="", flush=True)
                        if tc.get("args"):
                            print(f"\033[90m{tc['args']}\033[0m", end="", flush=True)

            # 处理没有输出直接结束的极端情况
            if not first_token:
                stop_timer.set()
                await timer_task

            if first_token:
                print("\033[0m")  # 恢复颜色并换行

            # 检查这一轮是否包含完整的工具调用
            if not ai_chunk or not getattr(ai_chunk, "tool_calls", None):
                break

            # 有工具调用，存入历史
            self.messages.append(ai_chunk)

            # 并行执行工具
            print("\n")
            for tool_call in ai_chunk.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                print(f"\033[93m[🚀 执行动作]: {tool_name}\033[0m")
                print(f"\033[93m[📦 参数]: {json.dumps(tool_args, ensure_ascii=False)}\033[0m")

                if tool_name in tool_map:
                    try:
                        print(f"\033[90m(本地执行中...)\033[0m")
                        # ★ 极限优化：用 to_thread 异步执行工具，防止阻塞事件循环
                        tool_result = await asyncio.to_thread(tool_map[tool_name].invoke, tool_args)

                        res_str = str(tool_result)
                        display_res = res_str if len(res_str) < 300 else res_str[:300] + "...(截断)"
                        print(f"\033[92m[📥 结果]: \n{display_res}\033[0m")

                        self.messages.append(ToolMessage(content=res_str, tool_call_id=tool_id))
                    except Exception as e:
                        err = f"执行错误: {e}"
                        print(f"\033[91m[❌ {err}]\033[0m")
                        self.messages.append(ToolMessage(content=err, tool_call_id=tool_id))

            print("\033[90m--- AI 继续思考 ---\033[0m")

        total_time = time.perf_counter() - total_start
        print(f"\n\033[94m[🏁 本轮总耗时: {total_time:.2f} 秒]\033[0m")


# ================= 6. 控制台主循环 =================
async def main():
    agent = FastConsoleAgent()
    print("=" * 60)
    print("🤖 极限速度版 Agent 已启动 (纯网络流式)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                continue

            print("\n🤖 Agent: ", end="")
            await agent.chat(user_input)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n\033[91m发生异常: {e}\033[0m")


if __name__ == "__main__":
    asyncio.run(main())