import os
import shutil
import string
import asyncio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage

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


# ================= 3. 定义两个工具 =================

@tool
def list_all_drives() -> str:
    """获取本机所有的硬盘盘符列表。
    如果用户想知道有哪些硬盘，或者你需要查询特定硬盘但不知道盘符时，请先调用此工具。
    """
    print("[⚡ 本地执行中...] 正在扫描本机硬盘...")
    if os.name == 'nt':  # Windows 系统
        drives = []
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append(drive)
        return f"本机检测到的硬盘有: {', '.join(drives)}"
    else:  # Mac/Linux 系统
        return "本机是 Mac/Linux 系统，根目录挂载在 '/' 下。"


@tool(parse_docstring=True)
def check_disk_space(drive_path: str) -> str:
    """查询指定硬盘或路径的空间使用情况。

    Args:
        drive_path: 必须是一个具体的硬盘盘符或路径，例如 "C:\\" 或 "/"。
    """
    print(f"[⚡ 本地执行中...] 正在查询 {drive_path} 的空间...")
    try:
        # 清理一下路径格式，防止模型传 "C:" 导致 shutil 报错
        if os.name == 'nt' and len(drive_path) == 2 and drive_path[1] == ':':
            drive_path += "\\"

        total, used, free = shutil.disk_usage(drive_path)
        total_gb = total / (1024 ** 3)
        used_gb = used / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        used_percent = (used / total) * 100

        return (
            f"路径: {drive_path}\n"
            f"总容量: {total_gb:.2f} GB\n"
            f"已使用: {used_gb:.2f} GB ({used_percent:.1f}%)\n"
            f"剩余可用: {free_gb:.2f} GB"
        )
    except Exception as e:
        return f"查询硬盘 {drive_path} 失败，错误信息: {e}"


# 4. 绑定多个工具到模型
tools = [list_all_drives, check_disk_space]
model_with_tools = model.bind_tools(tools)
tool_map = {t.name: t for t in tools}


# ================= 5. 核心 Agent 循环（处理多工具串联） =================
async def run_agent_stream(user_query: str):
    print(f"\n[用户]: {user_query}\n")

    # ★ 提示词至关重要：指导大模型的工作流
    system_prompt = (
        f"你是一个系统助手。当前用户的操作系统是 {os.name}。\n"
        "注意你的工作流程：\n"
        "1. 如果用户询问硬盘空间，但不知道有哪些盘，或者没有指定盘符，你必须先调用 `list_all_drives` 工具获取硬盘列表。\n"
        "2. 然后再调用 `check_disk_space` 工具去查询具体硬盘的空间。\n"
        "3. 在最终回答中，请直接用人话总结结果。"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]

    # 我们把整个交互过程放在一个循环里，因为可能存在“连续多次调用工具”的情况
    while True:
        tool_calls_this_round = []

        # 发起请求
        async for event in model_with_tools.astream_events(messages, version="v2"):
            event_type = event["event"]

            # 流式打印大模型的普通文字
            if event_type == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    print(f"\033[96m{chunk.content}\033[0m", end="", flush=True)

            # 捕获完整输出，提取工具请求
            elif event_type == "on_chat_model_end":
                output_msg = event["data"]["output"]
                if hasattr(output_msg, "tool_calls") and output_msg.tool_calls:
                    messages.append(output_msg)
                    tool_calls_this_round.extend(output_msg.tool_calls)

        # 如果本轮没有工具调用，说明大模型已经给出了最终回答，退出循环
        if not tool_calls_this_round:
            print()
            break

        # 如果本轮有工具调用，执行它们，并把结果加回 messages
        for tool_call in tool_calls_this_round:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"\n\n[🔍 截获请求]: 调用 `{tool_name}`，参数: {tool_args}")

            if tool_name in tool_map:
                try:
                    # 执行工具
                    tool_result = tool_map[tool_name].invoke(tool_args)
                    print(f"[✅ 结果]: \n{tool_result}")

                    # 将结果送回给模型
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_id)
                    )
                except Exception as e:
                    messages.append(
                        ToolMessage(content=f"工具执行错误: {e}", tool_call_id=tool_id)
                    )
        print("\n--- AI 继续思考 ---")
        # 循环会继续，带着刚查到的结果再次请求模型


if __name__ == "__main__":
    # 注意：这次提问故意模糊，不说 C 盘，看 AI 是否先调 list_all_drives
    asyncio.run(run_agent_stream("帮我看看本机硬盘空间使用情况？"))