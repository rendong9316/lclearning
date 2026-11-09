import os
import time
from dotenv import load_dotenv
from rich import print as rprint
from langchain.chat_models import init_chat_model

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

# 3. 发起调用并计时
try:
    print("正在请求模型...")

    # 记录发起请求的绝对时间
    start_time = time.perf_counter()

    response = model.stream("北京今天天气如何？")
    print("\n--- 模型回复 ---")

    is_first_chunk = True
    first_token_time = 0.0

    for chunk in response:
        # 过滤掉空的心跳包，只抓取真正有文字的内容
        if chunk.text:
            if is_first_chunk:
                first_token_time = time.perf_counter() - start_time
                print(f"\n\033[93m[⏱️ 首字延迟 (TTFT): {first_token_time:.2f} 秒]\033[0m\n")
                is_first_chunk = False

            # 流式打印内容
            print(chunk.text, end="", flush=True)

    total_time = time.perf_counter() - start_time
    print(f"\n\n\033[94m[🏁 生成完毕，总耗时: {total_time:.2f} 秒]\033[0m")

except Exception as e:
    print(f"\n请求失败，错误信息：{e}")