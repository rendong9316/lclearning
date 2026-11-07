import os
from dotenv import load_dotenv
from rich import print as rprint
from langchain.chat_models import init_chat_model

# 0. 加载项目根目录下的 .env（无论从哪个目录运行脚本都能找到）
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# 1. 从 .env 读取 API 配置
AGNES_API_KEY = os.getenv("AGNES_API_KEY")
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.cn/v1")
MODEL_NAME = os.getenv("AGNES_MODEL", "agnes-3.0-flash")

# 2. 初始化模型
# 核心技巧：使用 "openai:" 前缀欺骗 LangChain 使用 OpenAI 协议，
# 但通过 base_url 将请求指向 Agnes 的服务器
model = init_chat_model(
    model=MODEL_NAME,          # 注意这里：直接传 "agnes-3.0-flash"，不要加 openai: 前缀！
    model_provider="openai",   # 用这个参数告诉 LangChain 使用 OpenAI 协议
    api_key=AGNES_API_KEY,
    base_url=AGNES_BASE_URL,
    # max_tokens=10
)

# 3. 发起调用
try:
    print("正在请求模型...")
    response = model.stream("为什么中国人并不承认64事件")
    print("\n--- 模型回复 ---")
    # rprint(response)
    for chunk in response:
        print(chunk.text,end="",flush=True)
except Exception as e:
    print(f"\n请求失败，错误信息：{e}")
