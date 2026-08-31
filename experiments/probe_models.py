import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
# 1. 测试 Qwen 关闭思考
qwen_key = os.getenv("QWEN_API_KEY")
qwen_base = os.getenv("QWEN_API_BASE")
qwen_model = os.getenv("LLM_DEFAULT_MODEL_5", "qwen3.8-flash")

print("--- 测试 1: Qwen (关闭 thinking) ---")
qwen_client = OpenAI(api_key=qwen_key, base_url=qwen_base)
try:
    t0 = time.perf_counter()
    resp = qwen_client.chat.completions.create(
        model=qwen_model,
        messages=[{"role": "user", "content": "用一句话介绍你自己"}],
        extra_body={"enable_thinking": False},
    )
    t1 = time.perf_counter()
    print(f"Qwen 成功! 耗时: {t1-t0:.2f}s, 内容: {resp.choices[0].message.content}")
    print(f"Token: {resp.usage}")
except Exception as e:
    print(f"Qwen extra_body enable_thinking 报错: {e}")
    # 尝试不带参数或不同参数
    try:
        t0 = time.perf_counter()
        resp = qwen_client.chat.completions.create(
            model=qwen_model, messages=[{"role": "user", "content": "用一句话介绍你自己"}]
        )
        t1 = time.perf_counter()
        print(f"Qwen 默认请求耗时: {t1-t0:.2f}s, 内容: {resp.choices[0].message.content}")
    except Exception as e2:
        print(f"Qwen 默认请求也失败: {e2}")

# 2. 测试 Local Gemini
gemini_key = os.getenv("Gemini_API_KEY")
gemini_base = os.getenv("Gemini_BASE_URL")
gemini_model = os.getenv("Gemini_MODEL_NAME", "gemini-3.7-flash-medium")

print("\n--- 测试 2: Local Gemini ---")
gemini_client = OpenAI(api_key=gemini_key, base_url=gemini_base)
try:
    t0 = time.perf_counter()
    resp = gemini_client.chat.completions.create(
        model=gemini_model,
        messages=[{"role": "user", "content": "用一句话介绍你自己"}],
        extra_body={"thinking_budget": 0},
    )
    t1 = time.perf_counter()
    print(f"Gemini 成功! 耗时: {t1-t0:.2f}s, 内容: {resp.choices[0].message.content}")
    print(f"Token: {resp.usage}")
except Exception as e:
    print(f"Gemini thinking_budget=0 报错: {e}")
    try:
        t0 = time.perf_counter()
        resp = gemini_client.chat.completions.create(
            model=gemini_model, messages=[{"role": "user", "content": "用一句话介绍你自己"}]
        )
        t1 = time.perf_counter()
        print(f"Gemini 默认请求成功! 耗时: {t1-t0:.2f}s, 内容: {resp.choices[0].message.content}")
        print(f"Token: {resp.usage}")
    except Exception as e2:
        print(f"Gemini 默认请求也失败: {e2}")
