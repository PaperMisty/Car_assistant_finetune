import json
import os
import random
import time
from dotenv import load_dotenv
from openai import OpenAI

from config.prompt import SFTDataGeneratorPrompt
from config.data_format_checker import DataFormatChecker

# 加载 .env
load_dotenv(override=True)

# 1. 初始化两组客户端与模型参数
qwen_client = OpenAI(api_key=os.getenv("QWEN_API_KEY"), base_url=os.getenv("QWEN_API_BASE"))
qwen_model = os.getenv("LLM_DEFAULT_MODEL_5", "qwen3.8-flash")

gemini_client = OpenAI(api_key=os.getenv("Gemini_API_KEY"), base_url=os.getenv("Gemini_BASE_URL"))
gemini_model = os.getenv("Gemini_MODEL_NAME", "gemini-3.7-flash-medium")

# 2. 读取种子数据并随机挑选 3 条（固定种子便于可复现对比）
seed_file_path = r"data/v2/seeds/train/category1_expanded_021_080.jsonl"
seeds = []
with open(seed_file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            seeds.append(json.loads(line))

random.seed(2026)
# 挑选3条涵盖不同子场景的种子
tool_seeds = [s for s in seeds if s.get("tool_required")]
conv_seeds = [s for s in seeds if not s.get("tool_required")]

selected_seeds = [
    conv_seeds[0],  # cat1_feature_021 车机蓝牙 (纯对话)
    tool_seeds[1],  # cat1_feature_023 远程控车 (Tool Call)
    conv_seeds[5],  # cat1_feature_031 座椅记忆 (纯对话)
]

print("==================================================")
print(f"已选取 3 条对比测试种子:")
for s in selected_seeds:
    print(f" - [{s.get('seed_id')}] {s.get('subcategory')} | 需工具: {s.get('tool_required')}")
print("==================================================\n")


def generate_with_model(client, model_name, is_qwen=True):
    results = []
    print(f"\n>>>>>> 开始测试模型: {model_name} (Thinking: Disabled) <<<<<<")

    for i, seed in enumerate(selected_seeds, 1):
        seed_id = seed.get("seed_id")
        subcat = seed.get("subcategory")
        print(f"[{i}/3] 正在生成: {seed_id} ({subcat})...")

        payload = SFTDataGeneratorPrompt.get_generator_payload(seed)

        # 请求参数准备
        kwargs = {
            "model": model_name,
            "messages": payload,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
        if is_qwen:
            # Qwen 关闭思考模式
            kwargs["extra_body"] = {"enable_thinking": False}

        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
            latency = time.perf_counter() - t0

            content = resp.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            parsed_data = json.loads(content)
            if isinstance(parsed_data, list) and len(parsed_data) > 0 and isinstance(parsed_data[0], dict):
                parsed_data = parsed_data[0]

            is_valid, errors, warnings = DataFormatChecker.check_sample(parsed_data)

            usage = resp.usage
            tokens = usage.completion_tokens if usage else 0
            speed = tokens / latency if latency > 0 else 0

            print(
                f"   -> 完成! 耗时: {latency:.2f}s | 生成 Token: {tokens} | 速度: {speed:.2f} t/s | 格式通过: {is_valid}"
            )

            results.append(
                {
                    "seed_id": seed_id,
                    "latency_seconds": latency,
                    "completion_tokens": tokens,
                    "speed_tps": speed,
                    "is_valid": is_valid,
                    "errors": errors,
                    "warnings": warnings,
                    "data": parsed_data,
                }
            )
        except Exception as e:
            print(f"   -> 失败: {e}")

    return results


# 执行 Qwen 测试
qwen_results = generate_with_model(qwen_client, qwen_model, is_qwen=True)

# 执行 Gemini 测试
gemini_results = generate_with_model(gemini_client, gemini_model, is_qwen=False)

# 保存两份日志
os.makedirs("test_samples_logs", exist_ok=True)

with open("test_samples_logs/3.qwen-flash-no-thinking.json", "w", encoding="utf-8") as f:
    json.dump(qwen_results, f, ensure_ascii=False, indent=2)

with open("test_samples_logs/3.gemini-flash.json", "w", encoding="utf-8") as f:
    json.dump(gemini_results, f, ensure_ascii=False, indent=2)

print("\n==================================================")
print("对比测试已全部完成！结果已分别保存至:")
print(" - test_samples_logs/3.qwen-flash-no-thinking.json")
print(" - test_samples_logs/3.gemini-flash.json")
print("==================================================")

