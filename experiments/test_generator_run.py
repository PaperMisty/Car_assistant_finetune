import json
import os
import random
import time
from dotenv import load_dotenv
from openai import OpenAI

from config.prompt import SFTDataGeneratorPrompt

# 加载 .env 配置
load_dotenv(override=True)

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("QWEN_API_BASE")
model_name = os.getenv("LLM_DEFAULT_MODEL", "qwen3.8-27b")
# 调高温度至 0.7，增强生成多样性
temperature = 0.7

print(f"正在连接模型: {model_name} @ {base_url} (Temperature: {temperature})")

client = OpenAI(api_key=api_key, base_url=base_url)

# 读取种子数据文件
seed_file_path = r"data/v2/seeds/train/category1_expanded_021_080.jsonl"
seeds = []
with open(seed_file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            seeds.append(json.loads(line))

print(f"共读取到 {len(seeds)} 条种子数据，正在随机抽取 3 条进行 v2 Prompt 测试...\n")

# 用当前时间戳作为随机种子，保证每次随机不同
random.seed(int(time.time()))
selected_seeds = random.sample(seeds, 3)

results = []

for i, seed in enumerate(selected_seeds, 1):
    seed_id = seed.get("seed_id")
    subcategory = seed.get("subcategory")
    scenario = seed.get("scenario")
    is_tool = seed.get("tool_required")
    print(f"[{i}/3] 正在合成 Seed: {seed_id} | 子类: {subcategory} | 是否需工具: {is_tool}")
    print(f"    场景描述: {scenario}")

    payload = SFTDataGeneratorPrompt.get_generator_payload(seed)

    try:
        response = client.chat.completions.create(
            model=model_name, messages=payload, temperature=temperature, response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()

        # 清理 markdown 代码块包裹
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        parsed_json = json.loads(content)
        if isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict):
            parsed_json = parsed_json[0]

        results.append(parsed_json)
        msg_count = len(parsed_json.get("messages", [])) if isinstance(parsed_json, dict) else len(parsed_json)
        print(f"    -> 成功生成！生成 messages 轮数: {msg_count}\n")
    except Exception as e:
        print(f"    -> 生成失败: {e}\n")

# 确保目录存在
os.makedirs("test_samples_logs", exist_ok=True)
output_file = "test_samples_logs/2.qwen-27b@prompt-v2.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"所有试跑完成！结果已保存至 {output_file}")
