import json
import os
import random
import time
from dotenv import load_dotenv
from openai import OpenAI

from config.prompt import SFTDataGeneratorPrompt
from config.data_format_checker import DataFormatChecker

# 加载 .env 配置
load_dotenv(override=True)

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("QWEN_API_BASE")
# 测试指定模型: qwen3.8-flash (从 LLM_DEFAULT_MODEL_5 或 fallback)
model_name = os.getenv("LLM_DEFAULT_MODEL_5", "qwen3.8-flash")
temperature = 0.7

print(f"==================================================")
print(f"正在测试模型: {model_name}")
print(f"API Base URL: {base_url}")
print(f"==================================================\n")

client = OpenAI(api_key=api_key, base_url=base_url)

# 读取 1 条随机种子
seed_file_path = r"data/v2/seeds/train/category1_expanded_021_080.jsonl"
seeds = []
with open(seed_file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            seeds.append(json.loads(line))

random.seed(int(time.time()))
sample_seed = random.choice(seeds)

print(f"[测试种子] Seed ID: {sample_seed.get('seed_id')} | 子类: {sample_seed.get('subcategory')}")
print(f"           场景: {sample_seed.get('scenario')}")
print(f"           是否需工具: {sample_seed.get('tool_required')}\n")

payload = SFTDataGeneratorPrompt.get_generator_payload(sample_seed)

# 开始计时并调用
start_time = time.perf_counter()

response = client.chat.completions.create(
    model=model_name,
    messages=payload,
    temperature=temperature,
    response_format={"type": "json_object"}
)

end_time = time.perf_counter()
latency = end_time - start_time

# 1. 打印原生 OpenAI Response 对象的顶层结构
print("=" * 60)
print("【1. OpenAI ChatCompletion 原生 Response 结构】")
print("=" * 60)
print(f"Response ID        : {response.id}")
print(f"Object Type        : {response.object}")
print(f"Created Timestamp  : {response.created}")
print(f"Model Name Returned: {response.model}")
if response.usage:
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    tps = completion_tokens / latency if latency > 0 else 0
    print(f"Usage Info         : Prompt={prompt_tokens} tokens, Completion={completion_tokens} tokens, Total={total_tokens} tokens")
    print(f"总耗时 (Latency)   : {latency:.2f} 秒")
    print(f"生成速率 (Speed)   : {tps:.2f} tokens/s")
print(f"Finish Reason      : {response.choices[0].finish_reason}")

# 2. 打印生成的内容
raw_content = response.choices[0].message.content.strip()

# 清洗 markdown 代码块
clean_content = raw_content
if clean_content.startswith("```json"):
    clean_content = clean_content[7:]
if clean_content.startswith("```"):
    clean_content = clean_content[3:]
if clean_content.endswith("```"):
    clean_content = clean_content[:-3]
clean_content = clean_content.strip()

parsed_data = json.loads(clean_content)
if isinstance(parsed_data, list) and len(parsed_data) > 0 and isinstance(parsed_data[0], dict):
    parsed_data = parsed_data[0]

# 3. 使用 DataFormatChecker 进行初筛校验
is_valid, errors, warnings = DataFormatChecker.check_sample(parsed_data)

print("\n" + "=" * 60)
print("【2. DataFormatChecker L1 格式校验结果】")
print("=" * 60)
print(f"校验是否通过 (is_valid): {'[PASS] 通过' if is_valid else '[FAIL] 失败'}")
if errors:
    print(f"错误列表 (Errors): {errors}")
if warnings:
    print(f"警告提示 (Warnings): {warnings}")

# 保存单条测试结果
os.makedirs("test_samples_logs", exist_ok=True)
out_path = f"test_samples_logs/test_qwen_flash_response.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "latency_seconds": latency,
        "token_usage": response.usage.model_dump() if response.usage else None,
        "is_valid": is_valid,
        "data": parsed_data
    }, f, ensure_ascii=False, indent=2)

print(f"\n完整生成内容已保存至: {out_path}")
print(f"生成的对白总轮数: {len(parsed_data.get('messages', []))} 轮")
