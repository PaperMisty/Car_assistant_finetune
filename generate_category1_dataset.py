"""
Category 1 全量数据合成与质检入库流水线 (5 路并发 + 双模型对半分担)
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import OpenAI

from config.prompt import SFTDataGeneratorPrompt
from config.data_format_checker import DataFormatChecker

# 加载配置
load_dotenv(override=True)

qwen_client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_API_BASE")
)
qwen_model = os.getenv("LLM_DEFAULT_MODEL_5", "qwen3.8-flash")

gemini_client = OpenAI(
    api_key=os.getenv("Gemini_API_KEY"),
    base_url=os.getenv("Gemini_BASE_URL")
)
gemini_model = os.getenv("Gemini_MODEL_NAME", "gemini-3.7-flash-medium")

# 1. 收集 Category 1 的全部种子 (80 条)
seed_dir = r"data/v2/seeds/train"
cat1_files = [
    os.path.join(seed_dir, "category1_user_features_001_010.jsonl"),
    os.path.join(seed_dir, "category1_user_features_011_020.jsonl"),
    os.path.join(seed_dir, "category1_expanded_021_080.jsonl"),
]

all_seeds = []
for fpath in cat1_files:
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_seeds.append(json.loads(line))

print(f"==================================================")
print(f"成功加载 Category 1 种子总数: {len(all_seeds)} 条")
print(f"调度策略: 5 路并发 | 对半分担 (前 {len(all_seeds)//2} 条 -> Qwen, 后 {len(all_seeds) - len(all_seeds)//2} 条 -> Gemini)")
print(f"==================================================\n")

# 2. 单条任务执行函数
def process_single_seed(seed_item: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    seed_id = seed_item.get("seed_id", "UNKNOWN")
    payload = SFTDataGeneratorPrompt.get_generator_payload(seed_item)
    
    client = qwen_client if model_type == "Qwen" else gemini_client
    model_name = qwen_model if model_type == "Qwen" else gemini_model
    
    kwargs = {
        "model": model_name,
        "messages": payload,
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }
    if model_type == "Qwen":
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
        tokens = resp.usage.completion_tokens if resp.usage else 0

        return {
            "status": "SUCCESS",
            "seed_id": seed_id,
            "model_type": model_type,
            "model_name": model_name,
            "latency": latency,
            "tokens": tokens,
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "data": parsed_data
        }
    except Exception as e:
        latency = time.perf_counter() - t0
        return {
            "status": "FAILED",
            "seed_id": seed_id,
            "model_type": model_type,
            "model_name": model_name,
            "latency": latency,
            "error_msg": str(e),
            "is_valid": False,
            "data": None
        }


# 3. 分配任务并以 5 路并发执行
half_idx = len(all_seeds) // 2
tasks = []
for idx, seed in enumerate(all_seeds):
    m_type = "Qwen" if idx < half_idx else "Gemini"
    tasks.append((seed, m_type))

output_dir = "data/v2/sft"
os.makedirs(output_dir, exist_ok=True)
sft_file_path = os.path.join(output_dir, "category1_sft_dataset.jsonl")
sample_5pct_path = os.path.join(output_dir, "category1_sft_samples_5pct.json")

all_results = []
valid_dataset = []
failed_records = []

start_total_time = time.perf_counter()

print(">>> 开始 5 路并发执行批量数据合成...")
with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_seed = {
        executor.submit(process_single_seed, seed, m_type): (seed.get("seed_id"), m_type)
        for seed, m_type in tasks
    }

    finished_count = 0
    total_tasks = len(tasks)

    for future in as_completed(future_to_seed):
        finished_count += 1
        res = future.result()
        all_results.append(res)
        
        sid = res.get("seed_id")
        m_type = res.get("model_type")
        status = res.get("status")
        valid = res.get("is_valid")
        lat = res.get("latency", 0)

        if status == "SUCCESS" and valid:
            valid_dataset.append(res["data"])
            print(f"[{finished_count:02d}/{total_tasks:02d}] [PASS] {sid} | 来自: {m_type:6s} | 耗时: {lat:.2f}s")
        else:
            failed_records.append(res)
            err = res.get("errors") or res.get("error_msg")
            print(f"[{finished_count:02d}/{total_tasks:02d}] [FAIL] {sid} | 来自: {m_type:6s} | 原因: {err}")

total_elapsed = time.perf_counter() - start_total_time

# 4. 格式合格数据入库写入 jsonl
with open(sft_file_path, "w", encoding="utf-8") as f:
    for item in valid_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 5. 随机抽取 5% (约 4 条) 质检样本
random.seed(42)
sample_size = max(1, int(len(valid_dataset) * 0.05))
sampled_items = random.sample(valid_dataset, sample_size) if len(valid_dataset) >= sample_size else valid_dataset

# 附带对应的元数据保存
with open(sample_5pct_path, "w", encoding="utf-8") as f:
    json.dump({
        "sample_rate": "5%",
        "sample_count": len(sampled_items),
        "total_valid_dataset": len(valid_dataset),
        "sampled_data": sampled_items
    }, f, ensure_ascii=False, indent=2)

print("\n==================================================")
print("【Category 1 数据工程与初筛完成汇总报告】")
print("==================================================")
print(f"1. 种子总数       : {len(all_seeds)} 条")
print(f"2. 成功入库样本   : {len(valid_dataset)} 条 (合格率: {len(valid_dataset)/len(all_seeds)*100:.2f}%)")
print(f"3. 失败/异常样本  : {len(failed_records)} 条")
print(f"4. 5路并发总耗时  : {total_elapsed:.2f} 秒 (平均单条: {total_elapsed/len(all_seeds):.2f}s)")
print(f"5. 全量SFT入库路径: {sft_file_path}")
print(f"6. 5%抽检文件路径 : {sample_5pct_path}")
print("==================================================")
