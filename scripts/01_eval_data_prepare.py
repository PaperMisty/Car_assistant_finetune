"""
01_eval_data_prepare.py - 智能汽车客服助手评估数据集准备与规则对齐脚本
功能：
1. 遍历 data/v2/validation 中的 8 大场景验证集数据 (category*_sft_dataset.jsonl)
2. 关联 data/v2/seeds/validation 中的质检规则库 (category*_validation.jsonl)
3. 提取 required_facts, required_questions, required_actions, prohibited_actions
4. 生成结构化评测集并保存至 custom_eval/car_assistant_eval.jsonl 及 EvalScope 格式
"""

import glob
import json
import os
import sys
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_seed_rules(seeds_dir: str) -> Dict[str, Dict[str, Any]]:
    """加载 validation seeds 质检规则元数据字典: seed_id -> seed_info"""
    seed_map = {}
    seed_files = glob.glob(os.path.join(seeds_dir, "category*_validation.jsonl"))  # glob支持正则匹配文件,返回列表

    if not seed_files:
        print(f"[Warning] 未在 {seeds_dir} 找到 category*_validation.jsonl 文件！")
        return seed_map

    for file_path in seed_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    seed_id = data.get("seed_id")
                    if seed_id:
                        seed_map[seed_id] = data
                except Exception as e:
                    print(f"[Error] 解析 Seed 文件 {file_path} 行 {line_idx} 失败: {e}")

    print(f"成功加载 Seed 规则数量: {len(seed_map)} 条 (来自 {len(seed_files)} 个场景分类)")
    return seed_map


def build_evaluation_dataset(
    validation_dir: str,
    seeds_dir: str,
    output_path: str,
    evalscope_qa_path: str,
    sample_limit: int = None,
) -> List[Dict[str, Any]]:
    """构建评测数据集并输出( 去种子数据找对应schema信息比如prohibit)"""
    seed_map = load_seed_rules(seeds_dir)

    val_files = sorted(glob.glob(os.path.join(validation_dir, "category*_sft_dataset.jsonl")))
    if not val_files:
        raise FileNotFoundError(f"未在 {validation_dir} 找到 category*_sft_dataset.jsonl 文件！")

    eval_items = []
    skipped_count = 0

    for file_path in val_files:
        cat_name = os.path.basename(file_path).split("_")[0]
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_sample = json.loads(line)
                except Exception as e:
                    print(f"[Error] 解析验证样本 {file_path} 行 {line_idx} 失败: {e}")
                    continue

                sample_seed_id = raw_sample.get("seed_id", "")
                base_seed_id = raw_sample.get("base_seed_id", "")

                # 优先用 base_seed_id 查找，找不到则用 sample_seed_id
                seed_info = seed_map.get(base_seed_id) or seed_map.get(sample_seed_id) or {}
                if not seed_info:
                    skipped_count += 1

                # 提取 messages 中的 system prompt 与首轮 user 提问
                messages = raw_sample.get("messages", [])
                system_prompt = ""
                user_query = ""
                reference_assistant_turn = ""

                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content")
                    if role == "system" and not system_prompt:
                        system_prompt = content or ""
                    elif role == "user" and not user_query:
                        user_query = content or ""
                    elif role == "assistant" and not reference_assistant_turn and content:
                        reference_assistant_turn = content

                if not user_query:
                    continue

                eval_item = {
                    "id": sample_seed_id or f"{cat_name}_{line_idx}",
                    "base_seed_id": base_seed_id,
                    "category_id": seed_info.get("category_id", ""),
                    "category": seed_info.get("category", raw_sample.get("scenario_type", "")),
                    "subcategory": seed_info.get("subcategory", ""),
                    "scenario": seed_info.get("scenario", ""),
                    "user_goal": seed_info.get("user_goal", ""),
                    "customer_role": seed_info.get("customer_role", "车主本人"),
                    "variation_name": raw_sample.get("variation_name", ""),
                    "system_prompt": system_prompt or "你是智能汽车官方客服助手。",
                    "query": user_query,
                    "full_dialog_history": messages,
                    "tools": raw_sample.get("tools", []),
                    # 关键业务质检规则
                    "required_facts": seed_info.get("required_facts", []),
                    "required_questions": seed_info.get("required_questions", []),
                    "required_actions": seed_info.get("required_actions", []),
                    "prohibited_actions": seed_info.get("prohibited_actions", []),
                    "tool_required": seed_info.get("tool_required", False),
                    "tool_name": seed_info.get("tool_name", ""),
                    "reference_response": reference_assistant_turn,
                }
                eval_items.append(eval_item)

                if sample_limit and len(eval_items) >= sample_limit:
                    break
        if sample_limit and len(eval_items) >= sample_limit:
            break

    print(f"共提取评测样本: {len(eval_items)} 条 (未匹配到 Seed 规则数: {skipped_count})")

    # 1. 保存完整的业务规则评测集 (JSONL)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ 业务全规则评测集已保存至: {output_path}")

    # 2. 保存 EvalScope 通用 QA 格式评测集
    os.makedirs(os.path.dirname(evalscope_qa_path), exist_ok=True)
    with open(evalscope_qa_path, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "query": item["query"],
                        "system": item["system_prompt"],
                        "response": item["reference_response"] or "dummy response",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"✅ EvalScope QA 标准评测集已保存至: {evalscope_qa_path}")

    return eval_items


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VAL_DIR = os.path.join(BASE_DIR, "data", "v2", "validation")
    SEEDS_DIR = os.path.join(BASE_DIR, "data", "v2", "seeds", "validation")

    OUT_FILE = os.path.join(BASE_DIR, "custom_eval", "car_assistant_eval.jsonl")
    EVALSCOPE_FILE = os.path.join(BASE_DIR, "custom_eval", "text", "qa", "car_eval.jsonl")

    print("=" * 70)
    print("STEP 1: 提取验证集与 Seed 质检规则并生成评测集")
    print("=" * 70)

    dataset = build_evaluation_dataset(
        validation_dir=VAL_DIR,
        seeds_dir=SEEDS_DIR,
        output_path=OUT_FILE,
        evalscope_qa_path=EVALSCOPE_FILE,
    )

    if dataset:
        print("\n【样本示例】:")
        print(
            json.dumps(
                {
                    "id": dataset[0]["id"],
                    "category": dataset[0]["category"],
                    "query": dataset[0]["query"],
                    "required_facts": dataset[0]["required_facts"],
                    "required_questions": dataset[0]["required_questions"],
                    "required_actions": dataset[0]["required_actions"],
                    "prohibited_actions": dataset[0]["prohibited_actions"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
