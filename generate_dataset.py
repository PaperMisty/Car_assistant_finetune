"""
智能汽车客服助手 - 全场景数据合成与质检入库流水线 (Category 1~8 通用版)
- 特性 1: 1 Seed -> 5 种正交多样性变体 (80 Seeds -> 400 条 SFT 样本)
- 特性 2: 50 路超高异步并发 (asyncio.Semaphore(50))
- 特性 3: 双模型全双工真并发 (Qwen 与 Gemini 任务交替调度，同时满载并发)
- 特性 4: Qwen 模型池动态轮询 (Round-Robin 轮询 LLM_DEFAULT_MODEL_*，分散额度与 TPM 压力)
- 特性 5: 场景断点续写与幂等跳过 (已生成达标的场景自动跳过，支持随时恢复重试)
- 特性 6: L1 自动初筛 + 5% 抽检归档 + Token 保护交互确认 (yes/no)
"""

import asyncio
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import AsyncOpenAI

from config.prompt import SFTDataGeneratorPrompt
from utils.data_format_checker import DataFormatChecker
from utils.logger import logger

import argparse

load_dotenv(override=True)
# 1. 解析命令行参数
parser = argparse.ArgumentParser(description="智能汽车客服助手 - 多数据集通用生成流水线")
parser.add_argument(
    "--split",
    type=str,
    choices=["train", "validation", "val", "test", "final_test"],
    default="train",
    help="数据集划分: train (训练集 3.2k), validation (验证集 160条), test (压测测试集 400条)",
)
parser.add_argument("--variations", type=int, default=None, help="每个 Seed 生成的样本数 (默认 train=5, val=1, test=1)")
parser.add_argument("--concurrency", type=int, default=50, help="异步并发数 (默认 50)")
parser.add_argument(
    "-y",
    "--yes",
    action="store_true",
    help="自动确认所有交互式提示（跳过中途 yes/no 确认，全自动连续生成全部 8 大场景）",
)
args = parser.parse_args()

# 规范化 split 名称
SPLIT = (
    "test"
    if args.split in ("test", "final_test")
    else ("validation" if args.split in ("validation", "val") else "train")
)

# 目录与参数映射
if SPLIT == "test":
    SEEDS_DIR = "data/v2/seeds/final_test"
    OUTPUT_DIR = "data/v2/test"
    DEFAULT_VARIATIONS = 1
    PROMPT_MODE = "test"
elif SPLIT == "validation":
    SEEDS_DIR = "data/v2/seeds/validation"
    OUTPUT_DIR = "data/v2/validation"
    DEFAULT_VARIATIONS = 1
    PROMPT_MODE = "train"
else:
    SEEDS_DIR = "data/v2/seeds/train"
    OUTPUT_DIR = "data/v2/sft"
    DEFAULT_VARIATIONS = 5
    PROMPT_MODE = "train"

VARIATIONS_PER_SEED = args.variations if args.variations is not None else DEFAULT_VARIATIONS
CONCURRENCY_LIMIT = args.concurrency
MAX_RETRY_ROUNDS = 3

# 自动收集 .env 中的百炼模型池 (LLM_DEFAULT_MODEL_*)
QWEN_MODEL_POOL = [
    os.getenv(k).strip()
    for k in sorted(os.environ.keys())
    if k.startswith("LLM_DEFAULT_MODEL_") and os.getenv(k) and os.getenv(k).strip()
]
if not QWEN_MODEL_POOL:
    QWEN_MODEL_POOL = ["qwen3.8-flash"]

# 初始化异步客户端
qwen_client = AsyncOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_API_BASE"),
    timeout=60.0,
)

gemini_client = AsyncOpenAI(
    api_key=os.getenv("Gemini_API_KEY"),
    base_url=os.getenv("Gemini_BASE_URL"),
    timeout=60.0,
)
gemini_model = os.getenv("Gemini_MODEL_NAME", "gemini-3.7-flash-medium")

CATEGORY_MAP = {
    1: "用车与智能功能支持",
    2: "服务政策与权益咨询",
    3: "故障初判与技术答疑",
    4: "维保预约与进店接待",
    5: "维修进度与交车服务",
    6: "救援与保险理赔协同",
    7: "抱怨与升级投诉处理",
    8: "老客关怀与服务运营",
}


def load_category_seeds(cat_id: int) -> List[Dict[str, Any]]:
    """
    收集指定 Category 的全部 80 条种子
    """
    seeds = []
    if not os.path.exists(SEEDS_DIR):
        logger.error(f"种子目录不存在: {SEEDS_DIR}")
        return seeds

    # 匹配 category{cat_id}_ 开头的所有 jsonl 文件
    prefix = f"category{cat_id}_"
    matched_files = [
        os.path.join(SEEDS_DIR, f) for f in os.listdir(SEEDS_DIR) if f.startswith(prefix) and f.endswith(".jsonl")
    ]
    matched_files.sort()

    for fpath in matched_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seeds.append(json.loads(line))

    return seeds


def is_category_completed(cat_id: int, expected_count: int = 400) -> bool:
    """
    检查该 Category 是否已在本地完整生成
    """
    sft_file_path = os.path.join(OUTPUT_DIR, f"category{cat_id}_sft_dataset.jsonl")
    if not os.path.exists(sft_file_path):
        return False

    line_count = 0
    with open(sft_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                line_count += 1

    return line_count >= expected_count


async def generate_single_sample_async(
    semaphore: asyncio.Semaphore,
    seed_item: Dict[str, Any],
    variation_idx: int,
    model_type: str,
    qwen_model_name: str,
    attempt: int = 1,
) -> Dict[str, Any]:
    """
    单条样本异步生成任务（独立异常捕获与隔离）
    """
    base_seed_id = seed_item.get("seed_id", "UNKNOWN")
    task_id = f"{base_seed_id}_var{variation_idx}"
    if PROMPT_MODE in ("test", "stress"):
        var_meta = SFTDataGeneratorPrompt.TEST_STRESS_VARIATIONS[
            variation_idx % len(SFTDataGeneratorPrompt.TEST_STRESS_VARIATIONS)
        ]
    else:
        var_meta = SFTDataGeneratorPrompt.ORTHOGONAL_VARIATIONS[
            variation_idx % len(SFTDataGeneratorPrompt.ORTHOGONAL_VARIATIONS)
        ]

    client = qwen_client if model_type == "Qwen" else gemini_client
    model_name = qwen_model_name if model_type == "Qwen" else gemini_model

    payload = SFTDataGeneratorPrompt.get_generator_payload(seed_item, variation_idx=variation_idx, mode=PROMPT_MODE)
    kwargs = {
        "model": model_name,
        "messages": payload,
        "temperature": 0.75,
        "response_format": {"type": "json_object"},
    }
    if model_type == "Qwen":
        kwargs["extra_body"] = {"enable_thinking": False}

    async with semaphore:
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(**kwargs)
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

            parsed_data["seed_id"] = task_id
            parsed_data["base_seed_id"] = base_seed_id
            parsed_data["variation_name"] = var_meta["name"]
            parsed_data["generated_by_model"] = model_name

            is_valid, errors, warnings = DataFormatChecker.check_sample(parsed_data)
            tokens = resp.usage.completion_tokens if resp.usage else 0

            if is_valid:
                logger.info(f"[PASS] {task_id:24s} | 模型: {model_name:24s} | 耗时: {latency:.2f}s")
                return {
                    "status": "SUCCESS",
                    "task_id": task_id,
                    "seed_item": seed_item,
                    "variation_idx": variation_idx,
                    "model_type": model_type,
                    "model_name": model_name,
                    "latency": latency,
                    "tokens": tokens,
                    "is_valid": True,
                    "errors": [],
                    "warnings": warnings,
                    "data": parsed_data,
                }
            else:
                logger.warning(f"[FAIL-FORMAT] {task_id} | 格式未通过: {errors}")
                return {
                    "status": "FORMAT_ERROR",
                    "task_id": task_id,
                    "seed_item": seed_item,
                    "variation_idx": variation_idx,
                    "model_type": model_type,
                    "model_name": model_name,
                    "latency": latency,
                    "is_valid": False,
                    "errors": errors,
                    "data": None,
                }

        except Exception as e:
            latency = time.perf_counter() - t0
            logger.error(f"[ERROR-API] {task_id} | 模型: {model_name} | 异常: {e}")
            return {
                "status": "API_ERROR",
                "task_id": task_id,
                "seed_item": seed_item,
                "variation_idx": variation_idx,
                "model_type": model_type,
                "model_name": model_name,
                "latency": latency,
                "is_valid": False,
                "errors": [f"API异常: {str(e)}"],
                "data": None,
            }


async def process_category(cat_id: int, semaphore: asyncio.Semaphore) -> bool:
    """
    处理单个 Category 的完整生命周期
    """
    cat_name = CATEGORY_MAP.get(cat_id, f"Category {cat_id}")
    sft_file_path = os.path.join(OUTPUT_DIR, f"category{cat_id}_sft_dataset.jsonl")
    sample_5pct_path = os.path.join(OUTPUT_DIR, f"category{cat_id}_sft_samples_5pct.json")

    # 1. 检查是否已有缓存数据（断点续写支持）
    if is_category_completed(cat_id, expected_count=400):
        logger.info(f"⏩ 【Category {cat_id} - {cat_name}】已存在完整数据集（>= 400条），自动跳过！")
        return True

    seeds = load_category_seeds(cat_id)
    total_seeds = len(seeds)
    total_tasks = total_seeds * VARIATIONS_PER_SEED

    if total_seeds == 0:
        logger.warning(f"Category {cat_id} 未找到种子文件，跳过。")
        return True

    logger.info("=" * 70)
    logger.info(f"🚀 开始生成【Category {cat_id} - {cat_name}】")
    logger.info(f"1. 种子数: {total_seeds} 条 | 变体倍数: {VARIATIONS_PER_SEED} | 规划样本: {total_tasks} 条")
    logger.info(f"2. 异步并发数: {CONCURRENCY_LIMIT} 路 | 双模型全双工并行")
    logger.info(f"3. Qwen 模型轮询池 ({len(QWEN_MODEL_POOL)} 个): {QWEN_MODEL_POOL}")
    logger.info(f"4. Gemini 模型: {gemini_model}")
    logger.info("=" * 70)

    # 2. 构造交替真并行任务列表 (Qwen, Gemini, Qwen, Gemini...)
    raw_tasks = []
    for seed in seeds:
        for v_idx in range(VARIATIONS_PER_SEED):
            raw_tasks.append((seed, v_idx))

    # 交替分配模型与 Qwen 轮询模型名
    pending_tasks = []
    for idx, (seed, v_idx) in enumerate(raw_tasks):
        m_type = "Qwen" if (idx % 2 == 0) else "Gemini"
        # 轮询从 Qwen 池中选模型
        q_model = QWEN_MODEL_POOL[(idx // 2) % len(QWEN_MODEL_POOL)]
        pending_tasks.append((seed, v_idx, m_type, q_model))

    valid_results: Dict[str, Dict[str, Any]] = {}
    failed_history: List[Dict[str, Any]] = []

    start_total_time = time.perf_counter()

    # 3. 异步并发与自动重试循环
    for round_idx in range(1, MAX_RETRY_ROUNDS + 1):
        if not pending_tasks:
            break

        logger.info(
            f"\n>>> [Category {cat_id}] 开始第 {round_idx}/{MAX_RETRY_ROUNDS} 轮批处理 (待生成: {len(pending_tasks)} 条) <<<"
        )

        coroutines = [
            generate_single_sample_async(semaphore, seed, v_idx, m_type, q_model, attempt=round_idx)
            for seed, v_idx, m_type, q_model in pending_tasks
        ]

        batch_results = await asyncio.gather(*coroutines, return_exceptions=False)

        next_pending = []
        for idx, res in enumerate(batch_results):
            tid = res["task_id"]
            if res["is_valid"] and res["data"] is not None:
                valid_results[tid] = res["data"]
            else:
                failed_history.append(res)
                # 失败换对侧模型
                alt_model = "Gemini" if res["model_type"] == "Qwen" else "Qwen"
                q_model = QWEN_MODEL_POOL[idx % len(QWEN_MODEL_POOL)]
                next_pending.append((res["seed_item"], res["variation_idx"], alt_model, q_model))

        pending_tasks = next_pending
        if pending_tasks and round_idx < MAX_RETRY_ROUNDS:
            logger.warning(
                f"[Category {cat_id}] 第 {round_idx} 轮完成，有 {len(pending_tasks)} 条未通过，自动切换模型进入第 {round_idx+1} 轮重试..."
            )

    total_elapsed = time.perf_counter() - start_total_time

    # 4. 写入 SFT 数据集
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    valid_list = list(valid_results.values())
    with open(sft_file_path, "w", encoding="utf-8") as f:
        for item in valid_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 5. 抽取 5% 样本供质检审查
    random.seed(42)
    sample_size = max(1, int(len(valid_list) * 0.05))
    sampled_items = random.sample(valid_list, sample_size) if len(valid_list) >= sample_size else valid_list

    with open(sample_5pct_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "category_id": cat_id,
                "category_name": cat_name,
                "sample_rate": "5%",
                "sample_count": len(sampled_items),
                "total_valid_dataset": len(valid_list),
                "sampled_data": sampled_items,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 6. 报告汇总
    logger.info("\n" + "=" * 70)
    logger.info(f"【Category {cat_id} - {cat_name} 汇总报告】")
    logger.info("=" * 70)
    logger.info(f"1. 规划样本总数: {total_tasks} 条 ({len(seeds)} × {VARIATIONS_PER_SEED})")
    logger.info(f"2. 最终入库样本: {len(valid_list)} 条 (合格率: {len(valid_list)/total_tasks*100:.2f}%)")
    logger.info(f"3. 历史重试拦截: {len(failed_history)} 次")
    logger.info(f"4. 50路并发耗时 : {total_elapsed:.2f} 秒 (平均吞吐: {total_elapsed/total_tasks:.2f}s/条)")
    logger.info(f"5. SFT 数据集  : {sft_file_path}")
    logger.info(f"6. 5% 抽检文件 : {sample_5pct_path}")
    logger.info("=" * 70)

    return True


async def main():
    logger.info("=" * 70)
    logger.info("🌟 智能汽车客服助手 - 全场景 SFT 数据集生产引擎启动")
    logger.info(f"当前并发上限: {CONCURRENCY_LIMIT} 路 | 模式: 8 大场景循环 + 断点续写 + Token 交互保护")
    logger.info("=" * 70)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    for cat_id in range(1, 9):
        cat_name = CATEGORY_MAP[cat_id]

        # 处理当前场景
        success = await process_category(cat_id, semaphore)

        # 最后一个场景处理完后无需提示继续
        if cat_id == 8:
            logger.info("🎉 全部 8 大场景 SFT 数据集已全部构建完成！")
            break

        # 如果刚才该场景是刚生成的（不是秒过的），则进行交互式确认（若传入 -y 则自动跳过）
        if args.yes:
            logger.info(
                f"⚡ 已启用 [-y] 自动确认模式，直接进入下一个场景【Category {cat_id+1} - {CATEGORY_MAP[cat_id+1]}】..."
            )
        else:
            print("\n" + "-" * 70)
            user_choice = (
                input(
                    f"🔔 [Token 保护确认] 【Category {cat_id} - {cat_name}】已处理完毕。\n"
                    f"👉 是否继续生成下一个场景【Category {cat_id+1} - {CATEGORY_MAP[cat_id+1]}】？(yes/no) [默认 no]: "
                )
                .strip()
                .lower()
            )
            print("-" * 70 + "\n")

            if user_choice not in {"yes", "y"}:
                logger.info(
                    f"用户选择暂停（输入 '{user_choice}'），流水线已在 Category {cat_id} 处安全停止，保护 Token 消耗。"
                )
                break


if __name__ == "__main__":
    asyncio.run(main())
