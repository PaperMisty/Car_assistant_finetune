import os
from datetime import datetime

from evalscope import TaskConfig, run_task

# ============================================================
# 1. 基础配置
# ============================================================

SFT_MODEL = "./finetuned/02_sft_demo"
DPO_MODEL = "./finetuned/03_dpo_demo"

DATASET_PATH = "custom_eval/text/qa"
SUBSET = "ultrafeedback"


# 本次实验统一放到同一个目录
WORK_DIR = f"./outputs/arena_demo/"

# ============================================================
# 3. SFT + DPO 候选模型推理
# ============================================================

models = [
    {
        "model": SFT_MODEL,
        "model_id": "sft_demo",
    },
    {
        "model": DPO_MODEL,
        "model_id": "dpo_demo",
    },
]

task_list = [

    TaskConfig(

        # 本地模型
        model=item["model"],
        model_id=item["model_id"],

        # EvalScope 1.6.0 本地 checkpoint，固定写法
        eval_type="llm_ckpt",

        # 统一使用 general_qa
        datasets=[
            "general_qa"
        ],

        dataset_args={
            "general_qa": {

                "dataset_id": DATASET_PATH,

                "subset_list": [
                    SUBSET
                ],

                # Arena 第一阶段只需要生成回答
                # 不需要计算 BLEU / Rouge
                "metric_list": ["Rouge"]
            }
        },


        # 推理 Batch Size
        eval_batch_size=20,

        # 生成参数
        generation_config={
            "do_sample": False,
            "max_tokens": 2048,
        },

        # 两个模型必须写入同一个实验目录
        work_dir=WORK_DIR,

        # 不让 EvalScope 再自动添加时间戳
        no_timestamp=True,

        seed=42,
    )

    for item in models
]

# ============================================================
# 4. Candidate Inference
# ============================================================

print("=" * 70)
print("STEP 1：SFT + DPO Candidate Inference")
print("=" * 70)

run_task(
    task_cfg=task_list
)
