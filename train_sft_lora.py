"""
Qwen-8B 智能汽车客服助手 SFT LoRA 训练执行脚本 (单文件自包含版)
技术栈: Unsloth + PEFT (All-Linear LoRA) + TRL SFTTrainer
硬件支持: RTX 4090 (24G) / RTX 5090 (32G) (Linux / AutoDL)

特性清单:
1. 单文件自包含: 全部超参数直接集中在脚本头部，修改查看一目了然
2. 极致显存优化: Unsloth Triton 融合加速 + 梯度检查点，显存直降 70%
3. All-Linear LoRA: 对 q, k, v, o, gate, up, down 7 组投影矩阵全量施加
4. ChatML 契约保障: 原生适配 Qwen3 标准多轮、工具调用与 <think> 标签
5. 仅对回复计算 Loss (assistant_only_loss=True): 对提示词自动打 -100 掩码
6. 动态进度与 ETA 监控: 原生 tqdm + 自定义 TrainingProgressCallback
7. 早停保护 (EarlyStoppingCallback): 每 100 步评估 eval_loss，连续未改善自动早停
"""

import glob
import json
import os
import time
from datetime import timedelta
from typing import Any, Dict, List

from datasets import Dataset
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from utils.logger import logger
from dotenv import load_dotenv

load_dotenv(override=True)


# ==============================================================================
# 🛠️ 【全局训练超参数配置】(集中管理，一目了然)
# ==============================================================================
CONFIG = {
    # 1. 模型与分词器基座配置
    "model_name_or_path": os.getenv("MODEL_PATH", "model/Qwen/models/Qwen--Qwen3-8B/snapshots/master"),
    "max_seq_length": 2048,  # 最大序列长度 (P99 对话约 1700 Tokens，2048 足够且省显存)
    "dtype": "bfloat16",  # RTX 4090 / 5090 开启原生 bfloat16
    "load_in_4bit": False,  # False 为 BF16 LoRA 微调，True 为 4-bit QLoRA
    "trust_remote_code": True,
    # 2. PEFT / LoRA 适配器配置 (All-Linear)
    "lora_r": 16,  # LoRA 秩 (Rank)
    "lora_alpha": 32,  # LoRA 缩放因子 Alpha
    "lora_dropout": 0.0,  # Unsloth 推荐 0.0 以开启极致 Triton 算子融合加速
    "lora_bias": "none",
    "target_modules": [  # 7 个全线性层全量施加
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "use_gradient_checkpointing": "unsloth",
    # 3. 数据集路径
    "train_dir": "data/v2/sft",  # SFT 训练集 (8 大场景 3,200 条)
    "eval_dir": "data/v2/validation",  # 验证集 (8 大场景 160 条)
    # 4. 训练与批次超参 (Mini-Batch=2, Accum=8 -> 等效 Batch Size=16)
    "output_dir": "output/qwen_8b_lora_sft",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "optim": "adamw_torch",  # 可选 "adamw_8bit" 或 "adamw_torch"
    # 5. 评估、保存与早停 (每 100 步评估一次)
    "logging_steps": 10,
    "eval_steps": 100,
    "save_steps": 100,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "early_stopping_patience": 3,  # 连续 3 次 eval (300步) 未改善则自动早停
    "early_stopping_threshold": 0.001,
    # 6. Loss 掩码与监控
    "assistant_only_loss": True,  # 仅对 Assistant 计算损失，对 System/User/Tool 提示词自动掩码
    "report_to": "tensorboard",
    "seed": 42,
}


# ==============================================================================
# ⏱️ 【训练进度与 ETA 实时监控监听器】
# ==============================================================================
class TrainingProgressCallback(TrainerCallback):
    """
    实时计算每步耗时、样本吞吐，并在终端输出格式化彩色进度与预计剩余时间 (ETA)
    """

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.last_logged_step = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.perf_counter()
        logger.info("\n" + "=" * 80)
        logger.info("⏱️  训练进度监听器已激活 | 开始实时计算步耗时与剩余预计时间 (ETA)")
        logger.info("=" * 80)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Any] = None,
        **kwargs,
    ):
        if not logs or not self.start_time:
            return

        current_step = state.global_step
        max_steps = state.max_steps
        if current_step <= 0 or current_step == self.last_logged_step:
            return
        self.last_logged_step = current_step

        now = time.perf_counter()
        elapsed_total = now - self.start_time
        steps_done = current_step
        steps_remaining = max(0, max_steps - steps_done)

        avg_time_per_step = elapsed_total / steps_done
        eta_seconds = int(steps_remaining * avg_time_per_step)
        eta_str = str(timedelta(seconds=eta_seconds))
        elapsed_str = str(timedelta(seconds=int(elapsed_total)))

        train_loss = logs.get("loss", None)
        lr = logs.get("learning_rate", None)
        epoch = logs.get("epoch", 0.0)
        progress_pct = (steps_done / max_steps) * 100 if max_steps > 0 else 0.0

        log_parts = [
            f"[进度: {steps_done}/{max_steps} ({progress_pct:.1f}%)]",
            f"Epoch: {epoch:.2f}",
        ]
        if train_loss is not None:
            log_parts.append(f"Train Loss: {train_loss:.4f}")
        if lr is not None:
            log_parts.append(f"LR: {lr:.2e}")
        log_parts.extend(
            [
                f"已用: {elapsed_str}",
                f"预计剩余 (ETA): {eta_str}",
                f"速度: {1.0 / avg_time_per_step:.2f} step/s" if avg_time_per_step > 0 else "",
            ]
        )

        logger.info(" | ".join([p for p in log_parts if p]))

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, Any] = None,
        **kwargs,
    ):
        if metrics:
            eval_loss = metrics.get("eval_loss", "N/A")
            eval_loss_str = f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else str(eval_loss)
            logger.info("\n" + "-" * 80)
            logger.info(f"📊 [Step {state.global_step} 验证评估] 当前 Eval Loss: {eval_loss_str} (早停监控中)")
            logger.info("-" * 80 + "\n")


# ==============================================================================
# 📂 【数据加载与清洗函数】
# ==============================================================================
def sanitize_messages_for_chat_template(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    将 SFT 样本中可能存在的 tool / tool_calls 等结构做标准化清洗，防止 Jinja 模版解析异常
    """
    clean_msgs = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        if role == "tool":
            clean_msgs.append({"role": "user", "content": f"【系统工具返回数据】:\n{content}"})
        elif role == "assistant" and tool_calls:
            call_strs = [
                f"<tool_call>\n{json.dumps({'name': tc.get('function', {}).get('name'), 'arguments': tc.get('function', {}).get('arguments')}, ensure_ascii=False)}\n</tool_call>"
                for tc in tool_calls
            ]
            full_content = (content + "\n" + "\n".join(call_strs)).strip()
            clean_msgs.append({"role": "assistant", "content": full_content})
        else:
            clean_msgs.append({"role": role, "content": str(content)})
    return clean_msgs


def load_dataset_from_directory(dir_path: str) -> Dataset:
    """
    扫描目录下全部 .jsonl 数据文件，解析并清洗为包含 standard messages 列表的 HuggingFace Dataset
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"数据集目录不存在: {dir_path}")

    jsonl_files = sorted(glob.glob(os.path.join(dir_path, "*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"在目录 {dir_path} 下未找到任何 .jsonl 数据集文件！")

    records = []
    logger.info(f"正在从 [{dir_path}] 加载数据文件 (共 {len(jsonl_files)} 个):")
    for fpath in jsonl_files:
        fname = os.path.basename(fpath)
        count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                messages = item.get("messages", [])
                if messages:
                    clean_convo = sanitize_messages_for_chat_template(messages)
                    records.append({"messages": clean_convo})
                    count += 1
        logger.info(f"  |-- {fname:35s} : 载入并清洗 {count} 条多轮对话")

    logger.info(f"  └── 汇总: 成功载入 {len(records)} 条样本\n")
    return Dataset.from_list(records)


# ==============================================================================
# 🚀 【主训练执行流程】
# ==============================================================================
def main():
    logger.info("=" * 80)
    logger.info("🚀 智能汽车客服助手 Qwen-8B SFT LoRA 微调训练流水线启动")
    logger.info("=" * 80)

    # 1. 动态导入 Unsloth 和 TRL (导入顺序关键: Unsloth 必须在 TRL 之前导入以完成 monkey-patch)
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error(
            '未检测到 unsloth 依赖库！在云端 GPU 服务器运行前，请先安装: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
        )
        raise
    from trl import SFTConfig, SFTTrainer  # 必须在 unsloth 之后导入!

    # 2. 加载基座模型 (通过 Unsloth 享受 Triton 加速与显存优化)
    logger.info(f"【Step 1/5】正在加载基座模型: {CONFIG['model_name_or_path']}")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name_or_path"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=CONFIG["dtype"],
        load_in_4bit=CONFIG["load_in_4bit"],
        trust_remote_code=CONFIG["trust_remote_code"],
    )

    # 3. Tokenizer 独立加载 (绕开 Unsloth 对 eos_token 的篡改，保持原生干净状态)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name_or_path"],
        trust_remote_code=CONFIG["trust_remote_code"],
    )
    logger.info(
        f"【Step 2/5】Tokenizer 独立加载完成: eos='{tokenizer.eos_token}'(id={tokenizer.eos_token_id}), "
        f"pad='{tokenizer.pad_token}'(id={tokenizer.pad_token_id})"
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    # 4. 施加 PEFT / LoRA 适配器 (All-Linear 全量施加)
    logger.info(
        f"【Step 3/5】施加 LoRA 适配器: r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}, target_modules={CONFIG['target_modules']}"
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=CONFIG["target_modules"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias=CONFIG["lora_bias"],
        use_gradient_checkpointing=CONFIG["use_gradient_checkpointing"],
        random_state=CONFIG["seed"],
    )

    # 5. 加载训练集与验证集
    logger.info("【Step 4/5】载入并格式化训练集与验证集")
    train_raw_dataset = load_dataset_from_directory(CONFIG["train_dir"])
    eval_raw_dataset = load_dataset_from_directory(CONFIG["eval_dir"])

    num_cpu_cores = min(4, os.cpu_count() or 1)
    train_dataset = train_raw_dataset.map(formatting_prompts_func, batched=True, num_proc=num_cpu_cores)
    eval_dataset = eval_raw_dataset.map(formatting_prompts_func, batched=True, num_proc=num_cpu_cores)

    # 计算等效 Batch Size 与总 Steps
    effective_batch_size = CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
    total_steps_est = int((len(train_dataset) / effective_batch_size) * CONFIG["num_train_epochs"])
    logger.info(
        f"训练样本数: {len(train_dataset)} | 验证样本数: {len(eval_dataset)} | "
        f"等效 Batch Size: {effective_batch_size} (mini={CONFIG['per_device_train_batch_size']} × accum={CONFIG['gradient_accumulation_steps']}) | "
        f"预估总 Steps: ~{total_steps_est} 步"
    )

    # 6. 配置 SFTConfig
    logger.info("【Step 5/5】初始化 SFTConfig 与 SFTTrainer (挂载 EarlyStopping 与 ETA 监听器)")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    sft_config = SFTConfig(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_steps=max(10, int(total_steps_est * CONFIG.get("warmup_ratio", 0.05))),
        weight_decay=CONFIG["weight_decay"],
        optim=CONFIG["optim"],
        logging_strategy="steps",
        logging_steps=CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],
        load_best_model_at_end=CONFIG["load_best_model_at_end"],
        metric_for_best_model=CONFIG["metric_for_best_model"],
        greater_is_better=CONFIG["greater_is_better"],
        bf16=True if CONFIG["dtype"] == "bfloat16" else False,
        fp16=True if CONFIG["dtype"] == "float16" else False,
        seed=CONFIG["seed"],
        report_to=CONFIG["report_to"],
        disable_tqdm=False,
        dataset_text_field="text",
        max_length=CONFIG["max_seq_length"],
        dataset_num_proc=num_cpu_cores,
        packing=False,
        assistant_only_loss=CONFIG["assistant_only_loss"],
        eos_token="<|im_end|>",
    )

    # 注册回调器
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=CONFIG["early_stopping_patience"],
            early_stopping_threshold=CONFIG["early_stopping_threshold"],
        ),
        TrainingProgressCallback(),
    ]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # 7. 应用 Unsloth 专属的 Completion-only Loss 响应掩码（双重加固）
    if CONFIG["assistant_only_loss"]:
        try:
            from unsloth.chat_templates import train_on_responses_only

            logger.info(
                "【Loss 掩码】应用 Unsloth train_on_responses_only：对 System / User / Tool 提示词自动添加 -100 掩码"
            )
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
        except Exception as e:
            logger.warning(
                f"Unsloth train_on_responses_only 自动挂载提示 ({e})，将由 SFTConfig(assistant_only_loss=True) 自动执行掩码。"
            )

    # 8. 开始训练
    logger.info("=" * 80)
    logger.info("🔥 启动微调训练 (原生 tqdm 与自定义 ETA 监控器双重实时追踪)...")
    logger.info("=" * 80)

    t_start = time.time()
    train_result = trainer.train()
    elapsed_minutes = (time.time() - t_start) / 60.0

    logger.info("=" * 80)
    logger.info(f"🎉 训练顺利完成！总耗时: {elapsed_minutes:.2f} 分钟")
    logger.info(f"全局步数: {train_result.global_step} | 最终 Train Loss: {train_result.training_loss:.4f}")
    logger.info("=" * 80)

    # 9. 保存最终的最优 LoRA 适配器及分词器
    best_lora_dir = os.path.join(CONFIG["output_dir"], "best_lora")
    logger.info(f"正在导出最优 LoRA 权重与 Tokenizer 至: {best_lora_dir}")
    os.makedirs(best_lora_dir, exist_ok=True)

    model.save_pretrained(best_lora_dir)
    tokenizer.save_pretrained(best_lora_dir)

    logger.info(
        f"✅ 权重导出成功！云端可直接通过 FastLanguageModel.from_pretrained('{best_lora_dir}') 加载进行推理与压测。"
    )


if __name__ == "__main__":
    main()
