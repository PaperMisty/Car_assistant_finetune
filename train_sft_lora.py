"""
Qwen-8B 智能汽车客服助手 SFT LoRA 训练执行脚本 (Unsloth + PEFT + SFTTrainer)
- 规范对齐: 严格按照 TRL 最新版 SFTConfig + SFTTrainer(processing_class=tokenizer) 标准接口设计
- 硬件支持: RTX 4090 (24G) / RTX 5090 (32G)
- 特性 1: 基于 Unsloth 极致 Triton 融合加速与梯度检查点，显存直降 70%
- 特性 2: LoRA all-linear (q, k, v, o, gate, up, down) 7 组投影矩阵全量施加
- 特性 3: 多轮多角色 (System, User, Assistant, Tool) ChatML 标准数据流
- 特性 4: 每 100 step 评估一次 eval_loss 并保存 Checkpoint，结合 EarlyStopping 自动早停
- 特性 5: 训练完毕自动将 eval_loss 最优的 LoRA 权重回载并导出
"""

import glob
import json
import os
import time
from typing import Any, Dict, List

from datasets import Dataset
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

# 导入配置
from config.train_config import SFTTrainConfig, default_train_config
from utils.logger import logger


def load_dataset_from_directory(dir_path: str) -> Dataset:
    """
    扫描目录下全部 category*_sft_dataset.jsonl 或 category*_validation.jsonl 文件，
    解析为包含 standard messages 列表的 HuggingFace Dataset
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
                    records.append({"messages": messages})
                    count += 1
        logger.info(f"  |-- {fname:35s} : 载入 {count} 条多轮对话")

    logger.info(f"  └── 汇总: 成功载入 {len(records)} 条样本\n")
    return Dataset.from_list(records)


def main(config: SFTTrainConfig = default_train_config):
    logger.info("=" * 80)
    logger.info("🚀 智能汽车客服助手 Qwen-8B SFT LoRA 微调训练流水线启动")
    logger.info("=" * 80)

    # 1. 动态按需导入 Unsloth (在云端 GPU 环境中初始化)
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        logger.error(
            '未检测到 unsloth 依赖库！在云端 GPU 服务器运行前，请先安装: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
        )
        raise

    # 2. 加载基座模型与 Tokenizer
    logger.info(f"【Step 1/5】正在加载 Qwen-8B 基座模型: {config.model.model_name_or_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.model_name_or_path,
        max_seq_length=config.model.max_seq_length,
        dtype=config.model.dtype,
        load_in_4bit=config.model.load_in_4bit,
        trust_remote_code=config.model.trust_remote_code,
    )

    # 3. 注入标准 ChatML 对话模版 (Qwen 标准格式)
    logger.info("【Step 2/5】配置 Qwen ChatML 对话模板")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=config.data.chat_template,
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    def formatting_prompts_func(examples):
        """
        利用 tokenizer.apply_chat_template 将多轮 messages 转换为带特殊 token 的连续训练文本
        """
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    # 4. 施加 PEFT / LoRA 适配器 (All-Linear 全量施加)
    logger.info(
        f"【Step 3/5】施加 LoRA 适配器: r={config.lora.r}, alpha={config.lora.lora_alpha}, target_modules={config.lora.target_modules}"
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        target_modules=config.lora.target_modules,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
        random_state=config.lora.random_state,
    )

    # 5. 加载训练集与验证集
    logger.info("【Step 4/5】载入并格式化训练集与验证集")
    train_raw_dataset = load_dataset_from_directory(config.data.train_dir)
    eval_raw_dataset = load_dataset_from_directory(config.data.eval_dir)

    train_dataset = train_raw_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_raw_dataset.map(formatting_prompts_func, batched=True)

    # 计算等效 Batch Size 与总 Steps
    effective_batch_size = config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
    total_steps_est = int((len(train_dataset) / effective_batch_size) * config.training.num_train_epochs)
    logger.info(
        f"训练样本数: {len(train_dataset)} | 验证样本数: {len(eval_dataset)} | "
        f"等效 Batch Size: {effective_batch_size} (mini={config.training.per_device_train_batch_size} × accum={config.training.gradient_accumulation_steps}) | "
        f"预估总 Steps: ~{total_steps_est} 步"
    )

    # 6. 配置标准 SFTConfig (参数全部收敛进 SFTConfig，0 飘红)
    logger.info("【Step 5/5】初始化 SFTConfig 与 SFTTrainer (挂载 EarlyStoppingCallback)")

    sft_config = SFTConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        optim=config.training.optim,
        logging_strategy="steps",
        logging_steps=config.training.logging_steps,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        seed=config.training.seed,
        report_to=config.training.report_to,
        # TRL 数据集与长文本超参规范
        dataset_text_field="text",
        max_length=config.model.max_seq_length,
        dataset_num_proc=4,
        packing=False,  # 多轮对话保持独立样本，不进行 packing 截断
    )

    # 注册基于 eval_loss 的早停回调
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping.early_stopping_patience,
            early_stopping_threshold=config.early_stopping.early_stopping_threshold,
        )
    ]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # 7. 开始训练
    logger.info("=" * 80)
    logger.info("🔥 启动微调训练 (每 100 步自动评估 eval_loss 并保存 Checkpoint)...")
    logger.info("=" * 80)

    t_start = time.time()
    train_result = trainer.train()
    elapsed_minutes = (time.time() - t_start) / 60.0

    logger.info("=" * 80)
    logger.info(f"🎉 训练顺利完成！总耗时: {elapsed_minutes:.2f} 分钟")
    logger.info(f"全局步数: {train_result.global_step} | 最终 Train Loss: {train_result.training_loss:.4f}")
    logger.info("=" * 80)

    # 8. 保存最终的最优 LoRA 适配器及分词器
    best_lora_dir = os.path.join(config.training.output_dir, "best_lora")
    logger.info(f"正在导出最优 LoRA 权重与 Tokenizer 至: {best_lora_dir}")
    os.makedirs(best_lora_dir, exist_ok=True)

    model.save_pretrained(best_lora_dir)
    tokenizer.save_pretrained(best_lora_dir)

    logger.info(
        f"✅ 权重导出成功！云端可直接通过 FastLanguageModel.from_pretrained('{best_lora_dir}') 加载进行推理与压测。"
    )


if __name__ == "__main__":
    main()
