"""
Qwen-8B 智能汽车客服助手 SFT LoRA 训练执行脚本 (Unsloth + PEFT + SFTTrainer)
- 规范对齐: 严格按照 TRL 最新版 SFTConfig + SFTTrainer(processing_class=tokenizer) 标准接口设计
- 硬件支持: AutoDL Linux 云端环境 (RTX 4090 / RTX 5090)
- 特性 1: 基于 Unsloth 极致 Triton 融合加速与梯度检查点，显存直降 70%
- 特性 2: LoRA all-linear (q, k, v, o, gate, up, down) 7 组投影矩阵全量施加
- 特性 3: 多轮多角色 (System, User, Assistant, Tool) ChatML 标准数据流清洗与映射
- 特性 4: 挂载 tqdm 与自定义 TrainingProgressCallback，实时展示 Step/Loss/ETA 剩余时间
- 特性 5: 每 100 step 评估一次 eval_loss 并保存 Checkpoint，结合 EarlyStopping 自动早停
- 特性 6: 训练完毕自动将 eval_loss 最优的 LoRA 权重回载并导出
"""

import glob
import json
import os
import time
from datetime import timedelta
from typing import Any, Dict, List

from datasets import Dataset
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import SFTConfig, SFTTrainer

# 导入配置
from config.train_config import SFTTrainConfig, default_train_config
from utils.logger import logger


class TrainingProgressCallback(TrainerCallback):
    """
    训练进度与耗时/ETA 实时监控回调器
    在终端输出格式化彩色进度、当前 Loss、步耗时与剩余预计时间 (ETA)
    """

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.step_start_time = None
        self.last_logged_step = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.perf_counter()
        self.step_start_time = time.perf_counter()
        logger.info("\n" + "=" * 80)
        logger.info("⏱️  训练进度监听器已激活 | 开始实时计算吞吐与剩余时间 (ETA)")
        logger.info("=" * 80)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
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

        # 计算平均每步耗时与 ETA
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
        log_parts.extend([
            f"已用: {elapsed_str}",
            f"预计剩余 (ETA): {eta_str}",
            f"速度: {1.0 / avg_time_per_step:.2f} step/s" if avg_time_per_step > 0 else ""
        ])

        logger.info(" | ".join([p for p in log_parts if p]))

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, Any] = None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss", "N/A")
            eval_loss_str = f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else str(eval_loss)
            logger.info("\n" + "-" * 80)
            logger.info(f"📊 [Step {state.global_step} 验证评估] 当前 Eval Loss: {eval_loss_str} (早停指标监控中)")
            logger.info("-" * 80 + "\n")


def sanitize_messages_for_chat_template(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    数据清洗与兼容层：
    将 SFT 样本中可能存在的 tool / tool_calls 等非标结构转换为纯净的标准 ChatML 文本字典，
    防止 Jinja 模版在 Linux/云端 tokenizer.apply_chat_template 时抛出 TemplateError
    """
    clean_msgs = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        if role == "tool":
            # 将工具输出格式化为 user 维度的清晰文本
            clean_msgs.append({
                "role": "user",
                "content": f"【系统工具返回数据】:\n{content}"
            })
        elif role == "assistant" and tool_calls:
            # 将 tool_calls 序列化为 assistant 文本
            call_strs = [
                f"<tool_call>\n{json.dumps({'name': tc.get('function', {}).get('name'), 'arguments': tc.get('function', {}).get('arguments')}, ensure_ascii=False)}\n</tool_call>"
                for tc in tool_calls
            ]
            full_content = (content + "\n" + "\n".join(call_strs)).strip()
            clean_msgs.append({
                "role": "assistant",
                "content": full_content
            })
        else:
            clean_msgs.append({
                "role": role,
                "content": str(content)
            })
    return clean_msgs


def load_dataset_from_directory(dir_path: str) -> Dataset:
    """
    扫描目录下全部 category*.jsonl 文件，解析并清洗为包含 standard messages 列表的 HuggingFace Dataset
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


def main(config: SFTTrainConfig = default_train_config):
    logger.info("=" * 80)
    logger.info("🚀 智能汽车客服助手 Qwen-8B SFT LoRA 微调训练流水线启动 (AutoDL Linux 增强版)")
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

    # 3. 注入标准 ChatML 对话模版 (自适应兼顾 qwen-2.5 / chatml)
    logger.info("【Step 2/5】配置 Qwen ChatML 对话模板")
    try:
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=config.data.chat_template,
            mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
        )
    except Exception as e:
        logger.warning(f"指定模板 '{config.data.chat_template}' 无法直接加载 ({e})，正在自动回退到 'chatml' 官方标准模板...")
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="chatml",
            mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
        )

    def formatting_prompts_func(examples):
        """
        利用 tokenizer.apply_chat_template 将清洗后的多轮 messages 转换为带特殊 token 的连续训练文本
        """
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            for convo in convos
        ]
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

    num_cpu_cores = min(4, os.cpu_count() or 1)
    train_dataset = train_raw_dataset.map(formatting_prompts_func, batched=True, num_proc=num_cpu_cores)
    eval_dataset = eval_raw_dataset.map(formatting_prompts_func, batched=True, num_proc=num_cpu_cores)

    # 计算等效 Batch Size 与总 Steps
    effective_batch_size = config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
    total_steps_est = int((len(train_dataset) / effective_batch_size) * config.training.num_train_epochs)
    logger.info(
        f"训练样本数: {len(train_dataset)} | 验证样本数: {len(eval_dataset)} | "
        f"等效 Batch Size: {effective_batch_size} (mini={config.training.per_device_train_batch_size} × accum={config.training.gradient_accumulation_steps}) | "
        f"预估总 Steps: ~{total_steps_est} 步"
    )

    # 6. 配置标准 SFTConfig (参数全部收敛进 SFTConfig，0 飘红，显式开启 tqdm)
    logger.info("【Step 5/5】初始化 SFTConfig 与 SFTTrainer (挂载 EarlyStoppingCallback 与 TrainingProgressCallback)")

    os.makedirs(config.training.output_dir, exist_ok=True)
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
        disable_tqdm=False,  # 显式开启原生 tqdm 进度条
        # TRL 数据集与长文本超参规范
        dataset_text_field="text",
        max_length=config.model.max_seq_length,
        dataset_num_proc=num_cpu_cores,
        packing=False,  # 多轮对话保持独立样本，不进行 packing 截断
    )

    # 注册基于 eval_loss 的早停回调 + 实时进度与 ETA 监听器
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping.early_stopping_patience,
            early_stopping_threshold=config.early_stopping.early_stopping_threshold,
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

    # 7. 开始训练
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
