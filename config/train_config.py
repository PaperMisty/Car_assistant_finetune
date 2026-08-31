"""
Qwen-8B 智能汽车客服助手 LoRA 微调超参数配置文件 (Train Config)
包含：模型基座、PEFT/LoRA 适配器、数据路径、TrainingArguments 及 EarlyStopping 早停配置
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """模型与分词器基座配置"""

    # 云端服务器模型路径或 HuggingFace/ModelScope 标识
    # 例如本地/云端绝对路径
    model_name_or_path: str = os.getenv("MODEL_PATH", "model/Qwen/Qwen3-8B")
    max_seq_length: int = 2048  # 最大序列长度 (多轮对话长度P99为不到2000字,约1700token,2048足够)
    dtype: Optional[str] = "bfloat16"
    load_in_4bit: bool = False  # False 为全精度 BF16 微调，True 为 QLoRA 4-bit 量化
    trust_remote_code: bool = True


@dataclass
class LoraConfig:
    """PEFT / LoRA 适配器超参数配置"""

    r: int = 16  # LoRA 秩 (Rank)
    lora_alpha: int = 32  # LoRA 缩放因子 Alpha
    lora_dropout: float = 0.0  # Unsloth 推荐 0.0 开启极致 Triton 加速与显存优化
    bias: str = "none"  # 偏置微调策略
    # all-linear 模块全量施加
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    use_gradient_checkpointing: str = "unsloth"  # 启用 Unsloth 专属梯度检查点优化 (显存直降 70%)
    random_state: int = 42


@dataclass
class DataConfig:
    """训练集与验证集路径配置"""

    train_dir: str = "data/v2/sft"  # 训练集目录 (包含 category1~8 的 jsonl)
    eval_dir: str = "data/v2/validation"  # 验证集目录 (包含 category1~8 的 jsonl)
    chat_template: str = "qwen-3"  # 标准 ChatML 模板


@dataclass
class TrainingHyperparameters:
    """SFTTrainer 与 TrainingArguments 超参数"""

    output_dir: str = "output/qwen_8b_lora_sft"  # 权重与 Checkpoint 输出路径
    num_train_epochs: int = 3  # 训练 Epoch 数

    # 批次大小设置：mini_batch = 2, 累积步数 = 8 -> 等效每个 Step 处理 16 个样本
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8

    # 优化器与学习率调度
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optim: str = "adamw_torch"  # 推荐 8-bit AdamW，亦可使用 "adamw_torch"

    # 日志与评估策略 (每 100 步进行一次验证集评估与模型保存)
    logging_steps: int = 10
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3  # 仅保留最佳的 3 个 checkpoint

    # 早停与最佳模型回载
    load_best_model_at_end: bool = True  # 训练结束自动回载 eval_loss 最优的权重
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # 硬件混合精度加速
    bf16: bool = True  # RTX 4090 / 5090 开启原生 BF16
    fp16: bool = False
    seed: int = 42
    report_to: str = "tensorboard"  # 支持 "none", "tensorboard", "wandb"


@dataclass
class EarlyStoppingConfig:
    """早停回调函数配置 (基于 eval_loss 动态监控)"""

    early_stopping_patience: int = 3  # 连续 3 次评估 (即 300 steps) eval_loss 未降低则提前终止
    early_stopping_threshold: float = 0.001  # 判定位有效改善的最小阈值


@dataclass
class SFTTrainConfig:
    """统一聚合微调总配置"""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


# 默认配置实例
default_train_config = SFTTrainConfig()
