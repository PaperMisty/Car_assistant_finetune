# 智能汽车客服助手大模型微调工程 (Car Assistant LLM Finetune)

本项目是一个面向智能新能源汽车售后与用车服务场景的端到端大模型微调与数据工程系统。涵盖从**种子构建、正交多样性数据合成、L1/L2 双层质检、LoRA/QLoRA 高效微调到 DPO 偏好对齐**的完整全生命周期。

---

## 📌 当前项目里程碑与阶段成果 (Current Progress)

- [x] **数据工程阶段 (Data Engineering) - 已 100% 完成**
  - **8 大业务场景覆盖**：涵盖用车功能、政策权益、故障初判、维保预约、维修进度、救援保险、客诉处理与老客运营；
  - **3,200 条高质量 SFT 数据集入库**：基于 640 个业务种子（Seeds），引入 5 组正交多样性扰动矩阵扩增；
  - **50 路全双工异步并发流水线**：集成 Qwen 动态模型池轮询与 Local Gemini 负载对半分担，吞吐提升至 0.35s/条；
  - **L1 协议初筛与自动容错重试**：实现 `DataFormatChecker` 语法、角色交替与 Tool Calling 闭环校验；
  - **安全与机密扫描守护**：内置严格模式机密扫描器 `secret_scanner.py`（基于 `git ls-files`，绝不泄露 API Key）；
  - **全景统计与数据剖析**：生成多维度统计指标分析报告与综合可视化大屏图。
- [ ] **SFT 监督微调阶段 (Supervised Fine-Tuning) - 进行中**
  - 基于 **Qwen3-8B / Qwen2.5-7B** 基座，采用 **Unsloth + LoRA / QLoRA** 在 RTX 4090 / 5090 (24G/32G) 硬件上进行高效微调。
- [ ] **DPO 偏好对齐阶段 (Direct Preference Optimization)**
  - 构造包含拒识、安全红线与服务闭环的高质量偏好对。

---

## 📂 项目结构概览

```text
car_assitant/
├── config/                                # 核心配置与生成器 Prompt
│   ├── prompt.py                          # SFT 数据生成器 Prompt 体系 (含 5 组正交扰动)
│   └── data_format_checker.py             # L1 数据格式与 Tool Calling 协议校验器
├── data/
│   └── v2/
│       ├── seeds/train/                   # 8 大场景的原始业务种子 (640 条 Seeds)
│       └── sft/                           # 合成入库的 SFT 训练集与 5% 质检抽检数据
│           ├── category1_sft_dataset.jsonl ~ category8_sft_dataset.jsonl (共 3,200 条)
│           ├── category1_sft_samples_5pct.json ~ category8_sft_samples_5pct.json
│           └── sft_dataset_statistics.png # 全局多维度统计可视化大图
├── experiments/                           # 探索性实验与基准测试脚本 (已归档)
├── scripts/                               # 数据分析与辅助工具
│   └── analyze_sft_dataset.py             # 全局 SFT 数据集统计分析与图表渲染脚本
├── utils/                                 # 通用工具模块
│   ├── data_format_checker.py             # 数据校验器
│   ├── logger.py                          # 彩色结构化日志模块
│   └── secret_scanner.py                  # 严格模式 API Key 安全扫描器 (pre-commit hook)
├── generate_dataset.py                    # 50 路异步并发全场景 SFT 数据集生产引擎
└── 智能汽车客服助手_数据工程与质检规范.md     # 工业级数据工程理论、架构与质检标准文档
```

---

## 📊 SFT 数据集全景统计指标

基于 `scripts/analyze_sft_dataset.py` 对全量 **3,200 条多轮对话（31,709 句消息，约 258 万 Tokens）** 的统计分析：

| 评估维度 | 指标值 | 特征解读 |
| :--- | :---: | :--- |
| **System 提示词平均长** | **105.6 字** | 明确人设、权限边界与禁忌红线。 |
| **User 用户输入平均长** | **58.7 字** | 口语化自然真实，长短兼具，还原真实车主沟通习惯。 |
| **Assistant 回复平均长** | **148.8 字** | 排查分步清晰、单轮追问 $\le 2$ 个，自然克制，服务完全闭环。 |
| **Tool 返回平均长** | **208.8 字** | 包含完整的系统真实 JSON 响应（故障码、保单、工时等）。 |
| **多轮对话总轮数分布** | **均值 9.91 轮** | 呈现 7~13 轮钟形正态分布（主峰在 9 轮与 11 轮）。 |
| **工具调用覆盖率** | **75.53%** | 超过 3/4 的样本包含完整的 Tool Call & Tool Response 闭环。 |

---

## 🚀 快速上手与运行指南

### 1. 环境准备
本项目采用 `uv` 进行现代 Python 依赖管理：
```bash
# 激活虚拟环境并安装依赖
uv sync
```

### 2. 本地机密安全扫描 (Strict Secret Scanner)
在提交代码前，使用自定义严格模式扫描器拦截任何硬编码的 `sk-*` 敏感 Key：
```bash
python utils/secret_scanner.py
```

### 3. 数据集生成与增量续写
运行 50 路超高异步并发引擎（具备断点续写与场景幂等跳过）：
```bash
python generate_dataset.py
```

### 4. 数据集统计与图表分析
一键重新生成统计指标与 4 合 1 高清可视化仪表板：
```bash
python scripts/analyze_sft_dataset.py
```

---

## 🛠️ 下一步技术路线：Unsloth + LoRA 微调
计划在单张 **RTX 4090 (24G) / RTX 5090 (32G)** 上对 **Qwen3-8B** 执行微调：
* **框架选型**：`Unsloth` + `FlashAttention-2` + `HuggingFace TRL`
* **显存优化**：结合算子融合（Fused Kernels）与梯度检查点（Gradient Checkpointing），将 8B 模型多轮长文本（4096 序列）训练显存牢牢锁定在 18GB~22GB 以内。
