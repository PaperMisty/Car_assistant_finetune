# Unsloth + TRL `<EOS_TOKEN>` 报错排查记录

## 问题现象

训练 Qwen3-8B LoRA 微调时，`SFTTrainer.__init__` 抛出：

```
ValueError: The specified `eos_token` ('<EOS_TOKEN>') is not found in the vocabulary 
of the given `processing_class` (Qwen2Tokenizer).
```

> [!NOTE]
> 显示 `Qwen2Tokenizer` 是正常的 —— Qwen3 复用了 Qwen2 的分词器类，分词器架构未变。

---

## 根因分析

| 层级 | 原因 |
|:---|:---|
| **直接原因** | `SFTConfig` 的 `eos_token` 字段默认值是 `"<EOS_TOKEN>"`（TRL/Unsloth 内部占位符），不在 Qwen 词表中 |
| **核心原因** | **Unsloth 的 monkey-patch 机制要求它必须在 TRL 之前被 import**。如果 TRL 先导入，`SFTTrainer` 拿到的是未被 Unsloth 修补的原版，内部的 `<EOS_TOKEN>` 默认检查不会被正确覆盖 |
| **次要原因** | Unsloth 的 `FastLanguageModel.from_pretrained()` 会篡改 tokenizer 的 `eos_token` 属性为虚拟的 `"<EOS_TOKEN>"`，且通过 property descriptor 使得直接赋值 `tokenizer.eos_token = "..."` 可能无效 |

---

## 失败尝试记录

| # | 尝试方案 | 结果 | 失败原因 |
|:---|:---|:---:|:---|
| 1 | 在 `from_pretrained()` 之后立刻设置 `tokenizer.eos_token = "<\|im_end\|>"` | ❌ | `get_peft_model()` 内部再次覆盖 |
| 2 | 在 `SFTTrainer()` 前最后一刻设置 `tokenizer.eos_token` | ❌ | Unsloth 的 tokenizer 可能用 property 覆盖 getter |
| 3 | 在 `SFTConfig` 中传 `eos_token="<\|im_end\|>"` | ❌ | Unsloth monkey-patch 未生效时，该参数被忽略 |
| 4 | 用 `AutoTokenizer` 独立加载 tokenizer（绕开 Unsloth） | ❌ | 检查逻辑来自 SFTConfig 默认值，不是 tokenizer |
| 5 | **修复导入顺序：Unsloth 先于 TRL 导入** | ✅ | 根因解决 |

---

## 最终解决方案

```python
def main():
    # ① Unsloth 必须最先导入（注入 monkey-patch 到 trl 模块）
    from unsloth import FastLanguageModel

    # ② TRL 在 Unsloth 之后导入（拿到已修补版本的 SFTTrainer）
    from trl import SFTConfig, SFTTrainer

    # ③ Tokenizer 用标准 AutoTokenizer 独立加载（避免 Unsloth 篡改 eos_token）
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # tokenizer.eos_token == "<|im_end|>"  ✅ 原生干净

    # ④ SFTConfig 中显式指定 eos_token（覆盖默认的 "<EOS_TOKEN>" 占位符）
    sft_config = SFTConfig(
        ...,
        eos_token="<|im_end|>",
    )
```

> [!IMPORTANT]
> **关键约束**：`from unsloth import FastLanguageModel` 必须出现在 `from trl import SFTTrainer` 之前。
> 不能把 TRL 放在文件顶部的全局 import 区域，否则 Unsloth 的 monkey-patch 会失效。

---

## 附加问题：tensorboardX

训练配置 `report_to="tensorboard"` 需要安装 tensorboard 依赖：

```bash
uv add tensorboardX
```

如果不需要可视化，也可以改为 `report_to="none"`。

---

## Qwen3 Chat Template 备注

- Qwen3-8B 自带原生 `chat_template`（嵌入在 `tokenizer_config.json` 中），**不需要**额外加载 jinja 文件
- `original_chat_template.jinja` vs `new_chat_template.jinja` 的区别仅在于 `{% generation %}` 标签和空值保护，与 `<EOS_TOKEN>` 问题无关
- 项目中的 `new_chat_template.jinja` 增加了 TRL 的 `{% generation %}` 标记，可配合 `assistant_only_loss=True` 实现 Loss 掩码
