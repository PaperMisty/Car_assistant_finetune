"""
验证 SFT 数据集 Loss 响应掩码机制 (Loss Masking Verifier)
底层原理验证：
1. 读取真实多轮对话样本，通过 Qwen3 官方 ChatML 模版渲染
2. 模拟 SFTTrainer(assistant_only_loss=True) 与 Unsloth(train_on_responses_only) 的真实 Token 掩码处理
3. 打印 input_ids 与 labels 对照表，直观展示哪些内容被打上 -100 忽略，哪些内容参与 Loss 计算
"""

import json
import os
from typing import List, Tuple
from transformers import AutoTokenizer


def apply_assistant_only_mask(
    input_ids: List[int],
    tokenizer: AutoTokenizer,
    response_prefix: str = "<|im_start|>assistant\n",
    end_token: str = "<|im_end|>"
) -> Tuple[List[int], List[int]]:
    """
    根据 ChatML 结构对非 Assistant 回复进行精确的 -100 掩码
    """
    prefix_ids = tokenizer.encode(response_prefix, add_special_tokens=False)
    end_ids = tokenizer.encode(end_token, add_special_tokens=False)
    end_token_id = end_ids[0] if end_ids else 151645

    labels = [-100] * len(input_ids)
    
    i = 0
    while i < len(input_ids):
        # 匹配 <|im_start|>assistant\n 起始位置
        if input_ids[i : i + len(prefix_ids)] == prefix_ids:
            # 找到回复开始位置（跳过 prefix 自身，也可包含）
            start_idx = i + len(prefix_ids)
            
            # 寻找对应的 <|im_end|> 结束位置
            end_idx = start_idx
            while end_idx < len(input_ids) and input_ids[end_idx] != end_token_id:
                end_idx += 1
            
            # 包含结束 token <|im_end|> 计算 Loss
            if end_idx < len(input_ids):
                end_idx += 1
            
            # 仅对 Assistant 生成区域赋予真实 token id (其余默认 -100)
            for k in range(start_idx, min(end_idx, len(input_ids))):
                labels[k] = input_ids[k]
                
            i = end_idx
        else:
            i += 1

    return input_ids, labels


def main():
    print("=" * 80)
    print("[SFT Loss Masking Verifier] 工具调用 (Tool Call & Response) 掩码机制抽查实测")
    print("=" * 80)

    # 1. 加载本地 Qwen3 分词器
    tok_path = "model/Qwen/models/Qwen--Qwen3-8B/snapshots/master"
    if not os.path.exists(tok_path):
        print(f"错误: 分词器路径不存在: {tok_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 扫描数据集中包含 tool / tool_calls 的样本
    target_sample = None
    target_category = None
    for cat_id in range(1, 9):
        sft_path = f"data/v2/sft/category{cat_id}_sft_dataset.jsonl"
        if not os.path.exists(sft_path):
            continue
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # 寻找包含 tool 角色或 tool_calls 的样本
                has_tool = any(m.get("role") == "tool" for m in item.get("messages", []))
                has_tool_call = any("tool_call" in m.get("content", "") or m.get("tool_calls") for m in item.get("messages", []))
                if has_tool or has_tool_call:
                    target_sample = item
                    target_category = f"Category {cat_id}"
                    break
        if target_sample:
            break

    if not target_sample:
        print("未找到包含 tool 调用的样本，读取默认第一条样本。")
        with open("data/v2/sft/category1_sft_dataset.jsonl", "r", encoding="utf-8") as f:
            target_sample = json.loads(f.readline())
        target_category = "Category 1"

    messages = target_sample["messages"]
    print(f"[抽检场景]    : {target_category}")
    print(f"[样本 Seed ID]: {target_sample.get('seed_id', 'sample')}")
    print(f"[多轮总条数]  : {len(messages)} 条\n")

    # 3. 渲染为标准 ChatML 文本并编码
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(formatted_text, add_special_tokens=False)

    # 4. 执行 Loss 响应掩码
    input_ids, labels = apply_assistant_only_mask(input_ids, tokenizer)

    # 5. 统计掩码指标
    total_tokens = len(input_ids)
    masked_tokens = sum(1 for x in labels if x == -100)
    trained_tokens = total_tokens - masked_tokens

    print("=" * 80)
    print(f"[Token 掩码统计结果]:")
    print(f"  - 总 Token 长度      : {total_tokens} tokens")
    print(f"  - 被掩码 Token 数    : {masked_tokens} tokens ({masked_tokens/total_tokens*100:.1f}%) -> 标记为 -100 (不计算 Loss)")
    print(f"  - 参与训练 Token 数  : {trained_tokens} tokens ({trained_tokens/total_tokens*100:.1f}%) -> 真实计算 Loss")
    print("=" * 80)

    # 6. 逐段对比打印 (完整展示每一轮角色及其掩码状态)
    print("\n[逐段 Token 状态映射明细 (重点关注 Tool 交互段落)]:")
    print("-" * 80)

    current_mode = None
    current_chunk_ids = []

    for t_id, l_id in zip(input_ids, labels):
        is_masked = (l_id == -100)
        mode = "[-] MASKED (-100 忽略)" if is_masked else "[+] TRAIN  (计算Loss)"

        if mode != current_mode:
            if current_chunk_ids:
                text_chunk = tokenizer.decode(current_chunk_ids, skip_special_tokens=False)
                clean_lines = [line.strip() for line in text_chunk.split("\n") if line.strip()]
                short_text = " \\n ".join(clean_lines)
                if len(short_text) > 100:
                    short_text = short_text[:50] + " ... " + short_text[-40:]
                print(f"[{current_mode:24s}] ({len(current_chunk_ids):3d} tokens) -> {short_text}")
            current_mode = mode
            current_chunk_ids = [t_id]
        else:
            current_chunk_ids.append(t_id)

    if current_chunk_ids:
        text_chunk = tokenizer.decode(current_chunk_ids, skip_special_tokens=False)
        clean_lines = [line.strip() for line in text_chunk.split("\n") if line.strip()]
        short_text = " \\n ".join(clean_lines)
        if len(short_text) > 100:
            short_text = short_text[:50] + " ... " + short_text[-40:]
        print(f"[{current_mode:24s}] ({len(current_chunk_ids):3d} tokens) -> {short_text}")

    print("-" * 80)
    print("[结论说明]:")
    print("1. Assistant 生成的 <tool_call> (意图识别与参数构造) -> 标记为 [+] TRAIN (参与计算 Loss)")
    print("2. 外部返回的 <tool_response> (环境客观数据/API响应)  -> 标记为 [-] MASKED (被掩码 -100，不计算 Loss)")
    print("3. Assistant 结合工具返回给客户的最终回答          -> 标记为 [+] TRAIN (参与计算 Loss)")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
