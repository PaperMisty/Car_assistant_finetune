"""
experiments/test_fasttext_tool_routing.py - FastText 工具路由召回率与多轮表现评测实验

目标：
1. 从 data/v2/sft 训练集中自动提取 (User输入 -> 工具标签/__label__none) 语料；
2. 使用 Jieba 分词后训练轻量级 FastText 监督分类器；
3. 加载 custom_eval/car_assistant_eval.jsonl 前 N 条样本的所有多轮轮次；
4. 评测 FastText 在单轮提问与多轮上下文下的 Top-1 准确率、Top-3 召回率与误触率。
"""

import os
import sys
import json
import glob
import argparse
import tempfile
from typing import List, Dict, Tuple, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import jieba
# 关闭 jieba 的调试日志输出
jieba.setLogLevel(jieba.logging.INFO)

# 针对 NumPy 2.0+ 兼容 FastText 库内部的 `np.array(probs, copy=False)` 问题进行热修补
import numpy as np
_orig_array = np.array
def _patched_array(obj, *args, **kwargs):
    if kwargs.get("copy") is False:
        kwargs.pop("copy")
        return np.asarray(obj, *args, **kwargs)
    return _orig_array(obj, *args, **kwargs)
np.array = _patched_array

try:
    import fasttext
except ImportError:
    raise ImportError("未检测到 fasttext 库，请先运行 `uv pip install fasttext-wheel`")


def tokenize_chinese(text: str) -> str:
    """对中文文本进行清洗和结巴分词，以空格连接供 FastText 处理"""
    if not text:
        return ""
    # 过滤换行与多余空白
    clean_text = text.replace("\n", " ").replace("\t", " ").strip()
    words = jieba.cut(clean_text)
    return " ".join([w for w in words if w.strip()])


def build_fasttext_training_data(sft_dir: str, max_samples_per_cat: int = 200, balance_none: bool = True) -> str:
    """
    从 SFT 数据集中提取轮次级分类训练数据：
    - balance_none: 是否平抑 __label__none 样本数量，防止严重类别不平衡导致模型只会预测 none
    """
    print(f"📦 正在从 {sft_dir} 提取 FastText 训练语料 (balance_none={balance_none})...")
    tool_lines = []
    none_lines = []
    files = glob.glob(os.path.join(sft_dir, "category*_sft_dataset.jsonl"))

    for fpath in files:
        count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                messages = data.get("messages", [])
                
                # 提取多轮问答对
                for idx in range(len(messages) - 1):
                    msg = messages[idx]
                    next_msg = messages[idx + 1]
                    
                    if msg.get("role") == "user" and next_msg.get("role") == "assistant":
                        user_text = msg.get("content") or ""
                        tool_calls = next_msg.get("tool_calls")
                        
                        seg_text = tokenize_chinese(user_text)
                        if not seg_text:
                            continue
                            
                        if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                            tool_name = tool_calls[0].get("function", {}).get("name", "unknown_tool")
                            tool_lines.append(f"__label__{tool_name} {seg_text}\n")
                        else:
                            none_lines.append(f"__label__none {seg_text}\n")
                            
                count += 1
                if max_samples_per_cat > 0 and count >= max_samples_per_cat:
                    break

    # 类别平衡：如果 none 太庞大，将其降采样至与 tool 相当的数量级
    if balance_none and len(none_lines) > len(tool_lines):
        import random
        random.seed(42)
        none_lines = random.sample(none_lines, len(tool_lines))

    lines = tool_lines + none_lines
    import random
    random.shuffle(lines)

    tmp_file = os.path.join(tempfile.gettempdir(), "fasttext_tool_train_balanced.txt")
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"✅ 训练语料构建完成: 共 {len(lines)} 条 (工具样本: {len(tool_lines)}, 平衡后非工具样本: {len(none_lines)})")
    return tmp_file


def train_fasttext_router(train_file: str):
    """训练 FastText 监督分类模型"""
    print("🚀 启动 FastText 模型训练 (epoch=25, lr=0.5, wordNgrams=2)...")
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        bucket=200000,
        dim=50,
        loss="softmax",
        verbose=0
    )
    print("✨ FastText 训练完成！耗时通常 < 1 秒")
    return model


def evaluate_on_eval_dataset(model, eval_file: str, num_eval_samples: int = 5):
    """
    加载 car_assistant_eval.jsonl 前 N 条样本的所有多轮轮次进行详细召回率评估
    """
    print("\n" + "=" * 90)
    print(f"🔍 开始在评测集 {eval_file} 前 {num_eval_samples} 个多轮样本上进行全轮次评估...")
    print("=" * 90)

    samples = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_eval_samples:
                break
            if line.strip():
                samples.append(json.loads(line))

    # 统计指标
    stats = {
        "tool_turns_total": 0,
        "tool_top1_correct": 0,
        "tool_top3_recall": 0,
        "none_turns_total": 0,
        "none_correct": 0,
        "none_false_alarm": 0,  # 本不需要工具却强行召回了工具
    }

    for sample_idx, sample in enumerate(samples, 1):
        sample_id = sample.get("id", f"sample_{sample_idx}")
        category = sample.get("category", "未分类")
        history = sample.get("full_dialog_history", [])
        
        print(f"\n──────────────────────────────────────────────────────────────────────────────────────────")
        print(f"📌 [样本 {sample_idx}/{len(samples)}] ID: {sample_id} | 业务分类: {category}")
        print(f"──────────────────────────────────────────────────────────────────────────────────────────")

        turn_no = 0
        for i in range(len(history) - 1):
            curr_msg = history[i]
            next_msg = history[i + 1]

            if curr_msg.get("role") == "user" and next_msg.get("role") == "assistant":
                turn_no += 1
                user_content = curr_msg.get("content") or ""
                tool_calls = next_msg.get("tool_calls")

                # 标注真实标签
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    gt_tool = tool_calls[0].get("function", {}).get("name", "unknown")
                else:
                    gt_tool = "none"

                # 分词与预测
                seg_query = tokenize_chinese(user_content)
                # 预测 Top-3 候选
                preds, probs = model.predict(seg_query, k=3)
                pred_labels = [p.replace("__label__", "") for p in preds]
                top1_pred = pred_labels[0]
                top1_prob = probs[0]

                # 统计判定
                is_tool_turn = (gt_tool != "none")
                if is_tool_turn:
                    stats["tool_turns_total"] += 1
                    hit_top1 = (top1_pred == gt_tool)
                    hit_top3 = (gt_tool in pred_labels)
                    if hit_top1:
                        stats["tool_top1_correct"] += 1
                    if hit_top3:
                        stats["tool_top3_recall"] += 1
                    result_badge = "✅ 命中 Top-1" if hit_top1 else ("⚠️ 命中 Top-3" if hit_top3 else "❌ 漏召回")
                else:
                    stats["none_turns_total"] += 1
                    is_none_hit = (top1_pred == "none")
                    if is_none_hit:
                        stats["none_correct"] += 1
                        result_badge = "✅ 正确识别无需工具"
                    else:
                        stats["none_false_alarm"] += 1
                        result_badge = f"⚠️ 误触工具 ({top1_pred})"

                # 打印单轮切片诊断
                print(f"  [Turn {turn_no}] 用户提问: \"{user_content[:45]}{'...' if len(user_content)>45 else ''}\"")
                print(f"         真实意图: 【{gt_tool}】 | 评测结论: {result_badge}")
                top_candidates_str = ", ".join([f"{name}({prob:.1%})" for name, prob in zip(pred_labels, probs)])
                print(f"         FastText 预测 Top-3: [{top_candidates_str}]")

    # 汇总报表
    print("\n" + "=" * 90)
    print("📊 FASTTEXT 工具路由测试总结报表")
    print("=" * 90)
    
    tool_total = stats["tool_turns_total"]
    tool_top1_acc = (stats["tool_top1_correct"] / tool_total * 100) if tool_total > 0 else 0
    tool_top3_rec = (stats["tool_top3_recall"] / tool_total * 100) if tool_total > 0 else 0
    
    none_total = stats["none_turns_total"]
    none_acc = (stats["none_correct"] / none_total * 100) if none_total > 0 else 0
    false_alarm_rate = (stats["none_false_alarm"] / none_total * 100) if none_total > 0 else 0

    print(f"1. 工具调用轮次 (Tool Turns):")
    print(f"   - 总工具轮次数: {tool_total}")
    print(f"   - Top-1 准确率 (Top-1 Accuracy) : {tool_top1_acc:.1f}% ({stats['tool_top1_correct']}/{tool_total})")
    print(f"   - Top-3 召回率 (Top-3 Recall)   : {tool_top3_rec:.1f}% ({stats['tool_top3_recall']}/{tool_total})")

    print(f"\n2. 无工具轮次 (None/闲聊/安抚/追问 Turns):")
    print(f"   - 总非工具轮次数: {none_total}")
    print(f"   - 正确识别率 (Correct None)     : {none_acc:.1f}% ({stats['none_correct']}/{none_total})")
    print(f"   - 工具误触率 (False Alarm Rate) : {false_alarm_rate:.1f}% ({stats['none_false_alarm']}/{none_total})")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastText 汽车客服工具路由召回率测试")
    parser.add_argument("--sft_dir", type=str, default="data/v2/sft", help="SFT 数据集目录")
    parser.add_argument("--eval_file", type=str, default="custom_eval/car_assistant_eval.jsonl", help="评测集文件")
    parser.add_argument("--num_eval_samples", type=int, default=6, help="评测样本数量")
    parser.add_argument("--max_train_samples", type=int, default=300, help="每个分类提取的训练样本上限")
    args = parser.parse_args()

    # 1. 提取语料
    train_file = build_fasttext_training_data(args.sft_dir, max_samples_per_cat=args.max_train_samples)
    
    # 2. 训练模型
    model = train_fasttext_router(train_file)
    
    # 3. 评测多轮召回率
    evaluate_on_eval_dataset(model, args.eval_file, num_eval_samples=args.num_eval_samples)
