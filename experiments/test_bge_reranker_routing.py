"""
experiments/test_bge_reranker_routing.py - 两阶段工具路由评测 (BGE-M3 粗排 Top-10 + BGE-Reranker 精排 Top-3)

两阶段设计范式：
1. 【粗排召回 Stage 1 - BGE-M3 (Bi-Encoder)】:
   从全量 54 个工具中，基于稠密向量点积极速初筛出相关度最高的 Top-10 候选工具（耗时 ~2ms）；
2. 【精排重排 Stage 2 - BGE-Reranker (Cross-Encoder)】:
   将用户 Query 与 Top-10 工具特征文本进行深度自注意力交互（Cross-Attention），输出意图置信度打分，重排选出最终 Top-3；
3. 【终极目标】:
   解决单纯依赖非对称语义检索时，强业务实体词（如“保养/里程/延保”）对投诉与纠纷意图的掩盖与干扰。
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from experiments.test_bge_m3_tool_routing import (
    BGEM3ToolRetriever,
    get_or_build_tool_embeddings,
    format_tool_signature_text,
    DEFAULT_BGE_M3_PATH
)

DEFAULT_RERANKER_PATH = r"D:\ai_models\modelscope_cache\models\BAAI--bge-reranker-base"


class BGEReranker:
    def __init__(self, model_path: str, device: str = "cpu"):
        print(f"📦 正在加载本地 BGE-Reranker 模型 (设备: {device}): {model_path} ...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device
        print(f"✅ BGE-Reranker 加载成功，耗时: {time.time() - t0:.2f} 秒\n")

    @torch.no_grad()
    def rerank(self, query: str, candidate_texts: List[str]) -> List[float]:
        """
        对 [query, candidate_text] 配对进行 Cross-Attention 深度打分
        返回 Sigmoid 归一化后的相关度得分 (0~1 之间)
        """
        pairs = [[query, text] for text in candidate_texts]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
        scores = torch.sigmoid(logits).cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores


def evaluate_two_stage_retrieval(
    retriever: BGEM3ToolRetriever,
    reranker: BGEReranker,
    tools: List[Dict[str, Any]],
    tool_embeddings: np.ndarray,
    eval_file: str = "custom_eval/car_assistant_eval.jsonl",
    num_samples: int = 64,
    top_k_coarse: int = 10
):
    tool_names = [t.get("function", t).get("name") for t in tools]
    tool_texts = [format_tool_signature_text(t) for t in tools]
    tool_name_to_text = {name: txt for name, txt in zip(tool_names, tool_texts)}

    with open(eval_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f][:num_samples]

    print("=" * 95)
    print(f"🚀 开始执行【两阶段工具路由全量评测】(粗排 Top-{top_k_coarse} -> BGE-Reranker 精排 Top-3)")
    print(f"评测样本数: {len(samples)} 个长对话 | 注册工具数: {len(tool_names)} 个")
    print("=" * 95)

    # 统计数据结构
    # stage1: 纯 BGE-M3 初筛
    # stage2: BGE-Reranker 重排后
    stats = {
        "tool_turns_total": 0,
        "stage1_top1": 0,
        "stage1_top3": 0,
        "stage1_top5": 0,
        "stage1_top10": 0,
        "stage2_top1": 0,
        "stage2_top3": 0,
        "stage2_top5": 0,
        "cat7_tool_turns": 0,
        "cat7_stage1_top3": 0,
        "cat7_stage2_top3": 0,
        "cat7_stage1_top5": 0,
        "cat7_stage2_top5": 0,
    }

    from collections import defaultdict
    cat_metrics = defaultdict(lambda: {"turns": 0, "s1_top3": 0, "s2_top3": 0, "s1_top5": 0, "s2_top5": 0})

    for s_idx, sample in enumerate(samples):
        cat_id = sample.get("category_id", (s_idx // 8) + 1)
        cat_name = sample.get("category", f"Category_{cat_id}")
        subcat = sample.get("subcategory", "")
        sid = sample.get("id", f"sample_{s_idx+1}")
        history = sample.get("full_dialog_history", [])

        print(f"\n───────────────────────────────────────────────────────────────────────────────────────────")
        print(f"📌 [样本 {s_idx+1}/{len(samples)}] ID: {sid} | 类别: {cat_name} - {subcat}")
        print(f"───────────────────────────────────────────────────────────────────────────────────────────")

        for i, turn in enumerate(history):
            if turn.get("role") == "user":
                user_content = turn.get("content", "").strip()
                turn_no = (i // 2) + 1

                # 判定本轮的期望调用工具
                gt_tool = "none"
                if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                    next_turn = history[i + 1]
                    if next_turn.get("tool_calls"):
                        gt_tool = next_turn["tool_calls"][0]["function"]["name"]

                is_tool_turn = (gt_tool != "none")
                if not is_tool_turn:
                    continue  # 重点聚焦评估工具调用轮次

                stats["tool_turns_total"] += 1
                cat_metrics[cat_name]["turns"] += 1
                if cat_id == 7:
                    stats["cat7_tool_turns"] += 1

                # 构造包含首轮诉求锚点的多轮上下文 Query
                if i > 1:
                    first_user_msg = ""
                    for m in history:
                        if m.get("role") == "user" and m.get("content"):
                            first_user_msg = m.get("content")
                            break
                    prev_msg = history[i - 1]
                    prev_content = prev_msg.get("content") or ""
                    query_to_embed = f"核心诉求: {first_user_msg[:60]} | 上文: {prev_content[:40]} | 当前输入: {user_content}"
                elif i > 0:
                    prev_msg = history[i - 1]
                    prev_content = prev_msg.get("content") or ""
                    query_to_embed = f"上下文: {prev_content[:60]} | 当前输入: {user_content}"
                else:
                    query_to_embed = user_content

                # ----------------------------------------------------
                # Stage 1: BGE-M3 向量粗排 (Top-10)
                # ----------------------------------------------------
                t0_s1 = time.time()
                query_vec = retriever.encode_texts([query_to_embed])[0]
                m3_scores = np.dot(tool_embeddings, query_vec)
                top10_indices = np.argsort(m3_scores)[::-1][:top_k_coarse]
                stage1_tools = [tool_names[idx] for idx in top10_indices]
                stage1_scores = [float(m3_scores[idx]) for idx in top10_indices]
                time_s1_ms = (time.time() - t0_s1) * 1000

                # 统计 Stage 1 召回指标
                s1_hit1 = (gt_tool == stage1_tools[0])
                s1_hit3 = (gt_tool in stage1_tools[:3])
                s1_hit5 = (gt_tool in stage1_tools[:5])
                s1_hit10 = (gt_tool in stage1_tools[:10])

                if s1_hit1: stats["stage1_top1"] += 1
                if s1_hit3:
                    stats["stage1_top3"] += 1
                    cat_metrics[cat_name]["s1_top3"] += 1
                    if cat_id == 7: stats["cat7_stage1_top3"] += 1
                if s1_hit5:
                    stats["stage1_top5"] += 1
                    cat_metrics[cat_name]["s1_top5"] += 1
                    if cat_id == 7: stats["cat7_stage1_top5"] += 1
                if s1_hit10: stats["stage1_top10"] += 1

                # ----------------------------------------------------
                # Stage 2: BGE-Reranker 跨注意力精排 (重排 Top-10 -> 选出 Top-3)
                # ----------------------------------------------------
                t0_s2 = time.time()
                cand_texts = [tool_name_to_text[name] for name in stage1_tools]
                # 同时将多轮上下文和当前输入传入 Reranker 交叉判定
                rerank_scores = reranker.rerank(query=query_to_embed, candidate_texts=cand_texts)
                rerank_order = np.argsort(rerank_scores)[::-1]
                stage2_tools = [stage1_tools[idx] for idx in rerank_order]
                stage2_scores = [float(rerank_scores[idx]) for idx in rerank_order]
                time_s2_ms = (time.time() - t0_s2) * 1000

                # 统计 Stage 2 召回指标
                s2_hit1 = (gt_tool == stage2_tools[0])
                s2_hit3 = (gt_tool in stage2_tools[:3])
                s2_hit5 = (gt_tool in stage2_tools[:5])

                if s2_hit1: stats["stage2_top1"] += 1
                if s2_hit3:
                    stats["stage2_top3"] += 1
                    cat_metrics[cat_name]["s2_top3"] += 1
                    if cat_id == 7: stats["cat7_stage2_top3"] += 1
                if s2_hit5:
                    stats["stage2_top5"] += 1
                    cat_metrics[cat_name]["s2_top5"] += 1
                    if cat_id == 7: stats["cat7_stage2_top5"] += 1

                # 打印对比详情
                q_disp = user_content[:50] + "..." if len(user_content) > 50 else user_content
                print(f"  [Turn {turn_no}] 用户: \"{q_disp}\"")
                print(f"         期望工具: 【{gt_tool}】")
                print(f"         Stage 1 (BGE-M3 粗排 Top-3):   {', '.join([f'{n}({s:.3f})' for n,s in zip(stage1_tools[:3], stage1_scores[:3])])} | 耗时: {time_s1_ms:.1f}ms")
                print(f"         Stage 2 (Reranker 精排 Top-3): {', '.join([f'{n}({s:.3f})' for n,s in zip(stage2_tools[:3], stage2_scores[:3])])} | 耗时: {time_s2_ms:.1f}ms")
                
                delta_str = "保持"
                if not s1_hit3 and s2_hit3:
                    delta_str = "🔥 Reranker 精排成功纠正/拉回 Top-3!"
                elif s1_hit3 and not s2_hit3:
                    delta_str = "⚠️ Reranker 出现负优化"
                elif s2_hit1:
                    delta_str = "✅ Reranker 锁定 Top-1 命中!"
                print(f"         判定结论: {delta_str}")

    # ==============================================================================
    # 最终对比汇总报表
    # ==============================================================================
    tot = stats["tool_turns_total"]
    print("\n" + "=" * 95)
    print("📊【两阶段路由 vs 单阶段向量检索】全量实测对比大盘")
    print("=" * 95)
    print(f"总评估工具调用轮次: {tot} 次\n")

    print(f"1. 综合召回指标对比:")
    print(f"   • Top-1 准确率: BGE-M3 初筛 {(stats['stage1_top1']/tot*100):.1f}%  ==>  Reranker 精排 {(stats['stage2_top1']/tot*100):.1f}%")
    print(f"   • Top-3 召回率: BGE-M3 初筛 {(stats['stage1_top3']/tot*100):.1f}%  ==>  Reranker 精排 {(stats['stage2_top3']/tot*100):.1f}%")
    print(f"   • Top-5 召回率: BGE-M3 初筛 {(stats['stage1_top5']/tot*100):.1f}%  ==>  Reranker 精排 {(stats['stage2_top5']/tot*100):.1f}%")
    print(f"   • Top-10 初筛覆盖率 (粗排漏检上限): {(stats['stage1_top10']/tot*100):.1f}% ({stats['stage1_top10']}/{tot})")

    c7_tot = stats["cat7_tool_turns"]
    if c7_tot > 0:
        print(f"\n2. Category 7 (投诉、争议与升级) 专项攻坚效果:")
        print(f"   • Cat 7 Top-3 召回率: BGE-M3 初筛 {(stats['cat7_stage1_top3']/c7_tot*100):.1f}%  ==>  Reranker 精排 {(stats['cat7_stage2_top3']/c7_tot*100):.1f}%")
        print(f"   • Cat 7 Top-5 召回率: BGE-M3 初筛 {(stats['cat7_stage1_top5']/c7_tot*100):.1f}%  ==>  Reranker 精排 {(stats['cat7_stage2_top5']/c7_tot*100):.1f}%")

    print(f"\n3. 各业务场景 Top-3 召回率矩阵:")
    for cat_name, m in cat_metrics.items():
        if m["turns"] > 0:
            s1_p = m["s1_top3"] / m["turns"] * 100
            s2_p = m["s2_top3"] / m["turns"] * 100
            delta = s2_p - s1_p
            delta_sign = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
            print(f"   • {cat_name:20s}: M3初筛 {s1_p:5.1f}%  ==>  Reranker {s2_p:5.1f}%  (变化: {delta_sign})")

    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BGE-Reranker 两阶段工具路由评测")
    parser.add_argument("--m3_path", type=str, default=DEFAULT_BGE_M3_PATH)
    parser.add_argument("--reranker_path", type=str, default=DEFAULT_RERANKER_PATH)
    parser.add_argument("--eval_file", type=str, default="custom_eval/car_assistant_eval.jsonl")
    parser.add_argument("--num_samples", type=int, default=64)
    args = parser.parse_args()

    # 1. 粗排 BGE-M3
    m3_retriever = BGEM3ToolRetriever(model_path=args.m3_path, device="cpu")
    tools, embeddings = get_or_build_tool_embeddings(m3_retriever, cache_dir="experiments/cache")

    # 2. 精排 BGE-Reranker
    reranker = BGEReranker(model_path=args.reranker_path, device="cpu")

    # 3. 执行两阶段评测
    evaluate_two_stage_retrieval(
        retriever=m3_retriever,
        reranker=reranker,
        tools=tools,
        tool_embeddings=embeddings,
        eval_file=args.eval_file,
        num_samples=args.num_samples,
        top_k_coarse=10
    )
