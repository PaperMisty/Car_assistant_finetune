import json
import numpy as np
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.test_bge_m3_tool_routing import BGEM3ToolRetriever, get_or_build_tool_embeddings, DEFAULT_BGE_M3_PATH

eval_file = "custom_eval/car_assistant_eval.jsonl"
cache_dir = "experiments/cache"

retriever = BGEM3ToolRetriever(model_path=DEFAULT_BGE_M3_PATH, device="cpu")
tools, embeddings = get_or_build_tool_embeddings(retriever, cache_dir=cache_dir)
tool_names = [t["name"] for t in tools]

samples = []
with open(eval_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= 25:
            break
        if line.strip():
            samples.append(json.loads(line))

print(f"总计读取 {len(samples)} 个样本进行失败轮次诊断...\n")

failures = []

for s_idx, sample in enumerate(samples, 1):
    s_id = sample.get("id")
    cat = sample.get("category")
    history = sample.get("full_dialog_history", [])
    sticky_tool = None

    for i in range(len(history) - 1):
        curr_msg = history[i]
        next_msg = history[i + 1]

        if curr_msg.get("role") == "user" and next_msg.get("role") == "assistant":
            user_content = curr_msg.get("content") or ""
            tool_calls = next_msg.get("tool_calls")
            gt_tool = "none"
            gt_args = None
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                gt_tool = tool_calls[0].get("function", {}).get("name", "none")
                gt_args = tool_calls[0].get("function", {}).get("arguments")

            # 构建带上下文的 Query
            prev_content = history[i - 1].get("content", "") if i > 0 else ""
            query_to_embed = f"上下文: {prev_content[:60]} | 当前提问: {user_content}" if i > 0 else user_content

            query_vec = retriever.encode_texts([query_to_embed])[0]
            scores = np.dot(embeddings, query_vec)
            top_indices = np.argsort(scores)[::-1][:10]
            retrieved_tools = [tool_names[idx] for idx in top_indices]
            retrieved_scores = [float(scores[idx]) for idx in top_indices]

            final_tools = list(retrieved_tools[:5])
            final_scores = list(retrieved_scores[:5])

            if sticky_tool and sticky_tool not in final_tools[:3]:
                sticky_idx = tool_names.index(sticky_tool)
                sticky_score = float(scores[sticky_idx])
                final_tools.insert(1, sticky_tool)
                final_scores.insert(1, sticky_score)
                final_tools = final_tools[:5]
                final_scores = final_scores[:5]

            # 仅诊断真需要调工具 (gt_tool != 'none') 的轮次
            if gt_tool != "none":
                hit3 = gt_tool in final_tools[:3]
                hit5 = gt_tool in final_tools[:5]
                rank = -1
                if gt_tool in retrieved_tools:
                    rank = retrieved_tools.index(gt_tool) + 1

                if not hit3:
                    failures.append({
                        "sample_idx": s_idx,
                        "sample_id": s_id,
                        "category": cat,
                        "turn": i // 2 + 1,
                        "prev_msg": prev_content,
                        "user_msg": user_content,
                        "gt_tool": gt_tool,
                        "gt_args": gt_args,
                        "rank": rank,
                        "top5_cands": list(zip(final_tools, final_scores)),
                        "hit5": hit5
                    })

            # 更新粘滞状态
            if gt_tool != "none":
                sticky_tool = None
            elif (i // 2 + 1) == 1 and retrieved_scores[0] > 0.50:
                sticky_tool = retrieved_tools[0]
            elif any(k in user_content for k in ["谢", "再见", "好的"]):
                sticky_tool = None

print("=" * 80)
print(f"🚨 诊断结果：共发现 {len(failures)} 个【Top-3 召回失败】的工具调用轮次：")
print("=" * 80)

for idx, f_item in enumerate(failures, 1):
    print(f"\n【失败案例 {idx}】")
    print(f"样本序号: #{f_item['sample_idx']} | 样本ID: {f_item['sample_id']}")
    print(f"业务场景: {f_item['category']} | 对话轮次: Turn {f_item['turn']}")
    if f_item['prev_msg']:
        print(f"前序助手说: \"{f_item['prev_msg']}\"")
    print(f"用户本轮说: \"{f_item['user_msg']}\"")
    print(f"真实工具 (GT): 【{f_item['gt_tool']}】 (参数: {f_item['gt_args']})")
    print(f"GT 在纯检索中的原始排名: 第 {f_item['rank']} 名" if f_item['rank'] > 0 else "GT 在前 10 名检索中完全未出现")
    print("召回出的 Top-5 候选:")
    for rank, (name, sc) in enumerate(f_item['top5_cands'], 1):
        flag = " 👈 (目标工具，但排在第4/5位)" if name == f_item['gt_tool'] else ""
        print(f"   {rank}. {name} (相似度: {sc:.3f}){flag}")
