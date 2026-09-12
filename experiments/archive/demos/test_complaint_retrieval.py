import json
import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.test_bge_m3_tool_routing import BGEM3ToolRetriever, get_or_build_tool_embeddings, DEFAULT_BGE_M3_PATH

retriever = BGEM3ToolRetriever(model_path=DEFAULT_BGE_M3_PATH, device="cpu")
tools, embeddings = get_or_build_tool_embeddings(retriever, cache_dir="experiments/cache")
tool_names = [t["name"] for t in tools]

# 测试几条典型的用户投诉 Query
queries = [
    "车架号是 LSVAU2A35JN123456。首先把车检查一遍，其次必须对技师恶劣态度正式立案投诉与升级督办！",
    "车牌是浙A·D58219，我要投诉滨江服务中心无障碍通道被试驾车严重堵塞，要求正式记录投诉工单并整改！",
    "我车架号是 LSVAA123456789012，投诉顾问强行捆绑买延保否则不给保养，要求对这种欺诈行为立案调查！"
]

target_tool = "complaint_case_create"
t_idx = tool_names.index(target_tool)

print("=" * 80)
print(f"🎯 测试带有明确投诉意图 + VIN 的用户提问对 【{target_tool}】 的检索排名:")
print("=" * 80)

for q in queries:
    q_vec = retriever.encode_texts([q])[0]
    scores = np.dot(embeddings, q_vec)
    top_indices = np.argsort(scores)[::-1][:5]
    print(f"\nQuery: {q}")
    print(f"目标工具 {target_tool} 相似度: {scores[t_idx]:.4f}")
    print("Top-5 候选:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {tool_names[idx]} ({scores[idx]:.4f})")
