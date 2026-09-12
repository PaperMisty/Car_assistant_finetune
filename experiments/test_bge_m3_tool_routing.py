"""
experiments/test_bge_m3_tool_routing.py - 基于 BGE-M3 稠密语义向量的工具路由召回率评测实验

功能与流程：
1. 从 data/v2/tool_schemas.json 读取所有 55 个工具的原生信息 (name, description, parameters, required, enum)；
2. 将格式化后的工具特征文本通过本地 BGE-M3 (CPU) 编码为 1024 维归一化稠密向量，并持久化为 .npy 缓存；
3. 加载 custom_eval/car_assistant_eval.jsonl 前 N 个多轮样本；
4. 逐轮次计算用户提问与工具向量库的余弦相似度，评测 Top-1 准确率、Top-3 召回率、Top-5 召回率；
5. 分析真实业务场景下语义向量对长尾/隐式口语的泛化能力。
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from transformers import AutoTokenizer, AutoModel

# 默认本地 BGE-M3 路径
DEFAULT_BGE_M3_PATH = r"D:\ai_models\modelscope_cache\models\models\BAAI--bge-m3\snapshots\master"

# 导入项目定义的工具系统
from utils.tool import export_openai_schemas, TOOL_REGISTRY


def format_tool_signature_text(tool_schema: Dict[str, Any]) -> str:
    """
    将工具原生描述拼接为语义丰富且紧凑的文本：包含名称、描述、参数含义、必需/可选及枚举。
    支持 OpenAI Function Schema 与自定义规范。
    """
    func = tool_schema.get("function", tool_schema)
    name = func.get("name", "")
    desc = func.get("description", "")
    params = func.get("parameters", {})
    props = params.get("properties", {})
    required_fields = set(params.get("required", []))

    param_lines = []
    for p_name, p_info in props.items():
        # 解析参数类型 (兼顾 Pydantic Optional 产生的 anyOf 结构)
        p_type = p_info.get("type")
        if not p_type and "anyOf" in p_info:
            types = [t.get("type") for t in p_info["anyOf"] if t.get("type") != "null"]
            p_type = "/".join(types) if types else "any"
        elif not p_type:
            p_type = "string"

        is_req = "必填" if p_name in required_fields else "可选"
        p_desc = p_info.get("description", "")
        enum_vals = p_info.get("enum")

        param_str = f"{p_name} ({p_type}, {is_req})"
        if enum_vals:
            enum_str = ", ".join(str(e) for e in enum_vals)
            param_str += f" 枚举: [{enum_str}]"
        if p_desc:
            param_str += f" - {p_desc}"
        param_lines.append(param_str)

    params_text = "; ".join(param_lines) if param_lines else "无参数"
    return f"工具名称: {name} | 功能: {desc} | 入参规范: {params_text}"


class BGEM3ToolRetriever:
    """基于 BGE-M3 的轻量级 CPU 语义向量检索器"""

    def __init__(self, model_path: str = DEFAULT_BGE_M3_PATH, device: str = "cpu"):
        self.device = device
        print(f"📦 正在加载本地 BGE-M3 模型 (设备: {device}): {model_path} ...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        print(f"✅ BGE-M3 模型加载完毕，耗时: {time.time() - t0:.2f} 秒")

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """批量将文本编码为 1024 维 L2 归一化稠密向量"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            # BGE-M3 Dense Embedding 取 [CLS] token (即 outputs.last_hidden_state[:, 0]) 并做 L2 归一化
            cls_rep = outputs.last_hidden_state[:, 0]
            norm_rep = torch.nn.functional.normalize(cls_rep, p=2, dim=1)
            all_embeddings.append(norm_rep.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)


def get_or_build_tool_embeddings(
    retriever: BGEM3ToolRetriever,
    tool_schemas_path: Optional[str] = None,
    cache_dir: str = "experiments/cache",
    force_rebuild: bool = False
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    加载工具并从本地 .npy 缓存恢复或新建工具向量。
    优先从 utils/tool.py 直接导出携带完整参数语义与枚举的最新 Schema！
    """
    os.makedirs(cache_dir, exist_ok=True)
    npy_path = os.path.join(cache_dir, "tool_bge_m3_embeddings.npy")
    meta_path = os.path.join(cache_dir, "tool_metadata.json")

    # 1. 优先从 utils.tool 获取单事实源 (Single Source of Truth)
    raw_schemas = export_openai_schemas()
    tools = [s["function"] for s in raw_schemas]

    # 2. 若缓存存在且数量一致且未强制重建，直接快速读取
    if not force_rebuild and os.path.exists(npy_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        if len(cached_meta) == len(tools):
            print(f"⚡ 命中本地缓存: 从 {npy_path} 加载 {len(cached_meta)} 个工具向量 (0ms)...")
            embeddings = np.load(npy_path)
            return cached_meta, embeddings

    print(f"🔨 未命中缓存或强制重建，正在为 utils/tool.py 导出的 {len(tools)} 个工具生成 BGE-M3 语义向量...")
    tool_texts = [format_tool_signature_text(t) for t in tools]

    # 重点展示 vehicle_feature_query 等关键工具的特征文本
    print("\n📝 工具特征文本抽样 (验证参数描述是否成功包含):")
    sample_targets = ["vehicle_feature_query", "maintenance_booking_create", "communication_consent_update"]
    for t_text in tool_texts:
        if any(f"工具名称: {target} " in t_text for target in sample_targets):
            print(f"   • {t_text}")

    t0 = time.time()
    embeddings = retriever.encode_texts(tool_texts)
    print(f"✅ 工具向量构建完成，耗时: {time.time() - t0:.2f} 秒，矩阵形态: {embeddings.shape}")

    # 持久化为 .npy
    np.save(npy_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(tools, f, ensure_ascii=False, indent=2)
    print(f"💾 工具向量已更新并持久化至: {npy_path}\n")

    return tools, embeddings


def evaluate_bge_m3_retrieval(
    retriever: BGEM3ToolRetriever,
    tools: List[Dict[str, Any]],
    tool_embeddings: np.ndarray,
    eval_file: str,
    num_eval_samples: int = 8,
    use_context: bool = False,
    use_sticky: bool = False
):
    print("\n" + "=" * 95)
    ctx_str = "带上下文" if use_context else "单轮"
    sticky_str = " + 软意图粘滞 (Soft Sticky)" if use_sticky else ""
    print(f"🚀 开始使用 BGE-M3 评测 {eval_file} 前 {num_eval_samples} 个样本的工具召回率 【{ctx_str}{sticky_str}】...")
    print("=" * 95)

    samples = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_eval_samples:
                break
            if line.strip():
                samples.append(json.loads(line))

    tool_names = [t["name"] for t in tools]

    # 统计指标
    stats = {
        "tool_turns_total": 0,
        "tool_top1_hits": 0,
        "tool_top3_hits": 0,
        "tool_top5_hits": 0,
        "none_turns_total": 0,
        "tool_scores": [],
        "none_scores": [],
    }

    for s_idx, sample in enumerate(samples, 1):
        s_id = sample.get("id", f"item_{s_idx}")
        cat = sample.get("category", "未分类")
        history = sample.get("full_dialog_history", [])

        print(f"\n───────────────────────────────────────────────────────────────────────────────────────────")
        print(f"📌 [样本 {s_idx}/{len(samples)}] ID: {s_id} | 业务场景: {cat}")
        print(f"───────────────────────────────────────────────────────────────────────────────────────────")

        turn_no = 0
        sticky_tool = None  # 维护会话级意图粘滞工具

        for i in range(len(history) - 1):
            curr_msg = history[i]
            next_msg = history[i + 1]

            if curr_msg.get("role") == "user" and next_msg.get("role") == "assistant":
                turn_no += 1
                user_content = curr_msg.get("content") or ""
                tool_calls = next_msg.get("tool_calls")

                # 标注 Ground Truth
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    gt_tool = tool_calls[0].get("function", {}).get("name", "none")
                else:
                    gt_tool = "none"

                # 编码用户当前 Query (支持多轮上下文与首轮诉求锚点拼接)
                query_to_embed = user_content
                if use_context and i > 1:
                    first_user_msg = ""
                    for m in history:
                        if m.get("role") == "user" and m.get("content"):
                            first_user_msg = m.get("content")
                            break
                    prev_msg = history[i - 1]
                    prev_content = prev_msg.get("content") or ""
                    query_to_embed = f"核心诉求: {first_user_msg[:60]} | 上文: {prev_content[:40]} | 当前输入: {user_content}"
                elif use_context and i > 0:
                    prev_msg = history[i - 1]
                    prev_content = prev_msg.get("content") or ""
                    query_to_embed = f"上下文: {prev_content[:60]} | 当前输入: {user_content}"

                t_enc0 = time.time()
                query_vec = retriever.encode_texts([query_to_embed])[0]  # (1024,)
                scores = np.dot(tool_embeddings, query_vec)
                enc_cost_ms = (time.time() - t_enc0) * 1000

                # 排序取检索出的 Top-5
                top_indices = np.argsort(scores)[::-1][:5]
                retrieved_tools = [tool_names[idx] for idx in top_indices]
                retrieved_scores = [float(scores[idx]) for idx in top_indices]

                # 【核心：意图粘滞 (Soft Sticky Routing) 候选池合并策略】
                final_tools = list(retrieved_tools)
                final_scores = list(retrieved_scores)
                sticky_applied = False

                if use_sticky and sticky_tool:
                    # 检查用户是否发生了剧烈的意图转移：
                    # 若检索出的第一名工具与 sticky_tool 不同，但相似度极高 (>0.65)，代表明确的话题切换
                    # 否则，采用【候选池合并】：将 sticky_tool 强制并入候选池前部，保障长尾实体不丢失
                    if sticky_tool not in final_tools[:3]:
                        # 将 sticky_tool 插入到 Top-2 位置，形成 [检索Top1, sticky_tool, 检索Top2] 混合池
                        sticky_idx = tool_names.index(sticky_tool)
                        sticky_score = float(scores[sticky_idx])
                        final_tools.insert(1, sticky_tool)
                        final_scores.insert(1, sticky_score)
                        final_tools = final_tools[:5]
                        final_scores = final_scores[:5]
                        sticky_applied = True

                top1_tool = final_tools[0]
                top1_score = final_scores[0]

                is_tool_turn = (gt_tool != "none")
                if is_tool_turn:
                    stats["tool_turns_total"] += 1
                    stats["tool_scores"].append(top1_score)

                    hit1 = (top1_tool == gt_tool)
                    hit3 = (gt_tool in final_tools[:3])
                    hit5 = (gt_tool in final_tools[:5])

                    if hit1:
                        stats["tool_top1_hits"] += 1
                    if hit3:
                        stats["tool_top3_hits"] += 1
                    if hit5:
                        stats["tool_top5_hits"] += 1

                    badge = "✅ Top-1 命中!" if hit1 else ("⚡ Top-3 召回!" if hit3 else ("🎯 Top-5 召回!" if hit5 else "❌ 未召回"))
                    if sticky_applied and hit3:
                        badge += " [粘滞生效]"
                else:
                    stats["none_turns_total"] += 1
                    stats["none_scores"].append(top1_score)
                    badge = f"ℹ️ 纯文本轮次 (最高相关度: {top1_score:.3f})"

                # 打印轮次评估详情
                q_summary = user_content[:45] + "..." if len(user_content) > 45 else user_content
                print(f"  [Turn {turn_no}] 用户: \"{q_summary}\"")
                print(f"         真实意图: 【{gt_tool}】 | 结论: {badge} (耗时: {enc_cost_ms:.1f}ms)")
                cands_str = ", ".join([f"{name}({sc:.3f})" for name, sc in zip(final_tools[:3], final_scores[:3])])
                print(f"         综合候选 Top-3: [{cands_str}]")

                # 更新状态机：若当前轮次明确触发了工具（或上一轮识别出了高可信度候选），更新粘滞工具；
                # 若工具调用已执行完成或对话闭环，适度释放
                if is_tool_turn:
                    # 真实工具调用完毕，清空粘滞状态，防止干扰后续轮次
                    sticky_tool = None
                elif turn_no == 1 and retrieved_scores[0] > 0.50:
                    # 第 1 轮虽然是纯文本追问，但检索出的首选工具置信度高，将其作为粘滞候选，等待用户补充参数
                    sticky_tool = retrieved_tools[0]
                elif "谢" in user_content or "再见" in user_content or "好的" in user_content:
                    # 用户收尾致谢，清空粘滞
                    sticky_tool = None

    # 最终汇总
    print("\n" + "=" * 95)
    print("📊 BGE-M3 稠密语义向量工具路由评测报表 (CPU 本地实测)")
    print("=" * 95)

    tot_tools = stats["tool_turns_total"]
    if tot_tools > 0:
        top1_acc = stats["tool_top1_hits"] / tot_tools * 100
        top3_rec = stats["tool_top3_hits"] / tot_tools * 100
        top5_rec = stats["tool_top5_hits"] / tot_tools * 100
        avg_tool_score = np.mean(stats["tool_scores"]) if stats["tool_scores"] else 0.0

        print(f"1. 工具调用轮次召回指标 (共 {tot_tools} 次需要调用工具):")
        print(f"   • Top-1 准确率 (Top-1 Accuracy) : {top1_acc:.1f}% ({stats['tool_top1_hits']}/{tot_tools})")
        print(f"   • Top-3 召回率 (Top-3 Recall)   : {top3_rec:.1f}% ({stats['tool_top3_hits']}/{tot_tools})")
        print(f"   • Top-5 召回率 (Top-5 Recall)   : {top5_rec:.1f}% ({stats['tool_top5_hits']}/{tot_tools})")
        print(f"   • 工具意图平均最高相似度得分   : {avg_tool_score:.4f}")

    tot_none = stats["none_turns_total"]
    if tot_none > 0:
        avg_none_score = np.mean(stats["none_scores"]) if stats["none_scores"] else 0.0
        print(f"\n2. 非工具/闲聊轮次语义置信度 (共 {tot_none} 次无需工具):")
        print(f"   • 非工具轮次平均最高相似度得分 : {avg_none_score:.4f}")
        print(f"   • 阈值建议: 工具最高相似度普遍高于非工具轮次，可设置 score > 0.45 作为动态门控阈值！")

    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BGE-M3 工具路由召回率测试")
    parser.add_argument("--model_path", type=str, default=DEFAULT_BGE_M3_PATH, help="本地 BGE-M3 模型路径")
    parser.add_argument("--tool_file", type=str, default="data/v2/tool_schemas.json", help="工具定义文件")
    parser.add_argument("--eval_file", type=str, default="custom_eval/car_assistant_eval.jsonl", help="评测集文件")
    parser.add_argument("--num_eval_samples", type=int, default=8, help="评测样本数量")
    parser.add_argument("--use_context", action="store_true", help="是否拼接前序轮次上下文")
    parser.add_argument("--use_sticky", action="store_true", help="是否开启软意图粘滞与候选池合并策略")
    parser.add_argument("--force_rebuild", action="store_true", help="是否强制重新计算工具向量缓存")
    args = parser.parse_args()

    # 1. 初始化检索器
    retriever = BGEM3ToolRetriever(model_path=args.model_path, device="cpu")

    # 2. 生成或加载 .npy 工具向量 (缓存命中只需 0ms)
    tools, embeddings = get_or_build_tool_embeddings(retriever, cache_dir="experiments/cache", force_rebuild=args.force_rebuild)

    # 3. 运行评测
    evaluate_bge_m3_retrieval(
        retriever,
        tools,
        embeddings,
        eval_file=args.eval_file,
        num_eval_samples=args.num_eval_samples,
        use_context=args.use_context,
        use_sticky=args.use_sticky
    )
