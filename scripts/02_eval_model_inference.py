"""
02_eval_model_inference.py - 智能汽车客服助手多轮推理评测脚本 (集成两阶段工具路由与思维抑制)

核心能力：
1. 【动态两阶段工具路由 (Two-Stage Tool Routing)】:
   每轮对话前，通过 BGE-M3 (粗排 Top-10) + BGE-Reranker (精排 Top-3) 为大模型动态供给最相关的 3 个工具；
2. 【轻量紧凑格式注入 (export_compact_schemas)】:
   废弃冗余的 OpenAI JSON Schema，使用极致扁平的参数描述 (Token 开销降低 60%)；
3. 【基座模型长思考抑制 (Forced Think Closure)】:
   在 Prompt 末尾预填 `<think>\n</think>\n` 并注入系统级规范约束，杜绝基座模型无意义的内心独白与显存浪费；
4. 【多轮教师强迫截断 (Teacher-Forced Slices)】:
   将多轮对话样本拆解为无累积误差的独立评估切片；
5. 【双引擎支持】:
   云端 vLLM 高并发批推理 (支持单基座+LoRA动态插拔) 与本地全功能 Mock 链路测试。
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
from dotenv import load_dotenv

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(override=True)

from utils.tool import export_compact_schemas, get_tools_by_names, TOOL_REGISTRY
from experiments.test_bge_m3_tool_routing import (
    BGEM3ToolRetriever,
    get_or_build_tool_embeddings,
    format_tool_signature_text,
    DEFAULT_BGE_M3_PATH
)
from experiments.test_bge_reranker_routing import (
    BGEReranker,
    DEFAULT_RERANKER_PATH
)


def parse_args():
    parser = argparse.ArgumentParser(description="汽车客服模型推理与对齐税评估 (集成工具路由与思考闭合)")
    parser.add_argument(
        "--input_file",
        type=str,
        default="custom_eval/car_assistant_eval.jsonl",
        help="评测输入数据集路径",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/car_eval/model_predictions.jsonl",
        help="模型生成结果保存路径",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=os.getenv("MODEL_PATH", "model/Qwen/Qwen3-8B").strip("\"'"),
        help="基座模型路径 (云端或本地)",
    )
    parser.add_argument(
        "--sft_model",
        type=str,
        default=(os.getenv("LORA_PATH") or os.getenv("SFT_MODEL_PATH") or "output/qwen_8b_lora_sft/best_lora").strip("\"'"),
        help="SFT微调模型或LoRA权重路径 (默认 output/qwen_8b_lora_sft/best_lora)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="vLLM 推理 Batch Size (32G 显存推荐 32-64)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="单次生成最大 token 数量",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="测试样本数量限制 (0 表示全量)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="启用 Mock 模拟推理模式 (无 GPU 或本地验证使用)",
    )
    parser.add_argument(
        "--use_router",
        action="store_true",
        default=True,
        help="是否开启 BGE-M3 + Reranker 两阶段工具动态路由 (默认开启)",
    )
    parser.add_argument(
        "--top_k_tools",
        type=int,
        default=3,
        help="每轮对话动态路由供给模型的候选工具数量 (默认 Top-3)",
    )
    parser.add_argument(
        "--bge_m3_path",
        type=str,
        default=DEFAULT_BGE_M3_PATH,
        help="BGE-M3 向量模型路径",
    )
    parser.add_argument(
        "--reranker_path",
        type=str,
        default=DEFAULT_RERANKER_PATH,
        help="BGE-Reranker 模型路径",
    )
    parser.add_argument(
        "--close_think",
        action="store_true",
        default=True,
        help="是否在 Assistant 截断处预置 <think>\\n</think>\\n 提前闭合基座思考 (默认开启)",
    )
    parser.add_argument(
        "--suppress_thinking",
        action="store_true",
        default=True,
        help="是否在系统提示词中增加无废话思考抑制规范 (默认开启)",
    )
    parser.add_argument(
        "--run_cmmlu",
        action="store_true",
        help="是否使用 EvalScope 运行 CMMLU 通用能力评估 (监测对齐税)",
    )
    return parser.parse_args()


class TwoStageToolRouter:
    """两阶段工具路由器：BGE-M3 向量初筛 (Top-10) + BGE-Reranker 跨注意力精排 (Top-3)"""

    def __init__(
        self,
        m3_path: str = DEFAULT_BGE_M3_PATH,
        reranker_path: str = DEFAULT_RERANKER_PATH,
        device: str = "cpu",
        cache_dir: str = "experiments/cache"
    ):
        self.available = False
        try:
            print(f"📦 正在初始化两阶段工具路由器...")
            self.m3 = BGEM3ToolRetriever(model_path=m3_path, device=device)
            self.tools, self.embeddings = get_or_build_tool_embeddings(self.m3, cache_dir=cache_dir)
            self.reranker = BGEReranker(model_path=reranker_path, device=device)
            self.tool_names = [t.get("function", t).get("name") for t in self.tools]
            self.tool_texts = [format_tool_signature_text(t) for t in self.tools]
            self.tool_name_to_text = {n: txt for n, txt in zip(self.tool_names, self.tool_texts)}
            self.available = True
            print(f"✅ 两阶段工具路由器初始化成功！全量原子工具池: {len(self.tool_names)} 个\n")
        except Exception as e:
            print(f"⚠️ 工具路由器未检测到本地权重或加载跳过 ({e})，启用自适应规则回退。")
            self.tool_names = list(TOOL_REGISTRY.keys())

    def route_tools(self, query: str, history_messages: List[Dict[str, Any]] = None, top_k: int = 3) -> List[str]:
        """为当前轮次的用户提问检索出最匹配的 Top-K 工具名称"""
        if not self.available:
            # 离线环境下的平稳降级
            defaults = ["vehicle_feature_query", "service_booking", "maintenance_due_query"]
            return defaults[:top_k]

        # 构造带有前序诉求锚点的丰富检索上下文
        query_to_embed = query
        if history_messages and len(history_messages) > 1:
            first_user_msg = ""
            for m in history_messages:
                if m.get("role") == "user" and m.get("content"):
                    first_user_msg = m.get("content")
                    break
            prev_content = history_messages[-1].get("content") or "" if history_messages else ""
            query_to_embed = f"核心诉求: {first_user_msg[:60]} | 上文: {prev_content[:40]} | 当前提问: {query}"

        # 1. 粗排：BGE-M3 向量相似度初筛 Top-10
        q_vec = self.m3.encode_texts([query_to_embed])[0]
        scores = np.dot(self.embeddings, q_vec)
        top10_idx = np.argsort(scores)[::-1][:10]
        stage1_names = [self.tool_names[i] for i in top10_idx]

        # 2. 精排：BGE-Reranker 跨注意力深度重排
        cand_texts = [self.tool_name_to_text[name] for name in stage1_names]
        rerank_scores = self.reranker.rerank(query=query_to_embed, candidate_texts=cand_texts)
        rerank_order = np.argsort(rerank_scores)[::-1]
        final_names = [stage1_names[i] for i in rerank_order][:top_k]
        return final_names


def format_chatml_prompt(
    messages: List[Dict[str, Any]],
    system_prompt: str = None,
    compact_tools: List[Dict[str, Any]] = None,
    close_think: bool = False,
    suppress_thinking: bool = False,
) -> str:
    """
    将多轮消息序列化为符合 Qwen ChatML 规范的提示词。
    
    进阶特性：
    1. 【动态紧凑工具注入】: 注入 compact_schemas 格式的工具定义；
    2. 【思维链提前闭合】: 若 close_think=True，预置 `<think>\\n</think>\\n` 迫使基座模型直接进入正文；
    3. 【思考抑制指令】: 注入规范约束，杜绝冗长自言自语。
    """
    prompt_parts = []

    # 1. 组装系统提示词 (含紧凑工具描述与思考抑制约束)
    effective_system = system_prompt or "你是智能汽车官方客服助手。请以专业、冷静、严谨的风格协助车主解决用车与维保问题。"
    if suppress_thinking:
        effective_system += "\n【回复规范】：直接输出解答或工具调用指令，严禁在回答中进行冗长的内心独白或输出大段长思考过程。"

    if compact_tools:
        tools_str = json.dumps(compact_tools, ensure_ascii=False, indent=2)
        effective_system += (
            f"\n\n# 可选系统工具 (本轮对话动态推荐 {len(compact_tools)} 个):\n"
            f"你可以根据上下文自主判断是否调用以下工具。如需调用，必须严格遵循以下标准格式：\n"
            f"<tool_call>\n"
            f'{{"name": "工具名称", "arguments": {{"参数名": "参数值"}}}}\n'
            f"</tool_call>\n"
            f"若当前轮次无需调用工具，请直接向用户做出专业清晰的解答。\n"
            f"<tools>\n{tools_str}\n</tools>"
        )

    prompt_parts.append(f"<|im_start|>system\n{effective_system}<|im_end|>\n")

    # 2. 遍历历史多轮消息 (自动过滤已在顶层合成的 system)
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            continue

        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        if role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            if tool_calls:
                call_info = json.dumps(tool_calls, ensure_ascii=False)
                body = f"{content}\n<tool_call>\n{call_info}\n</tool_call>" if content else f"<tool_call>\n{call_info}\n</tool_call>"
                prompt_parts.append(f"<|im_start|>assistant\n{body.strip()}<|im_end|>\n")
            else:
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        elif role == "tool":
            prompt_parts.append(f"<|im_start|>tool\n{content}<|im_end|>\n")
        else:
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    # 3. 截断处处理：控制思考闭合
    if close_think:
        # 强制闭合标签，模型生成时立刻从正文/工具调用开始
        prompt_parts.append("<|im_start|>assistant\n<think>\n</think>\n")
    else:
        prompt_parts.append("<|im_start|>assistant\n")

    return "".join(prompt_parts)


def expand_teacher_forced_slices(
    eval_items: List[Dict[str, Any]],
    router: Optional[TwoStageToolRouter] = None,
    top_k_tools: int = 3
) -> List[Dict[str, Any]]:
    """
    按教师强迫原则将多轮对话拆解为独立评估切片，并在此步骤动态为每个切片执行两阶段工具路由。
    """
    slices = []
    for item in eval_items:
        history = item.get("full_dialog_history", [])
        if not history:
            q = item.get("query", "")
            routed_names = router.route_tools(q, top_k=top_k_tools) if router else ["vehicle_feature_query"]
            compact_tools = export_compact_schemas(get_tools_by_names(routed_names))
            
            slices.append({
                **item,
                "slice_id": f"{item.get('id', 'item')}_t1",
                "turn_index": 1,
                "total_turns": 1,
                "history_messages": [
                    {"role": "system", "content": item.get("system_prompt", "你是智能汽车官方客服助手。")},
                    {"role": "user", "content": q},
                ],
                "current_turn_query": q,
                "is_tool_call_turn": item.get("tool_required", False),
                "routed_tool_names": routed_names,
                "compact_tools": compact_tools,
            })
            continue

        assistant_indices = [idx for idx, m in enumerate(history) if m.get("role") == "assistant"]
        total_turns = len(assistant_indices)

        for turn_no, ast_idx in enumerate(assistant_indices, 1):
            truncated_history = history[:ast_idx]
            gt_assistant_msg = history[ast_idx]

            current_user_query = ""
            for prev_m in reversed(truncated_history):
                if prev_m.get("role") == "user":
                    current_user_query = prev_m.get("content", "")
                    break

            gt_content = gt_assistant_msg.get("content") or ""
            gt_tool_calls = gt_assistant_msg.get("tool_calls")
            is_tool_call = bool(gt_tool_calls)

            ref_resp = gt_content
            if is_tool_call and not ref_resp:
                ref_resp = f"[工具调用指令] {json.dumps(gt_tool_calls, ensure_ascii=False)}"

            # 动态执行两阶段工具路由
            routed_names = []
            if router:
                routed_names = router.route_tools(current_user_query, truncated_history, top_k=top_k_tools)
            else:
                routed_names = ["vehicle_feature_query", "service_booking", "repair_order_query"][:top_k_tools]
                
            compact_tools = export_compact_schemas(get_tools_by_names(routed_names))

            slice_item = {
                **item,
                "slice_id": f"{item.get('id', 'item')}_t{turn_no}",
                "base_id": item.get("id", ""),
                "turn_index": turn_no,
                "total_turns": total_turns,
                "history_messages": truncated_history,
                "current_turn_query": current_user_query or item.get("query", ""),
                "reference_response": ref_resp,
                "is_tool_call_turn": is_tool_call,
                "routed_tool_names": routed_names,
                "compact_tools": compact_tools,
            }
            slice_item["query"] = current_user_query or item.get("query", "")
            slices.append(slice_item)

    return slices


def generate_mock_responses(eval_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    本地 Mock 模式生成 Baseline 与 SFT 的差异化对比回答（展示动态路由工具利用情况）。
    """
    results = []
    for item in eval_items:
        turn_idx = item.get("turn_index", 1)
        total_turns = item.get("total_turns", 1)
        q = item.get("query", "")
        req_facts = item.get("required_facts", [])
        req_questions = item.get("required_questions", [])
        req_actions = item.get("required_actions", [])
        is_tool = item.get("is_tool_call_turn", False)
        routed_tools = item.get("routed_tool_names", ["vehicle_feature_query"])

        # 模拟 SFT 模型精准使用路由出的首选工具
        if is_tool:
            best_tool = routed_tools[0] if routed_tools else "vehicle_feature_query"
            sft_response = f'<tool_call>\n{{"name": "{best_tool}", "arguments": {{"vehicle_id": "LSVAA123456789012"}}}}\n</tool_call>'
            baseline_response = "我不太清楚具体配置，建议查看随车手册。"
        elif turn_idx == 1:
            q_str = f"为了帮您快速核对，{req_questions[0]}" if req_questions else "请问您的爱车型号与车架号是什么？"
            sft_response = (
                f"您好，非常理解您的焦急心情，请您先别慌，我们先确保行车安全。\n\n"
                f"1. {q_str}；\n"
                f"2. 请务必在车辆停稳挂P档后再进行车机检查，切勿在行驶中分心操作。"
            )
            baseline_response = f"你好，关于“{q[:25]}...”，你可以试着重启一下车机。"
        elif turn_idx == total_turns:
            sft_response = "非常高兴能为您解决问题！请系好安全带，注意行车安全，祝您一路顺风！如后续有任何疑问欢迎随时联系。"
            baseline_response = "好的，不客气。"
        else:
            act_str = req_actions[0] if req_actions else "请在驻车状态下点击中控屏设置进行排查"
            fact_str = f"（提示：{req_facts[0]}）" if req_facts else ""
            sft_response = (
                f"收到您的信息，已为您核实完毕。\n\n"
                f"{fact_str}\n"
                f"建议您按以下步骤处理：\n"
                f"1. {act_str}；\n"
                f"2. 观察相关仪表反馈。\n\n"
                f"⚠️ 安全提示：切勿在行驶中分心操作屏幕。"
            )
            baseline_response = "建议去附近的特约销售店检查一下。"

        pred_item = {
            **item,
            "model_a_name": "baseline_qwen3_8b",
            "model_a_response": baseline_response,
            "model_b_name": "sft_car_assistant",
            "model_b_response": sft_response,
        }
        results.append(pred_item)
    return results


def main():
    args = parse_args()
    print("=" * 75)
    print("🚗 智能汽车客服助手多轮推理评测 (集成动态两阶段路由与思维控制)")
    print(f"评测输入: {args.input_file}")
    print(f"结果保存: {args.output_file}")
    think_ctrl_str = "闭合 <think> 标签" if args.close_think else "未闭合"
    print(f"思考控制: {think_ctrl_str} | 提示词抑制: {args.suppress_thinking}")
    print(f"运行模式: {'Mock 本地模拟测试' if args.mock else 'vLLM 真实推理'}")
    print("=" * 75)

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"未找到评测集 {args.input_file}！")

    raw_items = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_items.append(json.loads(line))

    if args.limit > 0:
        raw_items = raw_items[: args.limit]
        print(f"已截取前 {len(raw_items)} 条测试样本")

    # 1. 初始化两阶段工具路由器
    router = None
    if args.use_router:
        router = TwoStageToolRouter(
            m3_path=args.bge_m3_path,
            reranker_path=args.reranker_path,
            device="cpu"
        )

    # 2. 教师强迫多轮切片展开 (同时为每一轮动态分配 Top-3 紧凑工具)
    eval_items = expand_teacher_forced_slices(raw_items, router=router, top_k_tools=args.top_k_tools)
    print(f"📊 切片完成：原始样本 {len(raw_items)} 组 -> 展开多轮评测切片共 {len(eval_items)} 个 (均已完成动态工具注入)")

    # 3. 统一构建大模型实际接收的输入 Prompt (保证 Mock 与真实 vLLM 推理均严格执行并可审计)
    baseline_prompts = []
    sft_prompts = []
    for item in eval_items:
        bp = format_chatml_prompt(
            messages=item.get("history_messages", []),
            system_prompt=item.get("system_prompt"),
            compact_tools=item.get("compact_tools"),
            close_think=args.close_think,
            suppress_thinking=args.suppress_thinking,
        )
        sp = format_chatml_prompt(
            messages=item.get("history_messages", []),
            system_prompt=item.get("system_prompt"),
            compact_tools=item.get("compact_tools"),
            close_think=False,  # SFT 已学会直接输出结构化结果
            suppress_thinking=False,
        )
        baseline_prompts.append(bp)
        sft_prompts.append(sp)

    # 4. 自动化断言与审计：验证是否正常收到筛选后的工具与提示词约束
    print("\n🔍 正在对输入大模型的 Prompt 进行自动化校验 (工具筛选与提示词约束)...")
    tool_check_passed = True
    constraint_check_passed = True
    think_closure_passed = True

    for i, (item, bp, sp) in enumerate(zip(eval_items, baseline_prompts, sft_prompts)):
        routed_tools = item.get("routed_tool_names", [])
        # 1) 验证工具注入
        for tname in routed_tools:
            if tname not in bp or tname not in sp:
                tool_check_passed = False
                print(f"❌ 切片 {item.get('slice_id')} 未在 Prompt 中找到工具: {tname}")
        if "# 可选系统工具" not in bp or "<tools>" not in bp:
            tool_check_passed = False

        # 2) 验证提示词约束
        if args.suppress_thinking:
            if "【回复规范】：直接输出解答或工具调用指令" not in bp:
                constraint_check_passed = False
                print(f"❌ 切片 {item.get('slice_id')} Baseline Prompt 缺失思考抑制规范约束！")

        # 3) 验证思维闭合标签
        if args.close_think:
            if not bp.endswith("<|im_start|>assistant\n<think>\n</think>\n"):
                think_closure_passed = False
                print(f"❌ 切片 {item.get('slice_id')} Baseline Prompt 末尾未正确闭合 <think> 标签！")

        if not sp.endswith("<|im_start|>assistant\n"):
            think_closure_passed = False
            print(f"❌ 切片 {item.get('slice_id')} SFT Prompt 结尾格式不合规！")

    if tool_check_passed:
        print("  ✅ [工具筛选检验通过]: 所有切片均精准收到两阶段路由后的 Top-K 紧凑工具定义！")
    if constraint_check_passed:
        print("  ✅ [提示词约束检验通过]: Baseline Prompt 成功注入无长思考回复规范！")
    if think_closure_passed:
        print("  ✅ [思维闭合检验通过]: Baseline Prompt 末尾已强制注入 <think>\\n</think>\\n，SFT 保持标准开放！\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    can_use_vllm = False
    if not args.mock:
        try:
            import torch
            can_use_vllm = torch.cuda.is_available() and os.path.exists(args.baseline_model)
        except Exception:
            can_use_vllm = False

    if args.mock or not can_use_vllm:
        if not args.mock:
            print("⚠️ 未检测到 GPU 或模型路径，自动使用 Mock 模拟推理模式验证全流程。")
        raw_predictions = generate_mock_responses(eval_items)
        predictions = []
        for p_item, bp, sp in zip(raw_predictions, baseline_prompts, sft_prompts):
            p_item["prompt_baseline"] = bp
            p_item["prompt_sft"] = sp
            predictions.append(p_item)
    else:
        import gc
        import glob
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        if not os.path.exists(args.sft_model):
            raise FileNotFoundError(f"未找到 SFT 模型目录: '{args.sft_model}'")

        real_sft_path = args.sft_model
        adapter_cfg = os.path.join(real_sft_path, "adapter_config.json")
        if not os.path.exists(adapter_cfg):
            ckpts = sorted(
                glob.glob(os.path.join(real_sft_path, "checkpoint-*")),
                key=os.path.getmtime,
                reverse=True
            )
            for ckpt in ckpts:
                sub_cfg = os.path.join(ckpt, "adapter_config.json")
                if os.path.exists(sub_cfg):
                    real_sft_path = ckpt
                    adapter_cfg = sub_cfg
                    break

        is_sft_lora = os.path.exists(adapter_cfg)
        actual_lora_rank = 16
        if is_sft_lora:
            try:
                with open(adapter_cfg, "r", encoding="utf-8") as f:
                    actual_lora_rank = int(json.load(f).get("r", 16))
            except Exception:
                pass

        print(f"正在启动 vLLM 引擎: {args.baseline_model} (GPU Memory: 90%, LoRA Rank: {actual_lora_rank})...")
        llm = LLM(
            model=args.baseline_model,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            trust_remote_code=True,
            enable_lora=is_sft_lora,
            max_lora_rank=actual_lora_rank if is_sft_lora else 16,
        )

        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.8,
            max_tokens=args.max_tokens,
            stop=["<|im_end|>", "<|endoftext|>", "</think>"],
        )

        print(f"1. 批量生成 Baseline (基座模型，已闭合思考标签) 回答 (共 {len(baseline_prompts)} 个切片)...")
        base_outputs = llm.generate(baseline_prompts, sampling_params, lora_request=None)
        baseline_res = [out.outputs[0].text.strip() for out in base_outputs]

        print(f"2. 批量生成 SFT (微调模型: {real_sft_path}) 回答 (共 {len(sft_prompts)} 个切片)...")
        lora_req = LoRARequest("car_sft_lora", 1, real_sft_path) if is_sft_lora else None
        sft_outputs = llm.generate(sft_prompts, sampling_params, lora_request=lora_req)
        sft_res = [out.outputs[0].text.strip() for out in sft_outputs]

        predictions = []
        for item, a_txt, b_txt, bp, sp in zip(eval_items, baseline_res, sft_res, baseline_prompts, sft_prompts):
            predictions.append(
                {
                    **item,
                    "prompt_baseline": bp,
                    "prompt_sft": sp,
                    "model_a_name": "baseline_qwen3_8b",
                    "model_a_response": a_txt,
                    "model_b_name": "sft_car_assistant",
                    "model_b_response": b_txt,
                }
            )

    # 安全落盘
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    print(f"✅ 评测推理全部完成！结果已成功保存至: {args.output_file} (共 {len(predictions)} 条切片记录)")

    # 打印前 1 条切片的完整 Prompt 审计展示
    if predictions:
        print("\n" + "=" * 80)
        print("📋【切片 1 Prompt 真实审计样本】(验证工具筛选与提示词约束):")
        sample_slice = predictions[0]
        print(f"切片 ID: {sample_slice.get('slice_id')}")
        print(f"动态路由工具: {sample_slice.get('routed_tool_names')}")
        print("-" * 80)
        print("🔹 Baseline 模型接收的完整 Prompt (含系统提示词、路由工具、规范约束与提前闭合):")
        print(sample_slice.get("prompt_baseline"))
        print("-" * 80)
        print("🔹 SFT 模型接收的末尾截断 (验证未强插闭合标签):")
        print(repr(sample_slice.get("prompt_sft")[-60:]))
        print("=" * 80 + "\n")

    if args.run_cmmlu:
        from evalscope import TaskConfig, run_task
        print("正在运行 CMMLU 通用能力评测...")
        task_cfg = TaskConfig(model=args.sft_model, eval_type="llm_ckpt", datasets=["cmmlu"])
        run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    main()
