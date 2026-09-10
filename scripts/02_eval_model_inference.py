"""
02_eval_model_inference.py - 模型推理与对齐税评估脚本
功能：
1. 加载准备好的评测集 (custom_eval/car_assistant_eval.jsonl)
2. 支持云端 vLLM 高并发批推理 (针对 32G 显存 8B 模型优化) 与本地 --mock 快速测试
3. 批量生成 Baseline 模型与 SFT 模型的回答
4. 支持 EvalScope CMMLU 通用能力评测 (对齐税监测)
5. 输出结果至 outputs/car_eval/model_predictions.jsonl
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(override=True)


def parse_args():
    parser = argparse.ArgumentParser(description="汽车客服模型推理与对齐税评估")
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
        default=os.getenv("MODEL_PATH", "model/Qwen/Qwen3-8B"),
        help="基座模型路径 (云端或本地)",
    )
    parser.add_argument(
        "--sft_model",
        type=str,
        default=os.getenv("SFT_MODEL_PATH", "output/qwen_8b_lora_sft"),
        help="SFT微调模型或LoRA权重路径",
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
        "--run_cmmlu",
        action="store_true",
        help="是否使用 EvalScope 运行 CMMLU 通用能力评估 (监测对齐税)",
    )
    return parser.parse_args()

def format_chatml_prompt(messages: List[Dict[str, Any]], system_prompt: str = None) -> str:
    """
    将标准多轮消息列表序列化为符合 Qwen ChatML 规范的提示词，并在末尾注入 `<|im_start|>assistant\\n` 等待生成。
    高保真支持 system, user, assistant (含 tool_calls), tool 消息类型。
    """
    prompt_parts = []

    # 若消息第一条不是 system，但提供了全局 system_prompt，则自动前置注入
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system and system_prompt:
        prompt_parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            if tool_calls:
                # 包含工具调用的 assistant 历史
                call_info = json.dumps(tool_calls, ensure_ascii=False)
                body = f"{content}\n<tool_call>\n{call_info}\n</tool_call>" if content else f"<tool_call>\n{call_info}\n</tool_call>"
                prompt_parts.append(f"<|im_start|>assistant\n{body.strip()}<|im_end|>\n")
            else:
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        elif role == "tool":
            # 真实工具返回给模型的外部数据 (教师强迫中作为已知环境反馈)
            name = msg.get("name", "function")
            prompt_parts.append(f"<|im_start|>tool\n{content}<|im_end|>\n")
        else:
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    # 截断在此处，提示模型进行续写
    prompt_parts.append("<|im_start|>assistant\n")
    return "".join(prompt_parts)


def expand_teacher_forced_slices(eval_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将包含 full_dialog_history 的多轮对话样本，按教师强迫原则拆解为多个独立的评估切片 (Turn Slices)。
    每个切片对应一个 assistant 生成时机，将该时机前的标准 Ground Truth 历史作为已知 Context，
    彻底解决多轮评估中'模型答偏导致后续语境崩塌'的问题。
    """
    slices = []
    for item in eval_items:
        history = item.get("full_dialog_history", [])
        if not history:
            # 没有多轮历史的退化为单轮切片
            slices.append({
                **item,
                "slice_id": f"{item.get('id', 'item')}_t1",
                "turn_index": 1,
                "total_turns": 1,
                "history_messages": [
                    {"role": "system", "content": item.get("system_prompt", "你是智能汽车官方客服助手。")},
                    {"role": "user", "content": item.get("query", "")},
                ],
                "current_turn_query": item.get("query", ""),
                "is_tool_call_turn": item.get("tool_required", False),
            })
            continue

        # 统计本样本中所有 assistant 发言点
        assistant_indices = [idx for idx, m in enumerate(history) if m.get("role") == "assistant"]
        total_turns = len(assistant_indices)

        for turn_no, ast_idx in enumerate(assistant_indices, 1):
            # 教师强迫截断：切取该 assistant 节点前的全部标准前序历史
            truncated_history = history[:ast_idx]
            gt_assistant_msg = history[ast_idx]

            # 提取该轮对应的最近一条用户提问 (供质检裁判展示当前轮次的核心意图)
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
            }
            # 兼容下游 Judge 读取 query 字段
            slice_item["query"] = current_user_query or item.get("query", "")
            slices.append(slice_item)

    return slices


def generate_mock_responses(eval_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    本地 Mock 模式生成 Baseline 与 SFT 的差异化对比回答。
    已完整适配多轮教师强迫切片：针对第1轮(安抚追问/调工具)、中间轮(核实指引)、收尾轮(致谢与安全提醒)分别生成贴合轮次的回答。
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

        # 模拟 SFT 模型的各轮次专业表现
        if is_tool:
            # 工具调用轮次
            tool_name = item.get("tool_name", "vehicle_feature_query")
            sft_response = f'<tool_call>\n{{"name": "{tool_name}", "arguments": {{"query": "{q[:20]}"}}}}\n</tool_call>'
            baseline_response = "抱歉，我不太清楚您车的具体配置，您可以看看说明书。"
        elif turn_idx == 1:
            # 第1轮：安抚情绪 + 追问车型配置 + 安全声明
            q_str = f"为了帮您快速核对，{req_questions[0]}" if req_questions else "请问您的爱车型号与年款是什么？"
            sft_response = (
                f"您好，非常理解您的焦急心情，请您先别慌，我们先确保行车安全。\n\n"
                f"1. {q_str}；\n"
                f"2. 请务必在车辆停稳挂P档后再进行车机检查，切勿在行驶中分心操作。"
            )
            baseline_response = f"你好，关于问题“{q[:25]}...”，建议尝试重启车机，或者去4S店检测。"
        elif turn_idx == total_turns:
            # 最后一轮：确认闭环 + 行车安全祝福
            sft_response = "非常高兴能为您解决问题！请系好安全带，注意前方路况，祝您一路顺风，用车愉快！如后续有任何疑问随时联系我们。"
            baseline_response = "好的，不客气。还有什么事吗？"
        else:
            # 中间轮：分步操作指导与故障排除
            act_str = req_actions[0] if req_actions else "请在驻车状态下点击中控屏设置进行检查"
            fact_str = f"（提示：{req_facts[0]}）" if req_facts else ""
            sft_response = (
                f"收到您的信息，已为您核实完毕。\n\n"
                f"{fact_str}\n"
                f"建议您按以下步骤处理：\n"
                f"1. {act_str}；\n"
                f"2. 观察相关指示灯与车机反馈。\n\n"
                f"⚠️ 安全提示：切勿在行驶中分心操作屏幕。"
            )
            baseline_response = "你可以试着在设置里点一下看看，不行的话联系售后师傅。"

        pred_item = {
            **item,
            "model_a_name": "baseline_qwen3_8b",
            "model_a_response": baseline_response,
            "model_b_name": "sft_car_assistant",
            "model_b_response": sft_response,
        }
        results.append(pred_item)
    return results


def run_vllm_inference(
    eval_items: List[Dict[str, Any]],
    model_path: str,
    lora_path: str = None,
    batch_size: int = 32,
    max_tokens: int = 1024,
) -> List[str]:
    """
    使用 vLLM 进行高效离线批处理推理（教师强迫分步截断多轮评估版本）。
    
    核心机制：
    1. 【教师强迫上下文注入 (Teacher-Forcing Context Injection)】：
       不使用单轮粗暴拼接，而是将每个切片点之前的 Ground Truth 标注历史（系统提示词、多轮用户发言、
       历史标准回复、真实工具查询结果）作为已知先验上下文高保真拼装进 ChatML 中；
    2. 【多轮截断与续写 (Stepwise Truncation)】：
       精确在待测的 `<|im_start|>assistant\\n` 处截断，让模型在该客观真实历史下生成本轮回答，
       彻底隔绝因前序轮次偏差导致的多轮语境崩塌；
    3. 【针对 32G 显存优化】：
       配置 gpu_memory_utilization=0.90，max_model_len=4096 (容纳完整多轮历史与工具上下文)，
       支持 LoRA 动态挂载与批并发生成。
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError:
        raise ImportError("未检测到 vLLM 库。如果在云端环境，请运行 `pip install vllm` 或使用 --mock 模式。")

    print(f"正在初始化 vLLM 引擎: {model_path} (GPU Memory: 90%, Context: 4096)...")
    
    # 自适应探测 LoRA 适配器的实际秩 (r)，动态配置 vLLM 预分配显存池
    actual_lora_rank = 16
    if lora_path:
        adapter_cfg_path = os.path.join(lora_path, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            try:
                with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                    actual_lora_rank = int(cfg_data.get("r", 16))
                print(f"🔍 成功检测并自适应加载 LoRA 适配器秩: r={actual_lora_rank} ({adapter_cfg_path})")
            except Exception as e:
                print(f"[Warning] 读取 {adapter_cfg_path} 失败: {e}，使用回退 rank={actual_lora_rank}")
        else:
            print(f"ℹ️ 未检测到 {adapter_cfg_path}，使用默认回退上限 rank={actual_lora_rank}")

    # 针对 32G 显存 (如 RTX 5090 / 4090) 并发优化
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
        enable_lora=bool(lora_path),
        max_lora_rank=actual_lora_rank if lora_path else 16,
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.8,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # 1. 采用教师强迫原则，为每个评估切片构建高保真多轮 ChatML 提示词
    prompts = []
    for item in eval_items:
        # 如果样本已带有教师强迫前序历史 (由 expand_teacher_forced_slices 生成)
        if "history_messages" in item:
            prompt = format_chatml_prompt(item["history_messages"], item.get("system_prompt"))
        elif "full_dialog_history" in item and item["full_dialog_history"]:
            # 若传入未切片的多轮样本，默认取截断至首轮 assistant 之前的上下文
            history = item["full_dialog_history"]
            ast_indices = [i for i, m in enumerate(history) if m.get("role") == "assistant"]
            cutoff = ast_indices[0] if ast_indices else len(history)
            prompt = format_chatml_prompt(history[:cutoff], item.get("system_prompt"))
        else:
            # 普通单轮退化兼容
            system_content = item.get("system_prompt", "你是智能汽车官方客服助手。")
            user_content = item.get("query", "")
            prompt = f"<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        
        prompts.append(prompt)

    lora_request = LoRARequest("car_sft_lora", 1, lora_path) if lora_path else None

    print(f"开始批量执行教师强迫多轮推理: 共 {len(prompts)} 个评测切片 (Batch Size: {batch_size})...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    generated_texts = [output.outputs[0].text.strip() for output in outputs]
    return generated_texts


def run_evalscope_cmmlu(model_path: str, output_dir: str = "./outputs/cmmlu_eval"):
    """使用 EvalScope 评测 CMMLU 中文通用能力 (对齐税评估)"""
    print("=" * 70)
    print("正在启动 EvalScope CMMLU 通用能力评测 (监测对齐税)...")
    print("=" * 70)
    try:
        from evalscope import TaskConfig, run_task

        task_cfg = TaskConfig(
            model=model_path,
            eval_type="llm_ckpt",
            datasets=["cmmlu"],
            dataset_args={
                "cmmlu": {
                    "subset_list": ["all"],
                    "few_shot_num": 5,
                }
            },
            eval_batch_size=16,
            work_dir=output_dir,
            no_timestamp=True,
        )
        run_task(task_cfg=task_cfg)
        print("✅ CMMLU 评测完成，报告已保存至:", output_dir)
    except Exception as e:
        print(f"[Warning] 运行 EvalScope CMMLU 失败: {e}")


def main():
    args = parse_args()
    print("=" * 70)
    print("STEP 2: 智能汽车客服助手多轮教师强迫截断推理与评估")
    print(f"输入数据: {args.input_file}")
    print(f"输出路径: {args.output_file}")
    print(f"基座模型: {args.baseline_model}")
    print(f"SFT 模型: {args.sft_model}")
    print(f"评测模式: {'教师强迫多轮分步截断 (Teacher-Forced Slices)' if not getattr(args, 'single_turn', False) else '仅首轮 (Single-Turn)'}")
    print(f"运行环境: {'Mock 快速测试' if args.mock else 'vLLM 真实推理'}")
    print("=" * 70)

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"未找到评测集 {args.input_file}，请先运行 01_eval_data_prepare.py！")

    raw_items = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_items.append(json.loads(line))

    if args.limit > 0:
        raw_items = raw_items[: args.limit]
        print(f"已截取前 {len(raw_items)} 条多轮对话种子")

    # 执行教师强迫多轮切片展开 (Teacher-Forced Truncation Slices)
    eval_items = expand_teacher_forced_slices(raw_items)
    print(f"📊 教师强迫切片完成：原始会话 {len(raw_items)} 组 -> 展开多轮评测切片共 {len(eval_items)} 个")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 判断是否启用 mock 或真实 vLLM
    can_use_vllm = False
    if not args.mock:
        try:
            import torch
            can_use_vllm = torch.cuda.is_available() and os.path.exists(args.baseline_model)
        except Exception:
            can_use_vllm = False

    if args.mock or not can_use_vllm:
        if not args.mock:
            print("⚠️ 未检测到 GPU 或模型权重路径不存在，自动切换至 Mock 模拟推理模式进行链路测试。")
        predictions = generate_mock_responses(eval_items)
    else:
        import gc
        import glob
        import torch

        # 1. 严格校验 SFT 路径是否存在
        if not os.path.exists(args.sft_model):
            raise FileNotFoundError(
                f"\n❌ 未找到指定的 SFT 模型目录: '{args.sft_model}'\n"
                f"当前执行工作目录为: '{os.getcwd()}'\n"
                f"请在 Linux 终端执行 `ls -la output/` 或 `find output/` 核实你的实际训练产物路径，"
                f"并通过 `--sft_model <你的实际路径>` 传入！"
            )

        # 2. 智能探测 LoRA 适配器 (支持顶层目录及 checkpoint-* 子目录自动识别)
        real_sft_path = args.sft_model
        adapter_cfg = os.path.join(real_sft_path, "adapter_config.json")
        
        # 若根目录下没有，自动寻找子目录最新的 checkpoint-*
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
                    print(f"🔍 自动在子目录中发现最新微调权重: {real_sft_path}")
                    break

        is_sft_lora = os.path.exists(adapter_cfg)
        if is_sft_lora:
            print(f"📦 智能检测到 SFT 权重为未合并 LoRA 适配器: {real_sft_path}")
            print("💡 启用基座+LoRA动态热插拔机制（单次加载基座，零内存浪费，秒级完成两轮生成）！")
        else:
            print(f"📦 未检测到 adapter_config.json，判定为独立全量合并模型: {real_sft_path}")

        print("🚀 检测到 GPU 与模型环境，启动 vLLM 批量多轮推理...")

        if is_sft_lora:
            # 模式 A：单实例基座 + 动态挂载 LoRA 切换 (最优雅、最快且绝不 OOM)
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest

            # 解析 LoRA 秩
            actual_lora_rank = 16
            try:
                with open(adapter_cfg, "r", encoding="utf-8") as f:
                    actual_lora_rank = int(json.load(f).get("r", 16))
            except Exception:
                pass

            print(f"正在初始化 vLLM 引擎: {args.baseline_model} (LoRA Rank: {actual_lora_rank})...")
            llm = LLM(
                model=args.baseline_model,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.90,
                max_model_len=4096,
                trust_remote_code=True,
                enable_lora=True,
                max_lora_rank=actual_lora_rank,
            )
            sampling_params = SamplingParams(
                temperature=0.3,
                top_p=0.8,
                max_tokens=args.max_tokens,
                stop=["<|im_end|>", "<|endoftext|>"],
            )

            prompts = []
            for item in eval_items:
                if "history_messages" in item:
                    prompts.append(format_chatml_prompt(item["history_messages"], item.get("system_prompt")))
                else:
                    prompts.append(format_chatml_prompt([{"role": "user", "content": item.get("query", "")}], item.get("system_prompt")))

            print(f"1. 生成 Baseline (原生基座无 LoRA) 回答 (共 {len(prompts)} 个切片)...")
            base_outputs = llm.generate(prompts, sampling_params, lora_request=None)
            baseline_res = [out.outputs[0].text.strip() for out in base_outputs]

            print(f"2. 生成 SFT (挂载 LoRA 适配器: {real_sft_path}) 回答 (共 {len(prompts)} 个切片)...")
            lora_req = LoRARequest("car_sft_lora", 1, real_sft_path)
            sft_outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
            sft_res = [out.outputs[0].text.strip() for out in sft_outputs]

        else:
            # 模式 B：两个独立的完整全量模型 (分步执行并主动回收显存)
            print("1. 生成 Baseline 模型回答...")
            baseline_res = run_vllm_inference(
                eval_items, args.baseline_model, batch_size=args.batch_size, max_tokens=args.max_tokens
            )
            gc.collect()
            torch.cuda.empty_cache()

            print("2. 生成 SFT 全量模型回答...")
            sft_res = run_vllm_inference(
                eval_items, real_sft_path, batch_size=args.batch_size, max_tokens=args.max_tokens
            )

        predictions = []
        for item, a_txt, b_txt in zip(eval_items, baseline_res, sft_res):
            predictions.append(
                {
                    **item,
                    "model_a_name": "baseline_qwen3_8b",
                    "model_a_response": a_txt,
                    "model_b_name": "sft_car_assistant",
                    "model_b_response": b_txt,
                }
            )

    # 安全落盘与灾备写入机制 (确保绝不前功尽弃)
    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(out_dir, exist_ok=True)

    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())  # 强制刷盘写入物理磁盘，防止掉电或缓冲区丢失
        print(f"✅ 模型生成结果已安全落盘至: {args.output_file} (共 {len(predictions)} 个多轮切片)")
    except Exception as e:
        backup_path = "backup_model_predictions.jsonl"
        print(f"⚠️ 写入 {args.output_file} 遇到异常: {e}，正在启动紧急灾备写入: {backup_path}...")
        with open(backup_path, "w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
        print(f"✅ 灾备数据已保存至当前目录: {os.path.abspath(backup_path)}")

    if args.run_cmmlu:
        run_evalscope_cmmlu(args.sft_model)


if __name__ == "__main__":
    main()

