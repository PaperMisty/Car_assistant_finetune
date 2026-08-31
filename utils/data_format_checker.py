"""
智能汽车客服助手 - 数据格式校验器 (DataFormatChecker)
用于 L1 阶段对大模型合成的 SFT 数据集进行严格的语法、字段、Tool Calling 协议及角色流转检查
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


class DataFormatChecker:
    """
    SFT 数据集格式与协议校验器 (L1 Filter)
    """

    VALID_ROLES = {"system", "user", "assistant", "tool"}
    VALID_SCENARIOS = {"clarification", "tool_use", "standard", "safety"}

    @classmethod
    def check_sample(cls, sample: Any) -> Tuple[bool, List[str], List[str]]:
        """
        校验单条样本数据格式
        返回: (is_valid: bool, errors: list[str], warnings: list[str])
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. 基础类型检查
        if not isinstance(sample, dict):
            return False, [f"样本必须是 JSON 对象 (dict)，当前类型为: {type(sample).__name__}"], []

        # 2. 顶层必要字段检查
        if "seed_id" not in sample or not sample["seed_id"]:
            errors.append("缺少必要顶层字段: 'seed_id' 或值为空")
        if "messages" not in sample or not isinstance(sample["messages"], list):
            errors.append("缺少必要顶层字段: 'messages'，且必须是 list 类型")
            return False, errors, warnings

        # 3. tools 字段检查 (可选但若有必须合规)
        tools = sample.get("tools", [])
        if not isinstance(tools, list):
            errors.append("字段 'tools' 必须是 list 类型")
        else:
            for idx, tool in enumerate(tools):
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    errors.append(f"tools[{idx}] 必须包含 'type': 'function'")
                fn = tool.get("function")
                if not isinstance(fn, dict) or "name" not in fn or "parameters" not in fn:
                    errors.append(f"tools[{idx}].function 必须包含 'name' 和 'parameters'")

        messages = sample["messages"]
        if len(messages) < 2:
            errors.append(f"messages 对话轮数过少 (当前为 {len(messages)} 轮)，无法构成有效对话")
            return False, errors, warnings

        # 4. 逐轮 message 语法与协议检查
        has_user = False
        has_assistant = False
        pending_tool_call_ids: List[str] = []

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"messages[{idx}] 不是合法的 dict 对象")
                continue

            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")

            # 4.1 role 校验
            if role not in cls.VALID_ROLES:
                errors.append(f"messages[{idx}] 非法角色: '{role}'，必须是 {cls.VALID_ROLES}")

            if role == "user":
                has_user = True
            elif role == "assistant":
                has_assistant = True

            # 4.2 tool_calls 与 content 联合协议校验
            if tool_calls is not None:
                if not isinstance(tool_calls, list) or len(tool_calls) == 0:
                    errors.append(f"messages[{idx}] 'tool_calls' 字段必须是非空 list")
                else:
                    for call_idx, call in enumerate(tool_calls):
                        call_id = call.get("id")
                        if not call_id:
                            errors.append(f"messages[{idx}].tool_calls[{call_idx}] 缺少 'id'")
                        else:
                            pending_tool_call_ids.append(call_id)

                        fn = call.get("function", {})
                        fn_name = fn.get("name")
                        fn_args = fn.get("arguments")

                        if not fn_name:
                            errors.append(f"messages[{idx}].tool_calls[{call_idx}] 缺少 function.name")
                        if fn_args is None:
                            errors.append(f"messages[{idx}].tool_calls[{call_idx}] 缺少 function.arguments")
                        elif isinstance(fn_args, str):
                            try:
                                json.loads(fn_args)
                            except Exception as e:
                                errors.append(f"messages[{idx}].tool_calls[{call_idx}] arguments 不是合法 JSON 字符串: {e}")
                        elif not isinstance(fn_args, dict):
                            errors.append(f"messages[{idx}].tool_calls[{call_idx}] arguments 类型必须为 JSON 字符串或 dict")

            # 4.3 tool 返回消息校验
            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if not tool_call_id:
                    errors.append(f"messages[{idx}] 角色为 'tool' 但缺少 'tool_call_id'")
                elif tool_call_id in pending_tool_call_ids:
                    pending_tool_call_ids.remove(tool_call_id)
                else:
                    warnings.append(f"messages[{idx}] tool_call_id '{tool_call_id}' 未在先前的 tool_calls 中声明")

                if content is None or not str(content).strip():
                    errors.append(f"messages[{idx}] 角色为 'tool' 但 content 为空")

            # 4.4 普通文本 content 校验
            if role in {"system", "user"}:
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"messages[{idx}] 角色 '{role}' 的 content 必须为非空字符串")
            elif role == "assistant" and tool_calls is None:
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"messages[{idx}] 普通 assistant 回复 content 不能为空")

        # 4.5 检查未闭合的 tool_call
        if pending_tool_call_ids:
            errors.append(f"存在未闭合的 tool_call_ids (未收到对应的 tool 响应): {pending_tool_call_ids}")

        if not has_user:
            errors.append("对话中未包含任何 'user' 消息")
        if not has_assistant:
            errors.append("对话中未包含任何 'assistant' 消息")

        # 5. 质量警示 (Warnings)
        if len(messages) < 4:
            warnings.append(f"对话总轮数偏少 ({len(messages)} 轮)，可能缺少必要的多轮追问或闭环")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    @classmethod
    def check_file(cls, file_path: str) -> Dict[str, Any]:
        """
        批量检查整个 JSON 或 JSONL 文件
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        samples = []
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except Exception as e:
                            return {
                                "status": "ERROR",
                                "summary": f"第 {line_idx} 行 JSON 解析失败: {e}",
                                "total": 0,
                                "passed": 0,
                                "failed": 1
                            }
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else [data]

        total = len(samples)
        passed = 0
        failed = 0
        report_details = []

        for idx, sample in enumerate(samples):
            # 兼容嵌套 list 的情况
            if isinstance(sample, list) and len(sample) > 0:
                sample = sample[0]

            is_valid, errors, warnings = cls.check_sample(sample)
            if is_valid:
                passed += 1
            else:
                failed += 1

            if not is_valid or warnings:
                report_details.append({
                    "index": idx,
                    "seed_id": sample.get("seed_id", "UNKNOWN") if isinstance(sample, dict) else "INVALID",
                    "is_valid": is_valid,
                    "errors": errors,
                    "warnings": warnings
                })

        return {
            "file_path": file_path,
            "total_samples": total,
            "passed_count": passed,
            "failed_count": failed,
            "pass_rate": f"{(passed / total * 100):.2f}%" if total > 0 else "0.00%",
            "issues": report_details
        }


if __name__ == "__main__":
    # 快速自测现有的生成日志
    test_log = "test_samples_logs/2.qwen-27b@prompt-v2.json"
    if os.path.exists(test_log):
        report = DataFormatChecker.check_file(test_log)
        print(json.dumps(report, ensure_ascii=False, indent=2))
