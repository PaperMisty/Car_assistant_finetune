"""Format a tool-call conversation with a tokenizer chat template."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODEL_PATH = Path("model/Qwen3-0.6B")
SCHEMA_PATH = Path("data/v2/tool_schemas.json")
SEEDS_DIR = Path("data/v2/seeds/train")
SEED_ID = "cat4_booking_001"
DEMO_USER_DETAILS = "我的车辆标识是 VIN_DEMO_001，想去 STORE_DEMO_001，周六上午有时间。"
TOOL_ARGUMENTS = {
    "vehicle_id": "VIN_DEMO_001",
    "store_id": "STORE_DEMO_001",
    "preferred_date": "2026-09-01",
}


def load_seed(seeds_dir: Path, seed_id: str) -> dict[str, Any]:
    """Load one existing V2 seed by its seed_id."""
    for path in sorted(seeds_dir.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seed = json.loads(line)
                if seed.get("seed_id") == seed_id:
                    return seed
    raise ValueError(f"seed not found: {seed_id}")


def load_tool_schema(schema_path: Path, tool_name: str) -> dict[str, Any]:
    """Load one tool definition from the local registry."""
    tools = json.loads(schema_path.read_text(encoding="utf-8")).get("tools", [])
    for tool in tools:
        if tool.get("name") == tool_name:
            return tool
    raise ValueError(f"tool schema not found: {tool_name}")

def run_demo(
    tokenizer: Any,
    seed: dict[str, Any],
    tool_schema: dict[str, Any],
    user_details: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:


    followup = "为了帮助您" + seed["user_goal"] + "，请提供：" + "、".join(
            seed["required_questions"]
        ) + "。"
    messages = [
            {"role": "user", "content": seed["scenario"]},
            {"role": "assistant", "content": followup},
            {"role": "user", "content": user_details},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_schema["name"],
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            },
        ]
    result = tokenizer.apply_chat_template(
        messages,
        tools=[tool_schema],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {
        "messages": messages,
        "formatted":result
    }


def main() -> int:
    # {"seed_id": "cat4_booking_001", "split": "train", "category_id": 4, "category": "预约与服务受理", "subcategory": "常规保养预约", "scenario": "客户希望预约周六上午做常规保养", "user_goal": "找到可用时段并完成预约", "customer_role": "车主本人", "service_stage": "预约中", "required_facts": ["预约需以门店实时工位为准", "应确认车型和服务项目"], "required_questions": ["意向门店", "车型车牌和联系电话"], "required_actions": ["查询时段并在客户确认后创建预约"], "prohibited_actions": ["不得未查询即保证具体时段"], "tool_required": true, "tool_name": "service_booking"}

    seed = load_seed(SEEDS_DIR, SEED_ID)
    tool_schema = load_tool_schema(SCHEMA_PATH, seed["tool_name"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    result = run_demo(
        tokenizer, seed, tool_schema, DEMO_USER_DETAILS, TOOL_ARGUMENTS
    )

    print("=== 1. 输入种子数据 ===")
    print(json.dumps(seed, ensure_ascii=False, indent=2))


    print("=== 2. 种子约束衍生出的训练 messages ===")
    print(json.dumps(result["messages"], ensure_ascii=False, indent=2))


    print("=== 3. 基于message格式化之后的结果 ===")
    print(result["formatted"])
    return 0


if __name__ == "__main__":
    main()
