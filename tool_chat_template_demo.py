"""Format a tool-call conversation with a tokenizer chat template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARGUMENTS = {
    "service_booking": {
        "vehicle_id": "VIN_DEMO_001",
        "store_id": "STORE_DEMO_001",
        "preferred_date": "2026-09-01",
    }
}


def load_tool_schema(schema_path: Path, tool_name: str) -> dict[str, Any]:
    """Load one tool definition from the local registry."""
    tools = json.loads(schema_path.read_text(encoding="utf-8")).get("tools", [])
    for tool in tools:
        if tool.get("name") == tool_name:
            return tool
    raise ValueError(f"tool schema not found: {tool_name}")


def build_tool_call_messages(
    user_message: str,
    tool_schema: dict[str, Any],
    arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one user turn followed by an assistant function call."""
    return [
        {"role": "user", "content": user_message},
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


def format_tool_call_sample(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tool_schema: dict[str, Any],
) -> str:
    """Return the exact text the tokenizer template renders for this sample."""
    return tokenizer.apply_chat_template(
        messages,
        tools=[tool_schema],
        tokenize=False,
        add_generation_prompt=False,
    )


def parse_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--arguments must be a JSON object: {exc.msg}") from exc
    if not isinstance(arguments, dict):
        raise ValueError("--arguments must be a JSON object")
    return arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("model/Qwen3-0.6B"))
    parser.add_argument("--schemas", type=Path, default=Path("data/v2/tool_schemas.json"))
    parser.add_argument("--tool-name", default="service_booking")
    parser.add_argument("--user-message", default="我想预约本周末到店保养。")
    parser.add_argument(
        "--arguments",
        help="Function arguments as a JSON object; required for tools without a demo default.",
    )
    args = parser.parse_args()

    tool_schema = load_tool_schema(args.schemas, args.tool_name)
    raw_arguments = args.arguments or json.dumps(DEFAULT_ARGUMENTS.get(args.tool_name, {}))
    arguments = parse_arguments(raw_arguments)
    messages = build_tool_call_messages(args.user_message, tool_schema, arguments)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    formatted = format_tool_call_sample(tokenizer, messages, tool_schema)


    print("=== apply_chat_template output ===")
    print(formatted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
