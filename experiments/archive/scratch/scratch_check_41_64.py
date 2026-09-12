import json

eval_file = "custom_eval/car_assistant_eval.jsonl"
with open(eval_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx < 40:
            continue
        item = json.loads(line)
        history = item.get("full_dialog_history", [])
        tools_called = []
        for msg in history:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls"):
                    tools_called.append(tc.get("function", {}).get("name"))
        print(f"#{idx+1} | ID: {item['id']} | 类别: {item.get('category')} | 声明tool: {item.get('tool_name')} | 实际调用: {tools_called}")
