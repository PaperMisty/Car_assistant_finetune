import json

eval_file = "custom_eval/car_assistant_eval.jsonl"
with open(eval_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if not line.strip():
            continue
        data = json.loads(line)
        history = data.get("full_dialog_history", [])
        for h_idx, msg in enumerate(history):
            if msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    args = tc.get("function", {}).get("arguments", "")
                    if "unknown" in args or "null" in args or "{}" == args.strip():
                        print(f"Line {idx} ({data.get('id')}) Turn {h_idx//2 + 1}: Suspicious args: {args}")
            if msg.get("role") == "tool":
                c = msg.get("content", "")
                if "error" in c.lower() or "missing" in c.lower():
                    print(f"Line {idx} ({data.get('id')}): Tool returned error/missing: {c[:80]}")
