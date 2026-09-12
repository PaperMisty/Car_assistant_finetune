import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

eval_file = "custom_eval/car_assistant_eval.jsonl"
with open(eval_file, "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

for idx in range(25):
    item = lines[idx]
    history = item.get("full_dialog_history", [])
    cat = item.get("category")
    subcat = item.get("subcategory")
    for h_idx in range(len(history) - 1):
        if history[h_idx]["role"] == "user" and history[h_idx+1]["role"] == "assistant":
            tc = history[h_idx+1].get("tool_calls")
            if tc:
                for c in tc:
                    fname = c.get("function", {}).get("name")
                    args = c.get("function", {}).get("arguments", "")
                    user_q = history[h_idx]["content"]
                    print(f"#{idx+1} [{cat} - {subcat}] Turn {h_idx//2 + 1}:")
                    print(f"   用户提问: {user_q[:50]}...")
                    print(f"   调用工具: {fname} | 参数: {args}")
