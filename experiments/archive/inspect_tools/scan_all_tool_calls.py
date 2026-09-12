import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

eval_file = "custom_eval/car_assistant_eval.jsonl"
with open(eval_file, "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

print(f"总计 {len(lines)} 个样本，扫描所有工具调用轮次:")
for idx, item in enumerate(lines, 1):
    history = item.get("full_dialog_history", [])
    for h_idx in range(len(history) - 1):
        if history[h_idx]["role"] == "user" and history[h_idx+1]["role"] == "assistant":
            tc = history[h_idx+1].get("tool_calls")
            if tc:
                for c in tc:
                    fname = c.get("function", {}).get("name")
                    args = c.get("function", {}).get("arguments", "")
                    user_text = history[h_idx]["content"]
                    # 检查是否有 unknown 或者异常
                    has_unknown = "unknown" in args or "null" in args or "{}" == args
                    print(f"#{idx} [{item.get('category')} - {item.get('subcategory')}] Turn {h_idx//2 + 1}: Tool={fname} | Args={args}")
                    if has_unknown:
                        print(f"   ⚠️ 发现未知/缺失参数异常: {args}")
                        print(f"   用户输入: {user_text[:60]}")
