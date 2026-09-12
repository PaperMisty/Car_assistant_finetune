import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("custom_eval/car_assistant_eval.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

for idx in [2, 4, 5, 6, 8]:
    s = lines[idx-1]
    print(f"\n=== 样本 #{idx} ({s['id']}) ===")
    for h_idx, m in enumerate(s['full_dialog_history']):
        if m['role'] == 'user':
            print(f"[{h_idx}] User: {m['content']}")
        elif m.get('tool_calls'):
            print(f"[{h_idx}] Tool Call: {m['tool_calls'][0]['function']['name']} -> {m['tool_calls'][0]['function']['arguments']}")
        elif m['role'] == 'assistant' and not m.get('tool_calls'):
            print(f"[{h_idx}] Assistant Text: {m['content'][:60]}...")
