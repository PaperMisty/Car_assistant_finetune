import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("custom_eval/car_assistant_eval.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

print("=== 分析样本 49~56 (Category 7: 投诉、质量争议与升级处理) ===")
for idx in range(49, 57):
    s = lines[idx-1]
    print(f"\n样本 #{idx} [{s.get('id')}] - {s.get('subcategory')}:")
    for h_idx, m in enumerate(s['full_dialog_history']):
        if m['role'] == 'user':
            print(f"  [User {h_idx//2+1}]: {m['content'][:70]}...")
        elif m.get('tool_calls'):
            fname = m['tool_calls'][0]['function']['name']
            fargs = m['tool_calls'][0]['function']['arguments']
            print(f"  -> Tool Call (Turn {h_idx//2+1}): {fname} | 参数: {fargs[:80]}...")
