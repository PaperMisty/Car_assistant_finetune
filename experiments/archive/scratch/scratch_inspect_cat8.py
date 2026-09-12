import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("custom_eval/car_assistant_eval.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

print("=== 扫描样本 57~64 (Category 8: 主动关怀与客户运营) ===")
for idx in range(57, 65):
    s = lines[idx-1]
    print(f"\n样本 #{idx} [{s.get('id')}] - {s.get('subcategory')}:")
    print(f"  定义工具: {s.get('tool_name')}")
    if "tools" in s and s["tools"]:
        t_func = s["tools"][0].get("function", {})
        print(f"  样本内自带工具描述: {t_func.get('description')}")
        print(f"  自带参数: {list(t_func.get('parameters', {}).get('properties', {}).keys())}")
    for h_idx, m in enumerate(s['full_dialog_history']):
        if m['role'] == 'user':
            print(f"  User: {m['content'][:60]}...")
        elif m.get('tool_calls'):
            print(f"  Tool Call: {m['tool_calls'][0]['function']['name']} -> {m['tool_calls'][0]['function']['arguments']}")
