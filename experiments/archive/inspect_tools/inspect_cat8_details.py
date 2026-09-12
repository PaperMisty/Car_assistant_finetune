import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('custom_eval/car_assistant_eval.jsonl', 'r', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f]

for i in range(56, 64):
    s = samples[i]
    subcat = s.get('subcategory', '')
    sid = s.get('id', '')
    print(f"=== Sample {i+1}: {subcat} ({sid}) ===")
    print("Scenario:", s.get('scenario'))
    for turn in s.get('full_dialog_history', []):
        if turn.get('role') == 'user':
            print("  User:", turn.get('content')[:80])
        elif turn.get('tool_calls'):
            for tc in turn['tool_calls']:
                print("  ToolCall:", tc['function']['name'], tc['function'].get('arguments'))
