import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('custom_eval/car_assistant_eval.jsonl', 'r', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f]

vin_pattern = re.compile(r'[A-HJ-NPR-Z0-9]{17}')

for cat_id in [1, 4, 7, 8]:
    print(f"\n==================== CATEGORY {cat_id} ====================")
    start_idx = (cat_id - 1) * 8
    end_idx = start_idx + 8
    for i in range(start_idx, end_idx):
        s = samples[i]
        sample_id = i + 1
        dialogue = s.get('full_dialog_history', [])
        scenario = s.get('scenario', '')
        
        # 遍历对话中的 tool_calls
        for t_idx, turn in enumerate(dialogue):
            if turn.get('role') == 'assistant' and turn.get('tool_calls'):
                # 检查在此轮之前（包含此轮），用户是否提到了 VIN
                user_text_history = " ".join([d['content'] for d in dialogue[:t_idx] if d.get('role') == 'user' and d.get('content')])
                vins_in_history = vin_pattern.findall(user_text_history)
                
                for tc in turn['tool_calls']:
                    fname = tc['function']['name']
                    args = tc['function'].get('arguments', {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            pass
                    vehicle_id_in_args = args.get('vehicle_id') or args.get('vin') if isinstance(args, dict) else None
                    
                    user_last = dialogue[t_idx-1]['content'] if t_idx > 0 else ''
                    
                    status = "OK"
                    if vehicle_id_in_args and not vins_in_history:
                        status = "MISSING_VIN_IN_USER_INPUT (用户未提供VIN却调用带VIN工具)"
                    elif not vehicle_id_in_args and ('vehicle_id' in str(tc) or 'vin' in str(tc)):
                        status = "NO_VIN_ARG"
                        
                    print(f"Sample {sample_id} [Turn {t_idx}] Tool: {fname} | Status: {status}")
                    print(f"   Scenario: {scenario}")
                    print(f"   Args: {args}")
                    print(f"   User Last: {user_last[:80]}...")
                    print(f"   VIN in History: {vins_in_history}\n")
