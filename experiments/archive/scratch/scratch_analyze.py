import json

eval_file = "custom_eval/car_assistant_eval.jsonl"
with open(eval_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= 16:
            break
        item = json.loads(line)
        print("=" * 80)
        print(f"#{idx+1} | ID: {item['id']} | 子类: {item.get('subcategory')} | 场景: {item.get('scenario')}")
        print(f"顶层 tool_name: {item.get('tool_name')}")
        history = item.get("full_dialog_history", [])
        for h_idx, msg in enumerate(history):
            role = msg.get("role")
            if role == "user":
                print(f"  [User {h_idx//2+1}]: {msg.get('content')[:60]}...")
            elif role == "assistant" and msg.get("tool_calls"):
                tc = msg.get("tool_calls")[0]
                t_name = tc.get("function", {}).get("name")
                t_args = tc.get("function", {}).get("arguments")
                print(f"  -> [Call Tool]: {t_name} 参数: {t_args}")
            elif role == "assistant" and not msg.get("tool_calls") and h_idx == 1:
                print(f"  -> [Text Reply]: {msg.get('content')[:50]}...")
