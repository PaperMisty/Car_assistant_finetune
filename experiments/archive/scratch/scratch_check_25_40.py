import json

eval_file = "custom_eval/car_assistant_eval.jsonl"
with open(eval_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx < 24 or idx >= 40:
            continue
        item = json.loads(line)
        print("=" * 80)
        print(f"#{idx+1} | ID: {item['id']} | 类别: {item.get('category')} | 子类: {item.get('subcategory')}")
        print(f"工具: {item.get('tool_name')}")
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
