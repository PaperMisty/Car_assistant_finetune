import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("custom_eval/car_assistant_eval.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

print("================================================================================")
print("🔍 重点排查 1、4、7、8 类别的工具调用轮次，检查 VIN 码提供情况与参数规范性")
print("================================================================================")

target_categories = [1, 4, 7, 8]

for idx, sample in enumerate(lines, 1):
    cat_id = sample.get("category_id")
    if cat_id not in target_categories:
        continue

    cat_name = sample.get("category")
    subcat = sample.get("subcategory")
    s_id = sample.get("id")
    history = sample.get("full_dialog_history", [])

    for h_idx in range(len(history) - 1):
        if history[h_idx]["role"] == "user" and history[h_idx+1]["role"] == "assistant":
            tc = history[h_idx+1].get("tool_calls")
            if tc:
                user_msg = history[h_idx]["content"]
                for c in tc:
                    fname = c["function"]["name"]
                    args_str = c["function"]["arguments"]
                    args = json.loads(args_str) if args_str else {}
                    
                    # 检查用户消息和历史消息中是否出现 VIN 码特征 (例如 17位长串，或以 L/车架号/VIN 标识)
                    all_user_text = " ".join([m["content"] for m in history[:h_idx+1] if m["role"] == "user"])
                    has_vin_in_user = any(k in all_user_text for k in ["VIN", "车架号", "车牌", "LSV", "LFV", "LDP", "京A", "粤B", "浙A", "沪A"])
                    has_vin_in_args = any(k in args for k in ["vehicle_id", "vin", "license_plate", "vin_or_plate", "license_plate_or_vin", "vehicle_vin"])

                    print(f"#{idx} [{cat_name} - {subcat}] Turn {h_idx//2 + 1} (Tool: {fname}):")
                    print(f"   用户本轮: {user_msg[:65]}...")
                    print(f"   参数: {args}")
                    print(f"   VIN检测: 用户是否提到了VIN/车牌={has_vin_in_user} | 参数是否有VIN字段={has_vin_in_args}")
                    if not has_vin_in_user or not has_vin_in_args:
                        print(f"   ⚠️ 缺少VIN/车牌标识！需重点排查修正！")
                    print()
