import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.tool import export_openai_schemas

schemas = export_openai_schemas()
vehicle_id_tools = set()
for s in schemas:
    fname = s["function"]["name"]
    reqs = s["function"].get("parameters", {}).get("required", [])
    if "vehicle_id" in reqs:
        vehicle_id_tools.add(fname)

print("工具表中严格要求必传 vehicle_id 的工具有:", vehicle_id_tools)

with open("custom_eval/car_assistant_eval.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f if l.strip()]

issues = []
for idx, sample in enumerate(lines, 1):
    history = sample.get("full_dialog_history", [])
    for h_idx in range(len(history) - 1):
        if history[h_idx]["role"] == "user" and history[h_idx+1]["role"] == "assistant":
            tc = history[h_idx+1].get("tool_calls")
            if tc:
                for c in tc:
                    fname = c["function"]["name"]
                    args_str = c["function"]["arguments"]
                    args = json.loads(args_str) if args_str else {}

                    if fname in vehicle_id_tools:
                        # 检查是否有 vehicle_id / vin / license_plate 实体
                        vid_keys = ["vehicle_id", "vin", "license_plate", "vin_or_plate", "license_plate_or_vin"]
                        has_vid = any(k in args and args[k] and str(args[k]).strip() for k in vid_keys)
                        if not has_vid:
                            issues.append({
                                "sample_idx": idx,
                                "id": sample.get("id"),
                                "category": sample.get("category"),
                                "subcategory": sample.get("subcategory"),
                                "turn": h_idx // 2 + 1,
                                "tool": fname,
                                "args": args,
                                "user_msg": history[h_idx]["content"]
                            })

print(f"\n🚨 扫描结果：共发现 {len(issues)} 处【工具必传 vehicle_id，但参数中完全没有 vehicle_id/VIN/车牌号】的异常轮次！\n")
for iss in issues:
    print(f"• 样本 #{iss['sample_idx']} [{iss['category']} - {iss['subcategory']}] (ID: {iss['id']}) 第 {iss['turn']} 轮:")
    print(f"  用户说: \"{iss['user_msg'][:70]}...\"")
    print(f"  测试集调用的工具: 【{iss['tool']}】")
    print(f"  测试集传的参数: {iss['args']}")
    print()
