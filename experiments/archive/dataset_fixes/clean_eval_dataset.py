import json
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_PATH = "custom_eval/car_assistant_eval.jsonl"
BACKUP_PATH = "custom_eval/car_assistant_eval.jsonl.bak"

# 确保有备份
if not os.path.exists(BACKUP_PATH):
    shutil.copyfile(EVAL_PATH, BACKUP_PATH)
    print(f"已创建备份文件: {BACKUP_PATH}")

with open(BACKUP_PATH, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

cleaned_count = 0
fixed_tool_count = 0

cleaned_lines = []

for idx, sample in enumerate(lines):
    s_id = sample.get("id", "")
    cat_id = sample.get("category_id")

    # 1. 针对样本 9 (cat2_val_001_var0)：清理 Turn 1 缺少必要车型信息的异常抢跑调用
    if s_id == "cat2_val_001_var0":
        history = sample.get("full_dialog_history", [])
        new_history = []
        skip_next_tool = False

        for h_idx, msg in enumerate(history):
            # 识别 Turn 1 处的异常 tool_calls ("unknown" 参数)
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tc_args = msg["tool_calls"][0].get("function", {}).get("arguments", "")
                if "unknown" in tc_args:
                    # 跳过这个伪调用
                    skip_next_tool = True
                    cleaned_count += 1
                    continue
            if msg.get("role") == "tool" and skip_next_tool:
                # 跳过对应的伪报错 tool 回包
                skip_next_tool = False
                continue
            new_history.append(msg)

        sample["full_dialog_history"] = new_history
        print(f"✅ 样本 9: 成功移除 Turn 1 缺乏车型年款时的伪工具调用 (历史消息从 {len(history)} 条规范化为 {len(new_history)} 条)")

    # 2. 针对 Category 2 (样本 9~16，即 idx 8~15)：纠正保养周期工具错配 (service_policy_query -> maintenance_policy_query)
    if cat_id == 2 or sample.get("category") == "保养、质保与服务政策":
        # 修正顶层 tool_name
        if sample.get("tool_name") == "service_policy_query":
            sample["tool_name"] = "maintenance_policy_query"
            fixed_tool_count += 1

        # 修正 tools 声明
        if "tools" in sample:
            for t in sample["tools"]:
                if t.get("function", {}).get("name") == "service_policy_query":
                    t["function"]["name"] = "maintenance_policy_query"

        # 修正对话历史里的所有调用与回包
        for msg in sample.get("full_dialog_history", []):
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("function", {}).get("name") == "service_policy_query":
                        tc["function"]["name"] = "maintenance_policy_query"
            if msg.get("role") == "tool" and msg.get("name") == "service_policy_query":
                msg["name"] = "maintenance_policy_query"

    cleaned_lines.append(sample)

# 保存清洗修正后的文件
with open(EVAL_PATH, "w", encoding="utf-8") as f:
    for item in cleaned_lines:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n" + "=" * 80)
print(f"🎉 评测集数据清洗完毕！")
print(f"• 修复异常抢跑工具调用: {cleaned_count} 处")
print(f"• 纠正保养业务工具标注错配 (service_policy_query -> maintenance_policy_query): {fixed_tool_count} 个样本")
print(f"• 已写回目标文件: {EVAL_PATH}")
print("=" * 80)
