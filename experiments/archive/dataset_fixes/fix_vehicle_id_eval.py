import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_PATH = "custom_eval/car_assistant_eval.jsonl"

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

fix_count = 0

for idx, sample in enumerate(lines, 1):
    s_id = sample.get("id")
    history = sample.get("full_dialog_history", [])

    # 1. 样本 #2 (cat1_val_002_var0)
    if s_id == "cat1_val_002_var0":
        for msg in history:
            if msg.get("role") == "user" and "我是2023款 G7 Max" in msg.get("content", ""):
                msg["content"] = "我是2023款 G7 Max，车机版本是OS 3.2，车架号是 LSVAA8888GG102938。我现在人就在车旁边，你快查查！"
                fix_count += 1
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVAA8888GG102938",
                    "feature_name": "外后视镜自动折叠"
                }, ensure_ascii=False)

    # 2. 样本 #4 (cat1_val_004_var0)
    elif s_id == "cat1_val_004_var0":
        for msg in history:
            if msg.get("role") == "user" and "我车是2023款ET7" in msg.get("content", ""):
                msg["content"] = msg["content"].replace("我车是2023款ET7！", "我车是2023款ET7，车架号是 LSVET720230012345！")
                fix_count += 1
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVET720230012345",
                    "feature_name": "天窗与遮阳帘"
                }, ensure_ascii=False)

    # 3. 样本 #5 (cat1_val_005_var0)
    elif s_id == "cat1_val_005_var0":
        for msg in history:
            if msg.get("role") == "user" and "是2022款的旗舰版" in msg.get("content", ""):
                msg["content"] = msg["content"].replace("是2022款的旗舰版，系统版本我看了一下是 V4.2.1。", "是2022款的旗舰版，系统版本我看了一下是 V4.2.1，车架号是 LSVAA2022FL009876。")
                fix_count += 1
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVAA2022FL009876",
                    "feature_name": "远程空调预冷"
                }, ensure_ascii=False)

    # 4. 样本 #6 (cat1_val_006_var0)
    elif s_id == "cat1_val_006_var0":
        for msg in history:
            if msg.get("role") == "user" and "车是2023款智享SUV" in msg.get("content", ""):
                msg["content"] = msg["content"].replace("车是2023款智享SUV，车机系统是上周刚升的OS 3.2，", "车是2023款智享SUV，车机系统是上周刚升的OS 3.2，车架号是 LSVSUV2023009988，")
                fix_count += 1
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVSUV2023009988",
                    "feature_name": "空气净化"
                }, ensure_ascii=False)

    # 5. 样本 #8 (cat1_val_008_var0)
    elif s_id == "cat1_val_008_var0":
        for msg in history:
            if msg.get("role") == "user" and "我的车是 2023款 智行Max" in msg.get("content", ""):
                msg["content"] = msg["content"].replace("我的车是 2023款 智行Max，系统是最新的 OS 3.2，", "我的车是 2023款 智行Max，系统是最新的 OS 3.2，车架号是 LSVMAX2023008877，")
                fix_count += 1
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVMAX2023008877",
                    "feature_name": "胎压显示复位"
                }, ensure_ascii=False)

    # 6. 样本 #42 (cat6_val_002_var0 - 道路救援)
    elif s_id == "cat6_val_002_var0":
        for msg in history:
            if msg.get("role") == "user" and "钥匙我拔了塞口袋了" in msg.get("content", ""):
                msg["content"] = "人很安全，钥匙我拔了塞口袋了，绝对没点火。车牌是冀A98765，车架号是 LSVGA123456789012。我现在在京港澳高速涿州服务区南向加油站内，你们拖车多久能到？"
                fix_count += 1
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVGA123456789012",
                    "question": "误加燃油拖车救援政策"
                }, ensure_ascii=False)

# 保存清洗结果
with open(EVAL_PATH, "w", encoding="utf-8") as f:
    for item in lines:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"🎉 成功补齐并规范化 {fix_count} 处缺少必填 vehicle_id 的评测集样本！")
