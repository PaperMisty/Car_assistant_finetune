import json
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

eval_file = "custom_eval/car_assistant_eval.jsonl"
bak_file = "custom_eval/car_assistant_eval.jsonl.bak2"

shutil.copyfile(eval_file, bak_file)
print(f"Backed up {eval_file} to {bak_file}")

with open(eval_file, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

# ----------------------------------------------------
# 1. Category 1 修复: Sample 1, 6, 8 (补齐首轮 VIN)
# ----------------------------------------------------
# Sample 1
s1 = samples[0]
s1["query"] = s1["query"].replace("你好！我现在特别着急，我正准备去赶飞机，", "你好！我现在特别着急，我正准备去赶飞机，车架号是 LSVAA123456789012，")
for d in s1["full_dialog_history"]:
    if d["role"] == "user" and "大灯居然没自动亮起来" in d["content"]:
        d["content"] = d["content"].replace("你好！我现在特别着急，我正准备去赶飞机，", "你好！我现在特别着急，我正准备去赶飞机，车架号是 LSVAA123456789012，")

# Sample 6
s6 = samples[5]
s6["query"] = s6["query"].replace("客服在吗？快帮我看看！", "客服在吗？快帮我看看！车架号是 LSVSUV2023009988。")
for d in s6["full_dialog_history"]:
    if d["role"] == "user" and "空气净化" in d["content"]:
        d["content"] = d["content"].replace("客服在吗？快帮我看看！", "客服在吗？快帮我看看！车架号是 LSVSUV2023009988。")

# Sample 8
s8 = samples[7]
s8["query"] = s8["query"].replace("你好！我赶着马上要上高速出差，", "你好！我车架号是 LSVMAX2023008877。我赶着马上要上高速出差，")
for d in s8["full_dialog_history"]:
    if d["role"] == "user" and "胎压显示一直闪烁报警" in d["content"]:
        d["content"] = d["content"].replace("你好！我赶着马上要上高速出差，", "你好！我车架号是 LSVMAX2023008877。我赶着马上要上高速出差，")

# ----------------------------------------------------
# 2. Category 4 修复: Sample 26, 28, 30, 32 (车牌转17位VIN, 消除首轮抢跑)
# ----------------------------------------------------
# Sample 26 (车牌 浙A7E98B -> LSVAA7E98B1029384)
s26 = samples[25]
s26_text = json.dumps(s26, ensure_ascii=False)
s26_text = s26_text.replace("浙A7E98B", "LSVAA7E98B1029384").replace('"license_plate"', '"vehicle_id"')
samples[25] = json.loads(s26_text)

# Sample 28 (车牌 粤B88899 -> LSVBB888991029384)
s28 = samples[27]
s28_text = json.dumps(s28, ensure_ascii=False)
s28_text = s28_text.replace("粤B88899", "LSVBB888991029384").replace('"license_plate"', '"vehicle_id"')
samples[27] = json.loads(s28_text)

# Sample 30 (车牌 京AD88921 -> LSVCC889211029384)
s30 = samples[29]
s30_text = json.dumps(s30, ensure_ascii=False)
s30_text = s30_text.replace("京AD88921", "LSVCC889211029384").replace('"vin_or_plate"', '"vehicle_id"')
samples[29] = json.loads(s30_text)
# 加强 Turn 2 的预约服务意图
for d in samples[29]["full_dialog_history"]:
    if d["role"] == "user" and "朝阳望京服务中心" in d["content"]:
        d["content"] = "车架号是 LSVCC889211029384，我去朝阳望京服务中心最方便，想要预约明天上午10点左右的服务工位配对智能钥匙，能排上吗？资料我这全带了！"

# Sample 32 (车牌 京A·D12345 -> LSVDD123451029384，消除首轮抢跑)
s32 = samples[31]
s32_text = json.dumps(s32, ensure_ascii=False)
s32_text = s32_text.replace("京A·D12345", "LSVDD123451029384").replace("京AD12345", "LSVDD123451029384").replace('"license_plate_or_vin"', '"vehicle_id"')
s32 = json.loads(s32_text)
# Turn 1 用户只是报修故障，助手不应抢跑调用预约，转为纯文本反问安抚；Turn 2 用户明确预约时调用
hist32 = s32["full_dialog_history"]
if len(hist32) >= 3 and hist32[2].get("tool_calls"):
    # 将 Turn 2 的 tool_calls 移除，替换为安抚指导文本
    hist32[2] = {
        "role": "assistant",
        "content": "车主您好，车机在特定路段黑屏卡死可能与离线高精地图加载异常或软件通信超载有关。为了确保行车安全，您方便提供车架号并预约就近服务中心进行车机软件专项检测吗？"
    }
    # 移除紧接着的 tool 返回
    if len(hist32) > 3 and hist32[3]["role"] == "tool":
        hist32.pop(3)
samples[31] = s32

# ----------------------------------------------------
# 3. Category 7 修复: Sample 50, 56 (车牌转17位VIN)
# ----------------------------------------------------
# Sample 50 (浙A·D58219 -> LSVAU582191029384)
s50 = samples[49]
s50_text = json.dumps(s50, ensure_ascii=False)
s50_text = s50_text.replace("浙A·D58219", "LSVAU582191029384").replace("浙AD58219", "LSVAU582191029384")
samples[49] = json.loads(s50_text)

# Sample 56 (京A9988X -> LSVAU9988X1029384)
s56 = samples[55]
s56_text = json.dumps(s56, ensure_ascii=False)
s56_text = s56_text.replace("京A9988X", "LSVAU9988X1029384")
samples[55] = json.loads(s56_text)

# ----------------------------------------------------
# 4. Category 8 修复: Sample 57~64 (采用方案 B，映射为真实原子工具)
# ----------------------------------------------------
cat8_tool_mapping = [
    # 57: 生日关怀偏好退订
    ("communication_consent_update", {"channel": "sms", "consent": False}),
    # 58: 节假日门店营业时间查询
    ("dealer_service_query", {"city": "杭州", "service_type": "国庆节假日营业时间"}),
    # 59: 冬季轮胎检查活动预约
    ("service_booking", {"vehicle_id": "LVSHFLLC5MS123456", "store_id": "沈阳浑南服务中心", "preferred_date": "明天上午"}),
    # 60: 夏季空调性能检测政策
    ("maintenance_policy_query", {"vehicle_model": "2024款", "mileage": 10000}),
    # 61: 低压电池健康度提示查询
    ("vehicle_health_query", {"vehicle_id": "LSVCC123456789XYZ"}),
    # 62: 胎压异常联网数据查询
    ("vehicle_health_query", {"vehicle_id": "LSVAA456789123456"}),
    # 63: 官方召回状态核查
    ("recall_status_query", {"vehicle_id": "LSVAU2A39MN123456"}),
    # 64: 爽约改期重新预约
    ("service_booking", {"vehicle_id": "LSVEE202311009876", "store_id": "直营服务中心", "preferred_date": "明天上午09:30"}),
]

for idx, (target_tool, target_args) in enumerate(cat8_tool_mapping):
    sample_idx = 56 + idx
    s = samples[sample_idx]
    s["tool_name"] = target_tool
    
    # 遍历更新 full_dialog_history 中的 tool_calls
    for turn in s.get("full_dialog_history", []):
        if turn.get("role") == "assistant" and turn.get("tool_calls"):
            for tc in turn["tool_calls"]:
                tc["function"]["name"] = target_tool
                tc["function"]["arguments"] = json.dumps(target_args, ensure_ascii=False)
        elif turn.get("role") == "tool":
            turn["name"] = target_tool
            turn["content"] = json.dumps({"status": "success", "result": f"{target_tool} 校验成功"}, ensure_ascii=False)
            
    # tools 字段如果有的话也更新
    if "tools" in s and s["tools"]:
        for t in s["tools"]:
            if t.get("function"):
                t["function"]["name"] = target_tool

# 保存回评测文件
with open(eval_file, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Successfully applied all fixes to {eval_file}! Total samples: {len(samples)}")
