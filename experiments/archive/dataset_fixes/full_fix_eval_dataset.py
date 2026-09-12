import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_PATH = "custom_eval/car_assistant_eval.jsonl"

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

print(f"开始全面修正 1, 4, 7, 8 类别的验证集数据 (总样本数: {len(samples)})...")

fixed_details = []

for idx, sample in enumerate(samples, 1):
    s_id = sample.get("id")
    cat_id = sample.get("category_id")
    history = sample.get("full_dialog_history", [])

    # =========================================================================
    # Category 1: 用车与智能功能支持
    # =========================================================================
    if s_id == "cat1_val_001_var0":
        # 样本 1: 自动大灯，补充用户原话中的 VIN
        for msg in history:
            if msg.get("role") == "user" and "刚才进隧道的时候大灯居然没自动亮起来" in msg.get("content", ""):
                if "LSVAA123456789012" not in msg["content"]:
                    msg["content"] = msg["content"].replace(
                        "差点撞上去！这车是不是有毛病？赶紧告诉我怎么处理，别耽误我赶飞机！",
                        "差点撞上去！我车架号是 LSVAA123456789012，这车是不是有毛病？赶紧告诉我怎么处理，别耽误我赶飞机！"
                    )
                    fixed_details.append("样本 #1 补齐用户 VIN: LSVAA123456789012")
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVAA123456789012",
                    "feature_name": "自动大灯"
                }, ensure_ascii=False)

    elif s_id == "cat1_val_004_var0":
        # 样本 4: 全景天窗与遮阳帘
        for msg in history:
            if msg.get("role") == "user" and "刚才按顶棚按键把天窗关了" in msg.get("content", ""):
                msg["content"] = "客服快帮我看看！我正赶着去机场接人，太阳特别晒，刚才按顶棚按键把天窗关了，结果玻璃关严了，遮阳帘怎么还停在中间半开着不动？急死我了，是不是电机坏了卡住了？我车是2023款ET7，车架号是 LSVET720230012345，快帮我查查天窗与电动遮阳帘的设置逻辑和按键功能！"
                fixed_details.append("样本 #4 优化提问并补齐天窗遮阳帘功能项与 VIN")
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVET720230012345",
                    "feature_name": "全景天窗与遮阳帘"
                }, ensure_ascii=False)

    # =========================================================================
    # Category 4: 预约与服务受理
    # =========================================================================
    elif s_id == "cat4_val_005_var0":
        # 样本 29: 前风挡玻璃修复预约
        for msg in history:
            if msg.get("role") == "user" and "前挡风玻璃被石子崩了个小" in msg.get("content", ""):
                msg["content"] = "急死我了！前挡风玻璃被石子崩了个小坑，正好在驾驶员视线正前方，太影响开车了。车架号是 LSVBC234567890123，我明天一早要跑长途，今天浦东服务中心能不能赶紧给我预约修好？别让我等太久！"
                fixed_details.append("样本 #29 补齐用户原话中的 VIN 与门店预约诉求")
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVBC234567890123",
                    "store_name": "浦东服务中心",
                    "service_item": "前挡玻璃修复预约",
                    "preferred_date": "今天下午"
                }, ensure_ascii=False)

    # =========================================================================
    # Category 7: 投诉、质量争议与升级处理 (重点补齐投诉提问中的 VIN 与核心诉求)
    # =========================================================================
    elif s_id == "cat7_val_001_var0":
        # 样本 49: 技师态度恶劣，Turn 3 补全投诉诉求
        for msg in history:
            if msg.get("role") == "user" and "车架号是 LSVAU2A35JN123456" in msg.get("content", ""):
                msg["content"] = "车架号是 LSVAU2A35JN123456。首先得把车仔细检查一遍，其次这种轻视老人的服务态度必须正式立案投诉追责并升级督办！"
                fixed_details.append("样本 #49 强化投诉立案督办语义")
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVAU2A35JN123456",
                    "issue_summary": "客户投诉旗舰店技师态度冷漠未出示检查报告，要求重新检查并立案投诉督办"
                }, ensure_ascii=False)

    elif s_id == "cat7_val_002_var0":
        # 样本 50: 无障碍通道被堵
        for msg in history:
            if msg.get("role") == "user" and "车牌是浙A·D58219" in msg.get("content", ""):
                msg["content"] = "就在东侧入口坡道正下方，车牌是浙A·D58219。两分钟内让人出来帮我挪车开路，并对无障碍通道被堵正式创建投诉工单严肃整改！"
                fixed_details.append("样本 #50 强化投诉创建语义")
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "浙A·D58219",
                    "issue_summary": "客户投诉服务中心东侧无障碍坡道被试驾车严重阻挡，要求挪车开路并立案整改"
                }, ensure_ascii=False)

    elif s_id == "cat7_val_003_var0":
        # 样本 51: 误导性捆绑延保
        for msg in history:
            if msg.get("role") == "user" and "那个顾问非说如果不买这个延保套餐就不给保养" in msg.get("content", ""):
                msg["content"] = "喂！我车刚提回来没两天，车架号是 LSVAA123456789012。那个顾问非说如果不买延保套餐就不给保养，这明显是欺诈强买强卖吧？你们赶紧给我正式创建投诉工单立案调查！"
                fixed_details.append("样本 #51 补齐用户 VIN 并明确立案投诉意图")
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVAA123456789012",
                    "issue_summary": "车主投诉服务顾问强制捆绑延保否则不提供保养，要求立案督办"
                }, ensure_ascii=False)

    elif s_id == "cat7_val_004_var0":
        # 样本 52: 手写收据与发票
        for msg in history:
            if msg.get("role") == "user" and "收据照片和转账记录都在我手机里" in msg.get("content", ""):
                msg["content"] = "车架号是 LSVG5288921890123，收据和转账记录都在我手机里！马上核实这笔 860 元费用并补开正式发票，对门店乱开手写收据行为正式创建投诉工单彻查！"
                fixed_details.append("样本 #52 补齐用户 VIN 与投诉工单诉求")
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVG5288921890123",
                    "issue_summary": "车主投诉城南直营店电路检修开具手写收据未给正式发票，要求核查票据真实性"
                }, ensure_ascii=False)

    elif s_id == "cat7_val_005_var0":
        # 样本 53: 未授权试车里程增加
        for msg in history:
            if msg.get("role") == "user" and "VIN 是 LFV3A23K8N3001234" in msg.get("content", ""):
                msg["content"] = "VIN 是 LFV3A23K8N3001234，是在北京朝阳大悦城店做的保养。手机 App 显示车子被私开出去了40公里，你们必须正式立案投诉并调取行车轨迹彻查！"
                fixed_details.append("样本 #53 强化未授权试车立案投诉意图")
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LFV3A23K8N3001234",
                    "issue_summary": "车主投诉车辆在店保养期间被私自行驶40公里，要求立案调查并封存轨迹"
                }, ensure_ascii=False)

    elif s_id == "cat7_val_006_var0":
        # 样本 54: 油漆提车里程异常增加
        for msg in history:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LSVFC2389PN019882",
                    "issue_summary": "车主投诉提车时仪表盘异常增加85公里，要求立案封存车机日志并调查"
                }, ensure_ascii=False)

    elif s_id == "cat7_val_007_var0":
        # 样本 55: 交付钥匙缺失
        for msg in history:
            if msg.get("role") == "user" and "VIN码是 LFV2A21J5M3000001" in msg.get("content", ""):
                msg["content"] = "VIN码是 LFV2A21J5M3000001，电话是13912345678。新车钥匙少了一把，必须给我正式创建升级投诉工单加急排查！"
                fixed_details.append("样本 #55 强化钥匙缺失升级投诉意图")
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "LFV2A21J5M3000001",
                    "issue_summary": "车主投诉交付时车钥匙缺失一把影响长途行程，要求创建升级投诉督办工单加急查找"
                }, ensure_ascii=False)

    elif s_id == "cat7_val_008_var0":
        # 样本 56: 财物遗失
        for msg in history:
            if msg.get("role") == "user" and "车牌是京A9988X" in msg.get("content", ""):
                msg["content"] = "车牌是京A9988X，今天下午两点十分提车，工单号WO2023102588。扶手箱现金丢失，你们赶紧正式立案投诉并保全封存施工工位监控！"
                fixed_details.append("样本 #56 强化财物争议立案投诉意图")
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": "京A9988X",
                    "issue_summary": "车主投诉保养提车后车内扶手箱现金缺失，要求立案调查并保全封存监控"
                }, ensure_ascii=False)

    # =========================================================================
    # Category 8: 主动关怀、回访与客户运营 (全面补齐用户提问中的 VIN 码并规范参数)
    # =========================================================================
    elif cat_id == 8 or sample.get("category") == "主动关怀、回访与客户运营":
        # 针对样本 57~64 补齐车架号
        vin_map = {
            "cat8_val_001_var0": "LSVAA892345670001",
            "cat8_val_002_var0": "LSVGA139580198760",
            "cat8_val_003_var0": "LVSHFLLC5MS123456",
            "cat8_val_004_var0": "LSVAA202405001234",
            "cat8_val_005_var0": "LSVCC123456789XYZ",
            "cat8_val_006_var0": "LSVAA456789123456",
            "cat8_val_007_var0": "LSVAU2A39MN123456",
            "cat8_val_008_var0": "LSVEE202311009876"
        }
        vin_to_use = vin_map.get(s_id, "LSVAA888899990001")

        # 处理首轮用户消息，补齐 VIN
        user_msg = history[1] if len(history) > 1 and history[1]["role"] == "user" else None
        if user_msg and vin_to_use not in user_msg["content"]:
            user_msg["content"] = f"我车架号是 {vin_to_use}。" + user_msg["content"]
            fixed_details.append(f"样本 #{idx} ({s_id}) 补齐用户 VIN: {vin_to_use}")

        # 规范工具调用参数统一为 vehicle_id
        for msg in history:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"][0]["function"]["arguments"] = json.dumps({
                    "vehicle_id": vin_to_use
                }, ensure_ascii=False)

# 保存清洗结果
with open(EVAL_PATH, "w", encoding="utf-8") as f:
    for item in samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n🎉 验证集数据全量修复完成！共修复/优化 {len(fixed_details)} 处：")
for d in fixed_details:
    print(f"  • {d}")
