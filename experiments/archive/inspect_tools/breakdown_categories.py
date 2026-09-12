import json
import sys
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

log_path = r"C:\Users\Acer\.gemini\antigravity-ide\brain\dafc0d60-56e0-4b9a-8aef-de8f6fa7a056\.system_generated\tasks\task-686.log"

with open(log_path, "r", encoding="utf-8") as f:
    text = f.read()

# 提取所有样本块
samples_raw = text.split("───────────────────────────────────────────────────────────────────────────────────────────\n📌 [样本 ")

cat_stats = {}

for block in samples_raw[1:]:
    lines = block.split("\n")
    header = lines[0]
    # 提取 ID 和 业务场景
    m_cat = re.search(r"业务场景:\s*([^\n\r]+)", header)
    cat_name = m_cat.group(1).strip() if m_cat else "未知分类"

    if cat_name not in cat_stats:
        cat_stats[cat_name] = {"tool_turns": 0, "top1": 0, "top3": 0, "top5": 0, "miss": 0}

    # 扫描其中的每个轮次
    for line in lines:
        if "真实意图:" in line and "【none】" not in line:
            cat_stats[cat_name]["tool_turns"] += 1
            if "Top-1 命中!" in line:
                cat_stats[cat_name]["top1"] += 1
                cat_stats[cat_name]["top3"] += 1
                cat_stats[cat_name]["top5"] += 1
            elif "Top-3 召回!" in line:
                cat_stats[cat_name]["top3"] += 1
                cat_stats[cat_name]["top5"] += 1
            elif "Top-5 召回!" in line:
                cat_stats[cat_name]["top5"] += 1
            else:
                cat_stats[cat_name]["miss"] += 1

print("=" * 80)
print(f"{'业务场景类别':<22} | {'调用轮次':<6} | {'Top-1':<7} | {'Top-3':<7} | {'Top-5':<7}")
print("-" * 80)

total_turns = 0
total_top1 = 0
total_top3 = 0
total_top5 = 0

for cat, s in cat_stats.items():
    tot = s["tool_turns"]
    if tot > 0:
        t1_rate = f"{s['top1']/tot*100:.1f}%"
        t3_rate = f"{s['top3']/tot*100:.1f}%"
        t5_rate = f"{s['top5']/tot*100:.1f}%"
    else:
        t1_rate = t3_rate = t5_rate = "N/A"
    print(f"{cat:<20} | {tot:<8} | {t1_rate:<7} | {t3_rate:<7} | {t5_rate:<7}")
    total_turns += tot
    total_top1 += s["top1"]
    total_top3 += s["top3"]
    total_top5 += s["top5"]

print("=" * 80)
print(f"{'总体平均':<22} | {total_turns:<8} | {total_top1/total_turns*100:.1f}%  | {total_top3/total_turns*100:.1f}%  | {total_top5/total_turns*100:.1f}%")
