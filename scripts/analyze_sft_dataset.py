"""
SFT 数据集统计分析与可视化脚本 (SFT Dataset Profiler & Visualizer)
统计指标：
1. 全局 System / User / Assistant / Tool 各角色单句长度分布 (均值/中位数/分位数/极值)
2. 单条多轮对话【总字符长度 / Token 长度】精细直方图与分位数分布
3. 8 大场景各自的各角色单句长度与整条对话总长对比
4. 多轮对话总轮数与回合数分布
5. 生成 6 合 1 多维度统计可视化图表仪表板 (PNG)
"""

import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置中文字体与样式，避免乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="deep", font="Microsoft YaHei")

SFT_DIR = "data/v2/sft"
CATEGORY_NAMES = {
    1: "用车与智能功能",
    2: "服务政策与权益",
    3: "故障初判与技术",
    4: "维保预约与接待",
    5: "维修进度与交车",
    6: "救援与保险协同",
    7: "抱怨与投诉处理",
    8: "老客关怀与运营",
}


def load_all_sft_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载全部 SFT 数据并构建两个分析 DataFrame：
    1. df_dialogues: 每条完整对话的样本级统计 (样本ID, 场景ID, 场景名, 消息总数, 用户轮数, 包含Tool数, 对话总字符长, 估算Tokens)
    2. df_sentences: 逐条单句级统计 (样本ID, 场景ID, 场景名, 角色Role, 文本长度, 是否含ToolCall)
    """
    dialogue_records = []
    sentence_records = []

    for cat_id in range(1, 9):
        fpath = os.path.join(SFT_DIR, f"category{cat_id}_sft_dataset.jsonl")
        if not os.path.exists(fpath):
            continue

        cat_name = CATEGORY_NAMES.get(cat_id, f"Category {cat_id}")

        with open(fpath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                seed_id = item.get("seed_id", f"cat{cat_id}_{line_idx}")
                messages = item.get("messages", [])

                total_char_len = 0
                role_counts = Counter()
                tool_call_count = 0

                for msg_idx, msg in enumerate(messages):
                    role = msg.get("role")
                    content = msg.get("content") or ""
                    tool_calls = msg.get("tool_calls")

                    # 若 assistant 为 tool_calls，将其 arguments JSON 计入有效输出长度
                    if role == "assistant" and tool_calls:
                        tool_call_count += len(tool_calls)
                        args_text = "".join(
                            tc.get("function", {}).get("arguments", "") for tc in tool_calls
                        )
                        sentence_len = len(content) + len(args_text)
                    else:
                        sentence_len = len(str(content))

                    total_char_len += sentence_len
                    role_counts[role] += 1

                    sentence_records.append({
                        "seed_id": seed_id,
                        "category_id": cat_id,
                        "category_name": cat_name,
                        "role": role,
                        "length": sentence_len,
                        "msg_index": msg_idx
                    })

                dialogue_records.append({
                    "seed_id": seed_id,
                    "category_id": cat_id,
                    "category_name": cat_name,
                    "msg_count": len(messages),
                    "user_turns": role_counts["user"],
                    "assistant_turns": role_counts["assistant"],
                    "tool_turns": role_counts["tool"],
                    "tool_calls_count": tool_call_count,
                    "total_length": total_char_len,
                    "est_tokens": int(total_char_len * 0.7)  # 中英文混合估算 1字约0.7 token
                })

    return pd.DataFrame(dialogue_records), pd.DataFrame(sentence_records)


def print_statistical_summary(df_diag: pd.DataFrame, df_sent: pd.DataFrame):
    """
    在终端打印详尽的格式化统计报表
    """
    print("=" * 80)
    print("[SFT Dataset Profiler] 智能汽车客服助手 SFT 数据集深度全景统计报表")
    print("=" * 80)
    print(f"1. 覆盖场景总数      : {df_diag['category_id'].nunique()} 个")
    print(f"2. SFT 样本对话总条数: {len(df_diag):,} 条")
    print(f"3. 消息单句总数量    : {len(df_sent):,} 句")
    print(f"4. 涵盖总字符量      : {df_diag['total_length'].sum():,} 字 (估算约 {df_diag['est_tokens'].sum():,} Tokens)")
    print("-" * 80)

    # 1. 全局各角色单句长度分布
    print("【一、 各角色单句长度分布 (Character Length per Message)】")
    roles = ["system", "user", "assistant", "tool"]
    role_summary = []
    for r in roles:
        subset = df_sent[df_sent["role"] == r]["length"]
        if len(subset) > 0:
            role_summary.append({
                "角色 (Role)": r,
                "句子总数": len(subset),
                "均值 (Mean)": f"{subset.mean():.1f}",
                "中位数 (Median)": f"{subset.median():.0f}",
                "标准差 (Std)": f"{subset.std():.1f}",
                "最小 (Min)": subset.min(),
                "P90分位": f"{subset.quantile(0.90):.0f}",
                "最大 (Max)": subset.max(),
            })
    df_role_stat = pd.DataFrame(role_summary)
    print(df_role_stat.to_string(index=False))
    print("-" * 80)

    # 2. 多轮对话总长度区间精细分布
    tot_len = df_diag["total_length"]
    print("【二、 单条多轮对话【总字符长度】精细分布统计 (Dialogue Total Length)】")
    print(f"- 均值 (Mean)       : {tot_len.mean():.1f} 字 (约 {int(tot_len.mean()*0.7)} Tokens)")
    print(f"- 中位数 (Median)   : {tot_len.median():.0f} 字")
    print(f"- 标准差 (Std)      : {tot_len.std():.1f} 字")
    print(f"- 极值范围 [Min,Max]: [{tot_len.min()}, {tot_len.max()}] 字")
    print(f"- 分位数统计        : P50={tot_len.quantile(0.50):.0f}字 | P75={tot_len.quantile(0.75):.0f}字 | P90={tot_len.quantile(0.90):.0f}字 | P95={tot_len.quantile(0.95):.0f}字 | P99={tot_len.quantile(0.99):.0f}字")
    
    # 区间分桶统计
    bins = [0, 500, 800, 1000, 1200, 1500, 2000, 5000]
    labels = ["<500字", "500-800字", "800-1000字", "1000-1200字", "1200-1500字", "1500-2000字", "2000字以上"]
    df_diag["len_bin"] = pd.cut(df_diag["total_length"], bins=bins, labels=labels)
    bin_counts = df_diag["len_bin"].value_counts().reindex(labels)
    
    bin_table = []
    for b_label in labels:
        cnt = bin_counts[b_label]
        ratio = cnt / len(df_diag) * 100
        bin_table.append({"长度区间": b_label, "对话条数": cnt, "占比 (%)": f"{ratio:.2f}%"})
    print("\n长度区间分布频次表:")
    print(pd.DataFrame(bin_table).to_string(index=False))
    print("-" * 80)

    # 3. 对话轮数与工具调用
    print("【三、 多轮对话轮数分布 (Turn Count per Dialogue)】")
    turn_subset = df_diag["msg_count"]
    print(f"- 对话消息轮数 (总条数) : 均值={turn_subset.mean():.2f} 轮 | 中位数={turn_subset.median():.0f} 轮 | 范围=[{turn_subset.min()}, {turn_subset.max()}] 轮")
    print(f"- 用户提问轮数 (User)   : 均值={df_diag['user_turns'].mean():.2f} 轮 | 范围=[{df_diag['user_turns'].min()}, {df_diag['user_turns'].max()}] 轮")
    print(f"- 工具调用率 (Tool Ratio): {(df_diag['tool_turns'] > 0).mean() * 100:.2f}% 的对话涉及工具调用与响应")
    print("-" * 80)

    # 4. 各场景横向对比表
    print("【四、 8 大业务场景各角色平均长度与轮次对比】")
    cat_summary = []
    for cat_id in sorted(df_diag["category_id"].unique()):
        cat_diag = df_diag[df_diag["category_id"] == cat_id]
        cat_sent = df_sent[df_sent["category_id"] == cat_id]
        c_name = CATEGORY_NAMES.get(cat_id, f"Cat{cat_id}")

        u_len = cat_sent[cat_sent["role"] == "user"]["length"].mean()
        a_len = cat_sent[cat_sent["role"] == "assistant"]["length"].mean()
        t_len = cat_sent[cat_sent["role"] == "tool"]["length"].mean() if "tool" in cat_sent["role"].values else 0
        
        cat_summary.append({
            "场景编号": f"Category {cat_id}",
            "场景名称": c_name,
            "样本量": len(cat_diag),
            "平均轮数": f"{cat_diag['msg_count'].mean():.1f} 轮",
            "User平均长": f"{u_len:.1f} 字",
            "Assistant平均长": f"{a_len:.1f} 字",
            "Tool平均长": f"{t_len:.1f} 字",
            "对话平均总长": f"{cat_diag['total_length'].mean():.1f} 字"
        })
    df_cat_stat = pd.DataFrame(cat_summary)
    print(df_cat_stat.to_string(index=False))
    print("=" * 80)


def generate_visualization(df_diag: pd.DataFrame, df_sent: pd.DataFrame, save_path: str = "data/v2/sft/sft_dataset_statistics.png"):
    """
    生成 6 合 1 综合可视化仪表板图 (2 行 3 列高清排版)
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 13), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    cats = [CATEGORY_NAMES[i] for i in range(1, 9)]

    # -------------------------------------------------------------
    # 子图 1: 全局 System / User / Assistant / Tool 单句长度分布箱线图
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    roles_order = ["system", "user", "assistant", "tool"]
    palette = {"system": "#4C72B0", "user": "#55A868", "assistant": "#C44E52", "tool": "#8172B2"}
    
    sns.boxplot(
        data=df_sent,
        x="role",
        y="length",
        order=roles_order,
        hue="role",
        palette=palette,
        legend=False,
        ax=ax1,
        showfliers=False,
        width=0.45
    )
    means = df_sent.groupby("role")["length"].mean()
    for i, r in enumerate(roles_order):
        if r in means:
            ax1.text(i, means[r] + 5, f"均值:{means[r]:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#111")

    ax1.set_title("1. 各角色单句长度分布 (Character Length Boxplot)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("角色 (Role)", fontsize=11)
    ax1.set_ylabel("单句字符长度 (Characters)", fontsize=11)
    ax1.set_xticks(range(len(roles_order)))
    ax1.set_xticklabels(["System (系统提示)", "User (车主提问)", "Assistant (客服回复)", "Tool (工具返回)"])

    # -------------------------------------------------------------
    # 子图 2: 多轮对话【总字符长度】精细直方图与核密度估计曲线 (KDE) —— 【重点新增！】
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    tot_len = df_diag["total_length"]
    
    # 绘制直方图 + KDE 概率曲线
    sns.histplot(
        tot_len,
        bins=35,
        kde=True,
        color="#2b7bba",
        edgecolor="#1a4e75",
        alpha=0.7,
        ax=ax2,
        line_kws={"linewidth": 2.5, "color": "#d9534f"}
    )
    
    # 标记均值、P90、P95 关键统计线
    mean_val = tot_len.mean()
    p90_val = tot_len.quantile(0.90)
    p95_val = tot_len.quantile(0.95)
    
    ax2.axvline(mean_val, color="#c0392b", linestyle="--", linewidth=2, label=f"均值 (Mean): {mean_val:.0f}字")
    ax2.axvline(p90_val, color="#e67e22", linestyle=":", linewidth=2, label=f"P90: {p90_val:.0f}字")
    ax2.axvline(p95_val, color="#8e44ad", linestyle="-.", linewidth=2, label=f"P95: {p95_val:.0f}字")
    
    ax2.set_title("2. 多轮对话【总字符长度】精细分布直方图 (Dialogue Length Histogram)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("整条样本多轮总字符数 (Total Characters per Dialogue)", fontsize=11)
    ax2.set_ylabel("样本频次 (Sample Count)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, fontsize=10)

    # -------------------------------------------------------------
    # 子图 3: 多轮对话总轮数分布柱状图
    # -------------------------------------------------------------
    ax3 = axes[0, 2]
    turn_counts = df_diag["msg_count"].value_counts().sort_index()
    bars = ax3.bar(turn_counts.index, turn_counts.values, color="#3470a3", alpha=0.85, width=0.6, edgecolor="#1f4e79")
    
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight="bold")

    ax3.set_title("3. 多轮对话总轮数分布 (Messages Count Distribution)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("单条样本消息总条数 (Messages per Dialogue)", fontsize=11)
    ax3.set_ylabel("样本数量 (Sample Count)", fontsize=11)
    ax3.set_xticks(turn_counts.index)

    # -------------------------------------------------------------
    # 子图 4: 8 大场景各自多轮对话总长度箱线图对比 —— 【新增场景总长对比！】
    # -------------------------------------------------------------
    ax4 = axes[1, 0]
    sns.boxplot(
        data=df_diag,
        x="category_name",
        y="total_length",
        order=cats,
        palette="Spectral",
        ax=ax4,
        showfliers=False,
        width=0.5
    )
    cat_means = df_diag.groupby("category_name")["total_length"].mean()
    for i, c in enumerate(cats):
        if c in cat_means:
            ax4.text(i, cat_means[c] + 15, f"{cat_means[c]:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax4.set_title("4. 8 大场景：多轮对话总长度分布对比 (Category Total Length)", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("业务场景 (Category)", fontsize=11)
    ax4.set_ylabel("对话总字符数 (Characters)", fontsize=11)
    ax4.set_xticks(range(len(cats)))
    ax4.set_xticklabels(cats, rotation=20, ha="right", fontsize=9.5)

    # -------------------------------------------------------------
    # 子图 5: 8 大场景各自的 User 与 Assistant 平均长度横向柱状图
    # -------------------------------------------------------------
    ax5 = axes[1, 1]
    cat_user_len = df_sent[df_sent["role"] == "user"].groupby("category_name")["length"].mean()
    cat_asst_len = df_sent[df_sent["role"] == "assistant"].groupby("category_name")["length"].mean()
    
    y_pos = np.arange(len(cats))
    height = 0.35

    ax5.barh(y_pos - height/2, [cat_user_len.get(c, 0) for c in cats], height, label="User (客户提问)", color="#55A868", alpha=0.85)
    ax5.barh(y_pos + height/2, [cat_asst_len.get(c, 0) for c in cats], height, label="Assistant (客服回复)", color="#C44E52", alpha=0.85)

    ax5.set_title("5. 8 大场景：客户 vs 客服单句平均长度对比", fontsize=13, fontweight="bold", pad=10)
    ax5.set_xlabel("平均单句字符数 (Avg Characters)", fontsize=11)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(cats, fontsize=9.5)
    ax5.legend(loc="lower right", frameon=True, fontsize=10)

    # -------------------------------------------------------------
    # 子图 6: 8 大场景平均对话轮次与工具调用占比双轴图
    # -------------------------------------------------------------
    ax6 = axes[1, 2]
    cat_avg_turns = [df_diag[df_diag["category_name"] == c]["msg_count"].mean() for c in cats]
    cat_tool_ratio = [
        (df_diag[df_diag["category_name"] == c]["tool_turns"] > 0).mean() * 100
        for c in cats
    ]

    color1 = "#2b5c8f"
    color2 = "#e67e22"

    ax6.bar(y_pos, cat_avg_turns, color=color1, alpha=0.75, width=0.45, label="平均对话轮数 (左轴)")
    ax6.set_ylabel("平均对话轮数 (Turns)", fontsize=11, color=color1)
    ax6.tick_params(axis='y', labelcolor=color1)
    ax6.set_xticks(y_pos)
    ax6.set_xticklabels(cats, rotation=20, ha="right", fontsize=9.5)

    # 双轴绘制工具调用率
    ax6_twin = ax6.twinx()
    ax6_twin.plot(y_pos, cat_tool_ratio, color=color2, marker="o", linewidth=2.5, markersize=8, label="工具调用覆盖率% (右轴)")
    ax6_twin.set_ylabel("工具调用覆盖率 (%)", fontsize=11, color=color2)
    ax6_twin.tick_params(axis='y', labelcolor=color2)
    ax6_twin.set_ylim(0, 105)
    ax6_twin.grid(False)

    ax6.set_title("6. 8 大场景：平均轮数与工具调用覆盖率", fontsize=13, fontweight="bold", pad=10)

    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] 6合1 综合统计可视化大图已成功保存至: {save_path}")


def main():
    os.makedirs("scripts", exist_ok=True)
    df_diag, df_sent = load_all_sft_data()

    if len(df_diag) == 0:
        print(f"错误: 未在 {SFT_DIR} 下找到任何 SFT 数据集文件。")
        return

    # 1. 输出终端全景报告
    print_statistical_summary(df_diag, df_sent)

    # 2. 生成多维度可视化图表
    chart_path = os.path.join(SFT_DIR, "sft_dataset_statistics.png")
    generate_visualization(df_diag, df_sent, save_path=chart_path)


if __name__ == "__main__":
    main()
