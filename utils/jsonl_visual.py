"""
utils/jsonl_visual.py - 多轮对话与模型竞技对决可视化工具
功能：
1. 自动识别输入 JSONL 类型：
   - 模型预测对比文件 (model_predictions.jsonl)：激活【A/B 双栏竞技对决模式】，展示前序切片上下文，并列对比 Baseline 与 SFT 的实际生成，提供 <think> 思考链折叠与黄金标准答案对照
   - 普通多轮数据集 (car_assistant_eval.jsonl, sft_dataset.jsonl)：激活【完整对话河流模式】，展示 system, user, assistant, tool_calls, tool 全角色气泡
2. 单文件自包含 HTML (内嵌 CSS/JS，完全断网离线可用，开箱即用)
3. 实时关键词过滤、分类筛选、键盘快捷键 (↑/↓/J/K) 丝滑切换
"""

import argparse
import html
import json
import os
import sys
import webbrowser
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="多轮对话与模型对决可视化生成工具")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="outputs/car_eval/model_predictions.jsonl",
        help="待可视化的 JSONL 文件路径 (默认: outputs/car_eval/model_predictions.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="生成的 HTML 文件保存路径 (留空时自动保存在 outputs/visualize/ 下)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="生成完成后自动使用默认浏览器打开",
    )
    return parser.parse_args()


def load_jsonl_samples(file_path: str) -> List[Dict[str, Any]]:
    """加载并解析 JSONL 数据，自动识别是否为 A/B 预测切片还是普通对话集"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到输入文件: {file_path}")

    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 判断是否为包含模型预测的 A/B 结果切片
                has_model_preds = "model_a_response" in data or "model_b_response" in data
                data["_is_battle_slice"] = has_model_preds

                if has_model_preds:
                    # 针对切片预测文件：优先提取该切片发生时的前序历史
                    context_history = data.get("history_messages") or []
                    if not context_history and "full_dialog_history" in data:
                        # 兜底截断
                        context_history = data.get("full_dialog_history")
                    data["_context_dialog"] = context_history
                else:
                    # 普通对话集：提取完整会话历史
                    full_dialog = data.get("full_dialog_history") or data.get("messages") or data.get("history") or []
                    if not full_dialog and "query" in data:
                        full_dialog = [
                            {"role": "system", "content": data.get("system_prompt") or data.get("system", "")},
                            {"role": "user", "content": data.get("query", "")},
                        ]
                        if data.get("reference_response"):
                            full_dialog.append({"role": "assistant", "content": data.get("reference_response")})
                    data["_context_dialog"] = full_dialog

                data["_index"] = idx
                samples.append(data)
            except Exception as e:
                print(f"[Warning] 第 {idx} 行解析 JSON 失败: {e}")

    return samples


def generate_html(samples: List[Dict[str, Any]], title: str, source_filename: str) -> str:
    """生成包含 A/B 竞技对决视图与全对话视图的现代交互式 HTML"""
    json_data_str = json.dumps(samples, ensure_ascii=False)
    is_battle_file = any(s.get("_is_battle_slice", False) for s in samples)

    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)} - 客服大模型对决与对话可视化</title>
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-secondary: #111827;
            --bg-card: #1f2937;
            --bg-hover: #374151;
            --text-main: #f9fafb;
            --text-sub: #9ca3af;
            --text-muted: #6b7280;
            --border: #374151;

            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --accent-purple: #8b5cf6;

            --base-a-color: #f87171;
            --base-a-bg: rgba(239, 68, 68, 0.08);
            --base-a-border: rgba(239, 68, 68, 0.35);

            --sft-b-color: #34d399;
            --sft-b-bg: rgba(16, 185, 129, 0.08);
            --sft-b-border: rgba(16, 185, 129, 0.35);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        /* 顶部 Header */
        header {{
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            z-index: 20;
        }}
        .header-left {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .header-title {{
            font-size: 1.15rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .mode-badge {{
            font-size: 0.75rem;
            padding: 2px 10px;
            border-radius: 9999px;
            font-weight: 600;
        }}
        .badge-battle {{
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.4);
        }}
        .badge-dialog {{
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.4);
        }}
        .header-stats {{
            font-size: 0.85rem;
            color: var(--text-sub);
        }}

        /* 主视口布局 */
        .main-container {{
            flex: 1;
            display: flex;
            overflow: hidden;
        }}

        /* 左侧边栏 */
        .sidebar {{
            width: 380px;
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }}
        .search-area {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .search-input {{
            width: 100%;
            background-color: var(--bg-primary);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            outline: none;
            transition: border-color 0.2s;
        }}
        .search-input:focus {{
            border-color: #38bdf8;
        }}
        .filter-select {{
            width: 100%;
            background-color: var(--bg-primary);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            outline: none;
        }}

        .sample-list {{
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .sample-card {{
            padding: 12px 14px;
            border-radius: 8px;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            cursor: pointer;
            transition: all 0.15s ease-in-out;
        }}
        .sample-card:hover {{
            background: var(--bg-card);
            border-color: #4b5563;
        }}
        .sample-card.active {{
            background: #112240;
            border-color: #38bdf8;
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
        }}
        .card-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        .card-id {{
            font-size: 0.78rem;
            font-family: monospace;
            color: #38bdf8;
            font-weight: 600;
        }}
        .card-turn-tag {{
            font-size: 0.7rem;
            background: rgba(139, 92, 246, 0.2);
            color: #c084fc;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }}
        .card-scenario {{
            font-size: 0.88rem;
            color: #f3f4f6;
            font-weight: 500;
            margin-bottom: 4px;
            line-height: 1.3;
        }}
        .card-query-preview {{
            font-size: 0.76rem;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        /* 右侧主展示区 */
        .content-panel {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            background-color: var(--bg-primary);
        }}

        /* 顶部元数据与质检规则看板 */
        .meta-header {{
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 16px 28px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .tag-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }}
        .tag {{
            font-size: 0.75rem;
            padding: 3px 10px;
            border-radius: 6px;
            font-weight: 500;
        }}
        .tag-cat {{ background: rgba(2, 132, 199, 0.2); color: #38bdf8; border: 1px solid rgba(2, 132, 199, 0.4); }}
        .tag-role {{ background: rgba(139, 92, 246, 0.2); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.4); }}
        .tag-var {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .tag-tool {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}

        .meta-rules-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 10px;
        }}
        .rule-box {{
            background: var(--bg-primary);
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid var(--border);
            font-size: 0.8rem;
        }}
        .rule-title {{
            font-weight: 600;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .title-fact {{ color: #34d399; }}
        .title-quest {{ color: #a78bfa; }}
        .title-act {{ color: #60a5fa; }}
        .title-prohib {{ color: #f87171; }}
        .rule-content {{
            color: var(--text-sub);
            line-height: 1.4;
        }}

        /* 核心内容容器 */
        .viewport-body {{
            padding: 24px 32px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            max-width: 1280px;
            margin: 0 auto;
            width: 100%;
        }}

        /* 折叠式前序历史容器 */
        .history-accordion {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }}
        .accordion-header {{
            padding: 12px 18px;
            background: var(--bg-card);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            font-size: 0.88rem;
            color: #93c5fd;
            user-select: none;
        }}
        .accordion-header:hover {{
            background: #283548;
        }}
        .accordion-content {{
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}

        /* 气泡样式 */
        .bubble-row {{
            display: flex;
            gap: 12px;
            width: 100%;
            align-items: flex-start;
        }}
        .bubble-avatar {{
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
        }}
        .bubble-user-wrap {{
            justify-content: flex-end;
        }}
        .bubble-user-wrap .bubble-body {{
            background: #2563eb;
            color: white;
            border-bottom-right-radius: 2px;
        }}
        .bubble-assistant-wrap .bubble-body {{
            background: var(--bg-card);
            border: 1px solid #4b5563;
            color: #f3f4f6;
            border-bottom-left-radius: 2px;
        }}
        .bubble-system-wrap {{
            justify-content: center;
        }}
        .bubble-system-wrap .bubble-body {{
            background: #1e2433;
            border: 1px dashed rgba(59, 130, 246, 0.3);
            color: #93c5fd;
            font-size: 0.82rem;
            max-width: 90%;
        }}
        .bubble-body {{
            padding: 10px 14px;
            border-radius: 10px;
            font-size: 0.88rem;
            line-height: 1.55;
            max-width: 85%;
            word-break: break-word;
            white-space: pre-wrap;
        }}

        /* 工具调用卡片 */
        .tool-card {{
            background: #1b160d;
            border: 1px solid #d97706;
            color: #fde68a;
            border-radius: 8px;
            padding: 10px 14px;
            font-family: monospace;
            font-size: 0.82rem;
        }}
        .tool-resp-card {{
            background: #0d221e;
            border: 1px solid #059669;
            color: #a7f3d0;
            border-radius: 8px;
            padding: 10px 14px;
            font-family: monospace;
            font-size: 0.82rem;
        }}

        /* ================================================================= */
        /* A/B 竞技对决专属看板样式 */
        /* ================================================================= */
        .battle-arena-section {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .arena-divider {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 700;
            font-size: 0.95rem;
            color: #f59e0b;
        }}
        .arena-divider::before, .arena-divider::after {{
            content: "";
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, transparent, #4b5563, transparent);
        }}

        .current-prompt-box {{
            background: #172554;
            border: 1px solid #3b82f6;
            border-radius: 8px;
            padding: 14px 18px;
            color: #dbeafe;
            font-size: 0.92rem;
        }}
        .current-prompt-box strong {{
            color: #60a5fa;
            margin-right: 6px;
        }}

        /* 双栏竞技对决布局 */
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }}
        @media (max-width: 900px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .model-col {{
            display: flex;
            flex-direction: column;
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }}
        .col-header {{
            padding: 12px 18px;
            font-weight: 700;
            font-size: 0.92rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .col-header-a {{
            background: var(--base-a-bg);
            border-bottom: 1px solid var(--base-a-border);
            color: var(--base-a-color);
        }}
        .col-header-b {{
            background: var(--sft-b-bg);
            border-bottom: 1px solid var(--sft-b-border);
            color: var(--sft-b-color);
        }}
        .col-content {{
            background: var(--bg-secondary);
            padding: 18px;
            flex: 1;
            font-size: 0.92rem;
            line-height: 1.65;
            word-break: break-word;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        /* 思考过程折叠卡片 */
        .think-collapse {{
            background: #182030;
            border: 1px solid #2d3748;
            border-radius: 6px;
            font-size: 0.82rem;
            overflow: hidden;
        }}
        .think-header {{
            padding: 8px 12px;
            background: #1f293d;
            cursor: pointer;
            color: #94a3b8;
            display: flex;
            justify-content: space-between;
            user-select: none;
            font-family: monospace;
        }}
        .think-header:hover {{
            color: #cbd5e1;
        }}
        .think-body {{
            padding: 10px 14px;
            color: #94a3b8;
            line-height: 1.5;
            white-space: pre-wrap;
            border-top: 1px solid #2d3748;
            max-height: 250px;
            overflow-y: auto;
        }}

        .response-text {{
            white-space: pre-wrap;
            color: #f3f4f6;
        }}

        /* 黄金参考答案卡片 */
        .ground-truth-box {{
            background: #181926;
            border: 1px solid #b45309;
            border-radius: 8px;
            overflow: hidden;
            font-size: 0.88rem;
        }}
        .gt-header {{
            padding: 10px 16px;
            background: rgba(180, 83, 9, 0.2);
            color: #fbbf24;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            user-select: none;
        }}
        .gt-body {{
            padding: 14px 18px;
            color: #fde68a;
            white-space: pre-wrap;
            line-height: 1.6;
        }}

        /* 强调高亮与Markdown转换 */
        strong {{
            color: #38bdf8;
            font-weight: 600;
        }}
    </style>
</head>
<body>

    <header>
        <div class="header-left">
            <h1 class="header-title">🚗 智能汽车多轮对话与模型评测竞技场</h1>
            <span class="mode-badge { 'badge-battle' if is_battle_file else 'badge-dialog' }">
                { '⚔️ A/B 对决评测模式 (Battle Arena)' if is_battle_file else '💬 完整多轮会话模式' }
            </span>
            <span style="font-size:0.75rem;background:#1e293b;padding:2px 8px;border-radius:4px;color:#94a3b8;">
                {html.escape(source_filename)}
            </span>
        </div>
        <div class="header-stats" id="stats-summary">
            加载中...
        </div>
    </header>

    <div class="main-container">
        <!-- 左侧样本筛选与列表 -->
        <div class="sidebar">
            <div class="search-area">
                <input type="text" id="search-input" class="search-input" placeholder="🔍 搜索场景 / 关键词 / 轮次...">
                <select id="category-filter" class="filter-select">
                    <option value="">全部场景类别 (All Categories)</option>
                </select>
            </div>
            <div class="sample-list" id="sample-list">
                <!-- 动态列表项 -->
            </div>
        </div>

        <!-- 右侧核心展示面板 -->
        <div class="content-panel" id="content-panel">
            <!-- 动态渲染选中切片/对话 -->
        </div>
    </div>

    <script>
        const SAMPLES = {json_data_str};
        let currentIndex = 0;
        let filteredIndices = SAMPLES.map((_, i) => i);

        window.addEventListener('DOMContentLoaded', () => {{
            populateCategories();
            renderSampleList();
            if (filteredIndices.length > 0) {{
                selectSample(filteredIndices[0]);
            }} else {{
                renderEmpty();
            }}

            document.getElementById('search-input').addEventListener('input', applyFilter);
            document.getElementById('category-filter').addEventListener('change', applyFilter);

            // 键盘快捷键支持 (上下或 J/K 极速翻页)
            window.addEventListener('keydown', (e) => {{
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
                const curPos = filteredIndices.indexOf(currentIndex);
                if (e.key === 'ArrowDown' || e.key === 'j') {{
                    if (curPos < filteredIndices.length - 1) {{
                        selectSample(filteredIndices[curPos + 1]);
                    }}
                }} else if (e.key === 'ArrowUp' || e.key === 'k') {{
                    if (curPos > 0) {{
                        selectSample(filteredIndices[curPos - 1]);
                    }}
                }}
            }});
        }});

        function populateCategories() {{
            const filterEl = document.getElementById('category-filter');
            const categories = new Set();
            SAMPLES.forEach(s => {{
                if (s.category) categories.add(s.category);
            }});
            categories.forEach(cat => {{
                const opt = document.createElement('option');
                opt.value = cat;
                opt.textContent = cat;
                filterEl.appendChild(opt);
            }});
        }}

        function applyFilter() {{
            const searchTxt = document.getElementById('search-input').value.toLowerCase().trim();
            const catVal = document.getElementById('category-filter').value;

            filteredIndices = [];
            SAMPLES.forEach((s, idx) => {{
                const matchCat = !catVal || s.category === catVal;
                const matchSearch = !searchTxt ||
                    (s.id && s.id.toLowerCase().includes(searchTxt)) ||
                    (s.slice_id && s.slice_id.toLowerCase().includes(searchTxt)) ||
                    (s.scenario && s.scenario.toLowerCase().includes(searchTxt)) ||
                    (s.query && s.query.toLowerCase().includes(searchTxt)) ||
                    (s.model_a_response && s.model_a_response.toLowerCase().includes(searchTxt)) ||
                    (s.model_b_response && s.model_b_response.toLowerCase().includes(searchTxt)) ||
                    (s.subcategory && s.subcategory.toLowerCase().includes(searchTxt));

                if (matchCat && matchSearch) {{
                    filteredIndices.push(idx);
                }}
            }});

            renderSampleList();
            if (filteredIndices.length > 0) {{
                if (!filteredIndices.includes(currentIndex)) {{
                    selectSample(filteredIndices[0]);
                }} else {{
                    highlightActiveItem();
                }}
            }} else {{
                renderEmpty();
            }}
        }}

        function renderSampleList() {{
            const listEl = document.getElementById('sample-list');
            listEl.innerHTML = '';

            document.getElementById('stats-summary').textContent =
                `当前展示: ${{filteredIndices.length}} / ${{SAMPLES.length}} 条`;

            filteredIndices.forEach(idx => {{
                const s = SAMPLES[idx];
                const itemEl = document.createElement('div');
                itemEl.className = 'sample-card' + (idx === currentIndex ? ' active' : '');
                itemEl.id = 'sample-item-' + idx;

                const isBattle = s._is_battle_slice;
                const turnTag = (isBattle && s.turn_index) 
                    ? `<span class="card-turn-tag">Turn ${{s.turn_index}}/${{s.total_turns || '?'}}</span>`
                    : `<span class="card-turn-tag">${{(s._context_dialog || []).length}} 轮</span>`;

                itemEl.innerHTML = `
                    <div class="card-top">
                        <span class="card-id">#${{s._index}} ${{s.slice_id || s.id || 'sample_' + idx}}</span>
                        ${{turnTag}}
                    </div>
                    <div class="card-scenario">${{escapeHtml(s.scenario || s.subcategory || '客服场景问答')}}</div>
                    <div class="card-query-preview">${{escapeHtml(s.current_turn_query || s.query || '')}}</div>
                `;

                itemEl.addEventListener('click', () => selectSample(idx));
                listEl.appendChild(itemEl);
            }});
        }}

        function selectSample(idx) {{
            currentIndex = idx;
            highlightActiveItem();
            renderContent(SAMPLES[idx]);
        }}

        function highlightActiveItem() {{
            document.querySelectorAll('.sample-card').forEach(el => el.classList.remove('active'));
            const activeEl = document.getElementById('sample-item-' + currentIndex);
            if (activeEl) {{
                activeEl.classList.add('active');
                activeEl.scrollIntoView({{ block: 'nearest' }});
            }}
        }}

        function renderContent(s) {{
            const contentArea = document.getElementById('content-panel');
            const isBattle = s._is_battle_slice;

            // 1. 顶部质检标准与场景画像
            let metaHtml = '';
            const facts = s.required_facts || [];
            const questions = s.required_questions || [];
            const actions = s.required_actions || [];
            const prohibited = s.prohibited_actions || [];

            metaHtml = `
                <div class="meta-header">
                    <div class="tag-row">
                        ${{s.category ? `<span class="tag tag-cat">📁 ${{escapeHtml(s.category)}} / ${{escapeHtml(s.subcategory || '')}}</span>` : ''}}
                        ${{s.customer_role ? `<span class="tag tag-role">👤 ${{escapeHtml(s.customer_role)}}</span>` : ''}}
                        ${{s.variation_name ? `<span class="tag tag-var">⚡ ${{escapeHtml(s.variation_name)}}</span>` : ''}}
                        ${{s.tool_required ? `<span class="tag tag-tool">🛠️ 必需调用工具: ${{escapeHtml(s.tool_name || '内置工具')}}</span>` : ''}}
                    </div>
                    ${{s.user_goal ? `<div style="font-size:0.88rem;color:#e2e8f0;"><strong>🎯 车主核心诉求：</strong>${{escapeHtml(s.user_goal)}}</div>` : ''}}
                    
                    ${{(facts.length || questions.length || actions.length || prohibited.length) ? `
                    <div class="meta-rules-grid">
                        ${{facts.length ? `<div class="rule-box"><div class="rule-title title-fact">📌 核心事实 (Facts)</div><div class="rule-content">${{escapeHtml(facts.join('； '))}}</div></div>` : ''}}
                        ${{questions.length ? `<div class="rule-box"><div class="rule-title title-quest">❓ 主动追问 (Questions)</div><div class="rule-content">${{escapeHtml(questions.join('； '))}}</div></div>` : ''}}
                        ${{actions.length ? `<div class="rule-box"><div class="rule-title title-act">✅ 指导动作 (Actions)</div><div class="rule-content">${{escapeHtml(actions.join('； '))}}</div></div>` : ''}}
                        ${{prohibited.length ? `<div class="rule-box"><div class="rule-title title-prohib">⛔ 禁止行为 (Prohibited)</div><div class="rule-content" style="color:#fca5a5;">${{escapeHtml(prohibited.join('； '))}}</div></div>` : ''}}
                    </div>` : ''}}
                </div>
            `;

            let mainBodyHtml = '<div class="viewport-body">';

            if (isBattle) {{
                // ============================================================
                // 模式 1：A/B 竞技对决模式 (展示前序上下文 + 两位选手回答并列对比)
                // ============================================================
                const prevContext = s._context_dialog || [];

                // 1.1 前序标准上下文折叠面板
                if (prevContext.length > 0) {{
                    mainBodyHtml += `
                        <div class="history-accordion">
                            <div class="accordion-header" onclick="toggleAccordion('prev-context-body')">
                                <span>📜 本切片前序标准上下文 (Teacher-Forced Ground Truth Context: ${{prevContext.length}} 条先验消息)</span>
                                <span id="prev-context-toggle-icon">▼ 点击收起</span>
                            </div>
                            <div class="accordion-content" id="prev-context-body">
                                ${{renderDialogStream(prevContext)}}
                            </div>
                        </div>
                    `;
                }}

                // 1.2 本切片评测提问核心卡片
                const currQuery = s.current_turn_query || s.query || '';
                mainBodyHtml += `
                    <div class="battle-arena-section">
                        <div class="arena-divider">⚔️ 评测切片点对决：第 ${{s.turn_index || 1}} / ${{s.total_turns || 1}} 轮问答</div>
                        
                        <div class="current-prompt-box">
                            <strong>🧑 车主本轮触发提问：</strong>
                            ${{escapeHtml(currQuery)}}
                        </div>

                        <!-- A/B 并列对决栏 -->
                        <div class="comparison-grid">
                            <!-- 选手 A: Baseline 基座模型 -->
                            <div class="model-col" style="border-color: var(--base-a-border);">
                                <div class="col-header col-header-a">
                                    <span>选手 A: ${{escapeHtml(s.model_a_name || 'Baseline 基座模型')}}</span>
                                    <span style="font-size:0.75rem;opacity:0.8;">(基座原生回答)</span>
                                </div>
                                <div class="col-content">
                                    ${{renderModelResponse(s.model_a_response)}}
                                </div>
                            </div>

                            <!-- 选手 B: SFT 微调模型 -->
                            <div class="model-col" style="border-color: var(--sft-b-border);">
                                <div class="col-header col-header-b">
                                    <span>选手 B: ${{escapeHtml(s.model_b_name || 'SFT 汽车客服模型')}}</span>
                                    <span style="font-size:0.75rem;opacity:0.8;">✨ (微调目标)</span>
                                </div>
                                <div class="col-content">
                                    ${{renderModelResponse(s.model_b_response)}}
                                </div>
                            </div>
                        </div>

                        <!-- 黄金参考答案 (Ground Truth) -->
                        ${{s.reference_response ? `
                            <div class="ground-truth-box">
                                <div class="gt-header" onclick="toggleAccordion('gt-body-wrap')">
                                    <span>🎯 原厂标准答案 (Ground Truth Reference)</span>
                                    <span id="gt-body-wrap-icon">▼ 展开/收起</span>
                                </div>
                                <div class="gt-body" id="gt-body-wrap">
                                    ${{escapeHtml(s.reference_response)}}
                                </div>
                            </div>
                        ` : ''}}
                    </div>
                `;
            }} else {{
                // ============================================================
                // 模式 2：全量纯多轮会话河流模式
                // ============================================================
                const fullDialog = s._context_dialog || [];
                mainBodyHtml += `
                    <div style="display:flex;flex-direction:column;gap:16px;">
                        <h3 style="color:#38bdf8;font-size:1rem;">💬 完整人机多轮会话河流 (${{fullDialog.length}} 条交互)</h3>
                        <div style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:10px;padding:20px;display:flex;flex-direction:column;gap:16px;">
                            ${{renderDialogStream(fullDialog)}}
                        </div>
                    </div>
                `;
            }}

            mainBodyHtml += '</div>';
            contentArea.innerHTML = metaHtml + mainBodyHtml;
        }}

        // 渲染对话气泡序列
        function renderDialogStream(messages) {{
            let htmlStr = '';
            messages.forEach(msg => {{
                const role = msg.role;
                const content = msg.content || '';
                const toolCalls = msg.tool_calls;

                if (role === 'system') {{
                    htmlStr += `
                        <div class="bubble-row bubble-system-wrap">
                            <div class="bubble-body">
                                <strong>⚙️ System Prompt:</strong> ${{escapeHtml(content)}}
                            </div>
                        </div>
                    `;
                }} else if (role === 'user') {{
                    htmlStr += `
                        <div class="bubble-row bubble-user-wrap">
                            <div class="bubble-body">
                                <div style="font-size:0.75rem;opacity:0.8;margin-bottom:2px;text-align:right;">车主 (User)</div>
                                ${{escapeHtml(content)}}
                            </div>
                            <div class="bubble-avatar" style="background:#2563eb;color:white;">🧑</div>
                        </div>
                    `;
                }} else if (role === 'assistant') {{
                    htmlStr += `
                        <div class="bubble-row bubble-assistant-wrap">
                            <div class="bubble-avatar" style="background:#0284c7;color:white;">🤖</div>
                            <div style="display:flex;flex-direction:column;gap:6px;max-width:85%;">
                                <div style="font-size:0.75rem;color:#9ca3af;">官方客服 (Assistant)</div>
                                ${{content ? `<div class="bubble-body">${{formatMarkdownText(content)}}</div>` : ''}}
                                ${{toolCalls ? `
                                    <div class="tool-card">
                                        <div style="color:#f59e0b;font-weight:600;margin-bottom:4px;">⚡ 发起工具调用 (Tool Call)</div>
                                        <pre style="margin:0;overflow-x:auto;">${{escapeHtml(JSON.stringify(toolCalls, null, 2))}}</pre>
                                    </div>
                                ` : ''}}
                            </div>
                        </div>
                    `;
                }} else if (role === 'tool') {{
                    htmlStr += `
                        <div class="bubble-row" style="padding-left:46px;">
                            <div class="tool-resp-card" style="max-width:90%;">
                                <div style="color:#10b981;font-weight:600;margin-bottom:4px;">📦 系统工具返回 [${{escapeHtml(msg.name || 'query_response')}}]</div>
                                <pre style="margin:0;overflow-x:auto;">${{escapeHtml(tryFormatJson(content))}}</pre>
                            </div>
                        </div>
                    `;
                }}
            }});
            return htmlStr;
        }}

        // 渲染选手模型回答 (优雅处理 <think> 思考链标签与 Markdown 格式)
        function renderModelResponse(rawText) {{
            if (!rawText) return '<span style="color:#6b7280;font-style:italic;">(无回答输出)</span>';

            let thinkContent = '';
            let mainContent = rawText;

            // 提取 <think> ... </think> 思考链
            const thinkMatch = rawText.match(/<think>([\\s\\S]*?)<\\/think>/);
            if (thinkMatch) {{
                thinkContent = thinkMatch[1].trim();
                mainContent = rawText.replace(/<think>[\\s\\S]*?<\\/think>/, '').trim();
            }}

            let outHtml = '';

            // 如果有思考过程，渲染为折叠卡片
            if (thinkContent) {{
                const thinkId = 'think-' + Math.random().toString(36).substr(2, 9);
                outHtml += `
                    <div class="think-collapse">
                        <div class="think-header" onclick="toggleAccordion('${{thinkId}}')">
                            <span>💭 思考过程 (Chain of Thought / ${{thinkContent.length}} 字符)</span>
                            <span id="${{thinkId}}-icon">▼ 点击收起</span>
                        </div>
                        <div class="think-body" id="${{thinkId}}">${{escapeHtml(thinkContent)}}</div>
                    </div>
                `;
            }}

            // 剩余正式回答
            outHtml += `<div class="response-text">${{formatMarkdownText(mainContent)}}</div>`;
            return outHtml;
        }}

        // 折叠/展开辅助函数
        function toggleAccordion(id) {{
            const el = document.getElementById(id);
            const icon = document.getElementById(id + '-icon');
            if (!el) return;
            if (el.style.display === 'none') {{
                el.style.display = '';
                if (icon) icon.textContent = '▼ 点击收起';
            }} else {{
                el.style.display = 'none';
                if (icon) icon.textContent = '▶ 点击展开';
            }}
        }}

        function renderEmpty() {{
            document.getElementById('content-panel').innerHTML = `
                <div style="margin:auto;text-align:center;color:#6b7280;padding:50px;">
                    <h2>未检索到匹配的样本</h2>
                    <p style="margin-top:10px;">请调整搜索关键字或场景下拉筛选条件</p>
                </div>
            `;
        }}

        function escapeHtml(str) {{
            if (!str) return '';
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }}

        function formatMarkdownText(text) {{
            if (!text) return '';
            let s = escapeHtml(text);
            // 加粗转换 **text**
            s = s.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
            return s;
        }}

        function tryFormatJson(str) {{
            if (!str) return '';
            try {{
                const obj = typeof str === 'string' ? JSON.parse(str) : str;
                return JSON.stringify(obj, null, 2);
            }} catch (e) {{
                return str;
            }}
        }}
    </script>
</body>
</html>
"""
    return html_template


def main():
    args = parse_args()
    print("=" * 70)
    print("智能汽车客服助手 - 多轮对话与模型评测竞技场可视化生成器")
    print(f"输入文件: {args.input}")
    print("=" * 70)

    samples = load_jsonl_samples(args.input)
    print(f"成功加载 {len(samples)} 条样本数据！")

    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join("outputs", "visualize", f"{base_name}_visual.html")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    html_content = generate_html(
        samples=samples,
        title=os.path.basename(args.input),
        source_filename=os.path.basename(args.input),
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ 可视化 HTML 文件已生成: {os.path.abspath(args.output)}")
    print("💡 提示: 现已全面激活 A/B 竞技对决双栏模式，直接双击 HTML 即可对比 Baseline 与 SFT 的实际表现！")

    if args.open:
        abs_path = os.path.abspath(args.output)
        print(f"🌐 正在自动打开浏览器: {abs_path} ...")
        webbrowser.open(f"file:///{abs_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
