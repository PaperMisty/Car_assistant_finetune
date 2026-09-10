"""
utils/jsonl_visual.py - 多轮对话与质检 JSONL 可视化工具
功能：
1. 读取任意包含对话历史的 JSONL 文件 (如 custom_eval/car_assistant_eval.jsonl, data/v2/validation, model_predictions.jsonl)
2. 自动兼容 full_dialog_history, messages, 对比评估等多种数据结构
3. 生成单文件自包含 (完全离线可用) 的现代化双栏交互 HTML 网页
4. 区分展示 system, user, assistant, tool_calls, tool 等多角色气泡与业务质检标签
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
    parser = argparse.ArgumentParser(description="多轮对话 JSONL 可视化生成工具")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="custom_eval/car_assistant_eval.jsonl",
        help="待可视化的 JSONL 文件路径 (默认: custom_eval/car_assistant_eval.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="生成的 HTML 文件保存路径 (留空时自动保存在同名 .html 或 outputs/visualize/ 下)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="生成完成后自动使用默认浏览器打开",
    )
    return parser.parse_args()


def load_jsonl_samples(file_path: str) -> List[Dict[str, Any]]:
    """加载并解析 JSONL 数据，自动适配多种结构"""
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
                # 统一识别对话历史列表
                history = data.get("full_dialog_history") or data.get("messages") or data.get("history") or []
                
                # 如果没有显式对话列表，但有 query 和模型预测/参考回答，则自动组装成虚拟多轮
                if not history and "query" in data:
                    mock_hist = []
                    if data.get("system_prompt") or data.get("system"):
                        mock_hist.append({"role": "system", "content": data.get("system_prompt") or data.get("system")})
                    mock_hist.append({"role": "user", "content": data.get("query")})
                    if data.get("model_b_response"):
                        mock_hist.append({"role": "assistant", "content": data.get("model_b_response")})
                    elif data.get("reference_response"):
                        mock_hist.append({"role": "assistant", "content": data.get("reference_response")})
                    history = mock_hist

                data["_normalized_dialog"] = history
                data["_index"] = idx
                samples.append(data)
            except Exception as e:
                print(f"[Warning] 第 {idx} 行解析 JSON 失败: {e}")

    return samples


def generate_html(samples: List[Dict[str, Any]], title: str, source_filename: str) -> str:
    """生成内嵌纯 CSS/JS 的单文件 HTML 网页"""
    json_data_str = json.dumps(samples, ensure_ascii=False)

    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)} - 对话过程可视化</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #334155;
            
            --user-bg: #2563eb;
            --user-text: #ffffff;
            --assistant-bg: #1e293b;
            --assistant-border: #475569;
            --system-bg: #1e2433;
            --system-border: #3b82f633;
            --tool-bg: #142e2b;
            --tool-border: #059669;
            --call-bg: #2d2417;
            --call-border: #d97706;

            --badge-cat: #0284c7;
            --badge-fact: #10b981;
            --badge-quest: #8b5cf6;
            --badge-act: #3b82f6;
            --badge-prohib: #ef4444;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        /* 顶部导航条 */
        header {{
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            z-index: 10;
        }}
        .header-title {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .header-title h1 {{
            font-size: 1.15rem;
            font-weight: 600;
            color: #38bdf8;
            letter-spacing: 0.5px;
        }}
        .file-badge {{
            font-size: 0.75rem;
            background: #0369a1;
            color: #e0f2fe;
            padding: 2px 8px;
            border-radius: 9999px;
        }}
        .stats-summary {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        /* 主工作区：双栏布局 */
        .main-container {{
            flex: 1;
            display: flex;
            overflow: hidden;
        }}

        /* 左侧样本导航面板 */
        .sidebar {{
            width: 380px;
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
        }}
        .search-box {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .search-input {{
            width: 100%;
            background-color: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            outline: none;
            transition: border-color 0.2s;
        }}
        .search-input:focus {{
            border-color: #38bdf8;
        }}
        .category-filter {{
            width: 100%;
            background-color: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            outline: none;
        }}

        .sample-list {{
            flex: 1;
            overflow-y: auto;
            padding: 8px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .sample-item {{
            padding: 12px;
            border-radius: 8px;
            background: var(--bg-primary);
            border: 1px solid transparent;
            cursor: pointer;
            transition: all 0.15s ease-in-out;
        }}
        .sample-item:hover {{
            background: #1e293b;
            border-color: #475569;
        }}
        .sample-item.active {{
            background: #0f2b48;
            border-color: #38bdf8;
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.15);
        }}
        .sample-item-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        .sample-id {{
            font-size: 0.78rem;
            font-family: monospace;
            color: #38bdf8;
            font-weight: 600;
        }}
        .sample-turns-badge {{
            font-size: 0.7rem;
            background: #334155;
            color: #94a3b8;
            padding: 1px 6px;
            border-radius: 4px;
        }}
        .sample-scenario {{
            font-size: 0.85rem;
            color: #f1f5f9;
            font-weight: 500;
            margin-bottom: 4px;
            line-height: 1.3;
        }}
        .sample-snippet {{
            font-size: 0.76rem;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        /* 右侧核心展示区域 */
        .content-area {{
            flex: 1;
            display: flex;
            flex-direction: column;
            background-color: var(--bg-primary);
            overflow-y: auto;
            position: relative;
        }}

        /* 顶部业务质检规则与元数据面板 */
        .meta-panel {{
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 28px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .meta-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }}
        .badge {{
            font-size: 0.75rem;
            padding: 3px 10px;
            border-radius: 6px;
            font-weight: 500;
        }}
        .badge-cat {{ background: rgba(2, 132, 199, 0.2); color: #38bdf8; border: 1px solid rgba(2, 132, 199, 0.4); }}
        .badge-role {{ background: rgba(139, 92, 246, 0.2); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.4); }}
        .badge-var {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 10px;
            font-size: 0.8rem;
        }}
        .meta-card {{
            background: var(--bg-primary);
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }}
        .meta-card-title {{
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
        .meta-card-content {{
            color: var(--text-secondary);
            line-height: 1.4;
        }}

        /* 对话河流容器 */
        .chat-stream {{
            flex: 1;
            padding: 24px 32px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            max-width: 1000px;
            margin: 0 auto;
            width: 100%;
        }}

        /* 气泡通用样式 */
        .msg-row {{
            display: flex;
            width: 100%;
            gap: 12px;
            align-items: flex-start;
        }}
        .msg-avatar {{
            width: 38px;
            height: 38px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            flex-shrink: 0;
            font-weight: 600;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}
        .msg-bubble-wrap {{
            max-width: 82%;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .msg-role-name {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 2px;
        }}
        .msg-bubble {{
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 0.92rem;
            line-height: 1.6;
            word-break: break-word;
            white-space: pre-wrap;
        }}

        /* 角色特定样式 */
        /* 1. System 系统提示 */
        .msg-row-system {{
            justify-content: center;
        }}
        .msg-bubble-system {{
            background: var(--system-bg);
            border: 1px dashed var(--system-border);
            color: #93c5fd;
            font-size: 0.82rem;
            border-radius: 8px;
            padding: 10px 16px;
            max-width: 90%;
            text-align: left;
        }}

        /* 2. User 车主提问 (右侧) */
        .msg-row-user {{
            justify-content: flex-end;
        }}
        .msg-row-user .msg-bubble-wrap {{
            align-items: flex-end;
        }}
        .msg-row-user .msg-bubble {{
            background-color: var(--user-bg);
            color: var(--user-text);
            border-bottom-right-radius: 2px;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        }}
        .avatar-user {{
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            order: 2;
        }}
        .msg-row-user .msg-bubble-wrap {{
            order: 1;
        }}

        /* 3. Assistant 客服回答 (左侧) */
        .msg-row-assistant {{
            justify-content: flex-start;
        }}
        .avatar-assistant {{
            background: linear-gradient(135deg, #0284c7, #0369a1);
            color: white;
        }}
        .msg-bubble-assistant {{
            background-color: var(--assistant-bg);
            border: 1px solid var(--assistant-border);
            color: #f1f5f9;
            border-bottom-left-radius: 2px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }}

        /* 4. Tool Calls 工具调用卡片 */
        .msg-bubble-call {{
            background-color: var(--call-bg);
            border: 1px solid var(--call-border);
            color: #fde68a;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.82rem;
            padding: 10px 14px;
        }}
        .call-header {{
            font-weight: 600;
            color: #f59e0b;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        /* 5. Tool 返回数据卡片 */
        .msg-bubble-tool {{
            background-color: var(--tool-bg);
            border: 1px solid var(--tool-border);
            color: #a7f3d0;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.82rem;
            padding: 10px 14px;
        }}
        .tool-header {{
            font-weight: 600;
            color: #10b981;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        /* 空状态提示 */
        .empty-state {{
            margin: auto;
            text-align: center;
            color: var(--text-muted);
            padding: 40px;
        }}

        /* 代码块与高亮样式 */
        code {{
            background: rgba(0,0,0,0.3);
            padding: 2px 4px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>

    <header>
        <div class="header-title">
            <h1>🚗 智能汽车多轮对话与质检可视化</h1>
            <span class="file-badge">{html.escape(source_filename)}</span>
        </div>
        <div class="stats-summary" id="stats-summary">
            加载中...
        </div>
    </header>

    <div class="main-container">
        <!-- 左侧样本导航 -->
        <div class="sidebar">
            <div class="search-box">
                <input type="text" id="search-input" class="search-input" placeholder="🔍 搜索场景 / ID / 用户提问...">
                <select id="category-filter" class="category-filter">
                    <option value="">全部场景类别 (All Categories)</option>
                </select>
            </div>
            <div class="sample-list" id="sample-list">
                <!-- 动态渲染样本列表 -->
            </div>
        </div>

        <!-- 右侧内容详情 -->
        <div class="content-area" id="content-area">
            <!-- 动态渲染选中样本 -->
        </div>
    </div>

    <script>
        const SAMPLES = {json_data_str};
        let currentIndex = 0;
        let filteredIndices = SAMPLES.map((_, i) => i);

        // 初始化
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

            // 支持键盘上下键切换
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
                    (s.scenario && s.scenario.toLowerCase().includes(searchTxt)) ||
                    (s.query && s.query.toLowerCase().includes(searchTxt)) ||
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
                `当前展示: ${{filteredIndices.length}} / ${{SAMPLES.length}} 组会话`;

            filteredIndices.forEach(idx => {{
                const s = SAMPLES[idx];
                const itemEl = document.createElement('div');
                itemEl.className = 'sample-item' + (idx === currentIndex ? ' active' : '');
                itemEl.id = 'sample-item-' + idx;

                const dialog = s._normalized_dialog || [];
                const turnCount = dialog.filter(m => m.role === 'assistant').length;

                itemEl.innerHTML = `
                    <div class="sample-item-header">
                        <span class="sample-id">#${{s._index}} ${{s.id || 'sample_' + idx}}</span>
                        <span class="sample-turns-badge">${{turnCount}} 轮问答</span>
                    </div>
                    <div class="sample-scenario">${{escapeHtml(s.scenario || s.subcategory || '车机对话样本')}}</div>
                    <div class="sample-snippet">${{escapeHtml(s.query || (dialog[1] && dialog[1].content) || '')}}</div>
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
            document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('active'));
            const activeEl = document.getElementById('sample-item-' + currentIndex);
            if (activeEl) {{
                activeEl.classList.add('active');
                activeEl.scrollIntoView({{ block: 'nearest' }});
            }}
        }}

        function renderContent(s) {{
            const contentArea = document.getElementById('content-area');
            const dialog = s._normalized_dialog || [];

            // 业务质检规则卡片 (若数据中存在)
            let metaHtml = '';
            const facts = s.required_facts || [];
            const questions = s.required_questions || [];
            const actions = s.required_actions || [];
            const prohibited = s.prohibited_actions || [];

            metaHtml = `
                <div class="meta-panel">
                    <div class="meta-tags">
                        ${{s.category ? `<span class="badge badge-cat">📁 ${{escapeHtml(s.category)}} / ${{escapeHtml(s.subcategory || '')}}</span>` : ''}}
                        ${{s.customer_role ? `<span class="badge badge-role">👤 ${{escapeHtml(s.customer_role)}}</span>` : ''}}
                        ${{s.variation_name ? `<span class="badge badge-var">⚡ ${{escapeHtml(s.variation_name)}}</span>` : ''}}
                        ${{s.tool_required ? `<span class="badge" style="background:rgba(16,185,129,0.2);color:#34d399;border:1px solid #10b981;">🛠️ 必须调用工具: ${{escapeHtml(s.tool_name || '内置工具')}}</span>` : ''}}
                    </div>
                    ${{s.user_goal ? `<div style="font-size:0.88rem;color:#e2e8f0;"><strong>🎯 车主诉求：</strong>${{escapeHtml(s.user_goal)}}</div>` : ''}}
                    
                    ${{(facts.length || questions.length || actions.length || prohibited.length) ? `
                    <div class="meta-grid">
                        ${{facts.length ? `<div class="meta-card"><div class="meta-card-title title-fact">📌 必须包含事实 (Facts)</div><div class="meta-card-content">${{escapeHtml(facts.join('； '))}}</div></div>` : ''}}
                        ${{questions.length ? `<div class="meta-card"><div class="meta-card-title title-quest">❓ 必须主动追问 (Questions)</div><div class="meta-card-content">${{escapeHtml(questions.join('； '))}}</div></div>` : ''}}
                        ${{actions.length ? `<div class="meta-card"><div class="meta-card-title title-act">✅ 必须指导动作 (Actions)</div><div class="meta-card-content">${{escapeHtml(actions.join('； '))}}</div></div>` : ''}}
                        ${{prohibited.length ? `<div class="meta-card"><div class="meta-card-title title-prohib">⛔ 绝对禁止行为 (Prohibited)</div><div class="meta-card-content" style="color:#fca5a5;">${{escapeHtml(prohibited.join('； '))}}</div></div>` : ''}}
                    </div>` : ''}}
                </div>
            `;

            // 对话河流
            let chatHtml = '<div class="chat-stream">';
            dialog.forEach((msg, mIdx) => {{
                const role = msg.role;
                const content = msg.content || '';
                const toolCalls = msg.tool_calls;

                if (role === 'system') {{
                    chatHtml += `
                        <div class="msg-row msg-row-system">
                            <div class="msg-bubble msg-bubble-system">
                                <strong>⚙️ System Prompt:</strong> ${{escapeHtml(content)}}
                            </div>
                        </div>
                    `;
                }} else if (role === 'user') {{
                    chatHtml += `
                        <div class="msg-row msg-row-user">
                            <div class="msg-avatar avatar-user">🧑</div>
                            <div class="msg-bubble-wrap">
                                <span class="msg-role-name">车主 (User)</span>
                                <div class="msg-bubble">${{escapeHtml(content)}}</div>
                            </div>
                        </div>
                    `;
                }} else if (role === 'assistant') {{
                    chatHtml += `
                        <div class="msg-row msg-row-assistant">
                            <div class="msg-avatar avatar-assistant">🤖</div>
                            <div class="msg-bubble-wrap">
                                <span class="msg-role-name">官方客服助手 (Assistant)</span>
                                ${{content ? `<div class="msg-bubble msg-bubble-assistant">${{formatMarkdownText(content)}}</div>` : ''}}
                                ${{toolCalls ? `
                                    <div class="msg-bubble msg-bubble-call">
                                        <div class="call-header">⚡ 发起系统工具调用 (Tool Call)</div>
                                        <pre style="margin:0;overflow-x:auto;">${{escapeHtml(JSON.stringify(toolCalls, null, 2))}}</pre>
                                    </div>
                                ` : ''}}
                            </div>
                        </div>
                    `;
                }} else if (role === 'tool') {{
                    chatHtml += `
                        <div class="msg-row msg-row-assistant" style="padding-left: 50px;">
                            <div class="msg-bubble-wrap" style="max-width: 90%;">
                                <div class="msg-bubble msg-bubble-tool">
                                    <div class="tool-header">📦 系统工具返回数据 [${{escapeHtml(msg.name || 'API Response')}}]</div>
                                    <pre style="margin:0;overflow-x:auto;">${{escapeHtml(tryFormatJson(content))}}</pre>
                                </div>
                            </div>
                        </div>
                    `;
                }}
            }});
            chatHtml += '</div>';

            contentArea.innerHTML = metaHtml + chatHtml;
        }}

        function renderEmpty() {{
            document.getElementById('content-area').innerHTML = `
                <div class="empty-state">
                    <h2>没有找到符合条件的对话样本</h2>
                    <p style="margin-top:8px;">请尝试更换搜索关键词或重置筛选条件</p>
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
            // 简单加粗转换
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
    print("智能汽车客服助手 - JSONL 对话过程可视化生成器")
    print(f"输入文件: {args.input}")
    print("=" * 70)

    samples = load_jsonl_samples(args.input)
    print(f"成功加载 {len(samples)} 条对话会话数据！")

    if not args.output:
        # 自动保存在 outputs/visualize/ 目录下
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
    print("💡 提示: 该 HTML 文件完全离线自包含，直接双击即可用浏览器打开查看。")

    if args.open:
        abs_path = os.path.abspath(args.output)
        print(f"🌐 正在自动打开浏览器: {abs_path} ...")
        webbrowser.open(f"file:///{abs_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
