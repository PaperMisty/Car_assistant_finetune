"""
utils/vllm_stress_test.py - vLLM SFT 汽车客服模型多梯度并发压力测试与性能基准评估

核心评估维度：
1. 【时延体验指标 (Latency)】:
   - P95 TTFT (Time To First Token) 首字生成时延 (ms)
   - P95 TPOT (Time Per Output Token) 每输出Token耗时 (ms/tok)
2. 【吞吐能力指标 (Capacity)】:
   - Generation Throughput (Tokens/s) 生成吞吐量 (硬件性能天花板)
   - RPS (Requests Per Second) 每秒成功处理请求数
3. 【显存与排队水位 (Memory)】:
   - GPU KV Cache 占用率峰值与均值 (gpu_cache_usage_factor)
   - KV Cache 前缀缓存命中率 (prefix_cache_hit_rate)
4. 【系统稳定性告警 (Stability)】:
   - Preemption Rate (抢占率 %): 显存耗尽时请求被换出 CPU 或重新计算的频率与次数

运行模式：
- 在线/生产模式: 异步流式请求 vLLM 服务端 (/v1/chat/completions) + Prometheus 接口 (/metrics)
- 离线 Mock 模式 (--mock): 本地全真仿真并发梯度测试，自动生成指标数据与高清可视化图表
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@dataclass
class RequestLatencyRecord:
    """单个请求的时延与采样指标"""
    request_id: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float                 # 首字生成延迟 (Time To First Token)
    total_latency_ms: float        # 端到端总时延
    tpot_ms: float                 # 平均每输出一个 Token 耗时 (Time Per Output Token)
    success: bool
    error_msg: Optional[str] = None


@dataclass
class LevelBenchmarkSummary:
    """单个并发梯度下的聚合评测指标"""
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    
    # 时延体验指标 (Latency)
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_mean_ms: float
    
    tpot_p50_ms: float
    tpot_p90_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    tpot_mean_ms: float
    
    # 吞吐与容量指标 (Capacity)
    generation_throughput_tps: float  # Output Tokens / s
    total_throughput_tps: float       # (Input + Output Tokens) / s
    rps: float                        # Requests / s
    
    # 显存与排队健康度 (vLLM Internal Metrics)
    gpu_cache_usage_peak_pct: float   # KV Cache 峰值占用率 %
    gpu_cache_usage_mean_pct: float   # KV Cache 平均占用率 %
    prefix_cache_hit_rate_pct: float  # 前缀缓存命中率 %
    
    # 稳定性生命线告警
    preemption_count: int             # 发生抢占的次数
    preemption_rate_pct: float        # 抢占率 % (Preemptions / Total Requests * 100)


def load_car_assistant_prompts(dataset_path: str = "custom_eval/car_assistant_eval.jsonl", max_samples: int = 128) -> List[List[Dict[str, str]]]:
    """从项目评测集加载真实车机客服对话 Prompt"""
    if os.path.exists(dataset_path):
        prompts = []
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    sys_prompt = item.get("system_prompt", "你是智能汽车官方客服助手。")
                    user_q = item.get("query", "你好，请帮我查询车辆配置。")
                    prompts.append([
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_q}
                    ])
                    if len(prompts) >= max_samples:
                        break
            if prompts:
                return prompts
        except Exception as e:
            print(f"⚠️ 读取评测集异常: {e}，将使用内置典型车机场景提示词。")

    # 内置回退提示词集 (真实多轮车主高频场景)
    default_cases = [
        "你好，车架号是 LSVAA123456789012，我刚才进隧道自动大灯没有自动点亮，请帮我核查该功能配置与设置路径。",
        "我的车显示胎压告警代码 TPMS-02，请问在高速行驶时需要立即靠边停车吗？",
        "我想预约明天下午 2 点的 4S 店常规 1 万公里保养，车架号 LSVBB987654321098。",
        "请帮我查询车辆当前是否存在未执行的安全召回活动或软件升级批次？",
        "行车过程中后备箱感应尾门突然打不开了，脚踢感应没有反应，怎么重置？",
    ]
    return [
        [
            {"role": "system", "content": "你是智能汽车官方客服助手。请严谨、专业、高效地解答车主问题。"},
            {"role": "user", "content": q}
        ]
        for q in default_cases
    ]


async def fetch_vllm_prometheus_metrics(metrics_url: str) -> Dict[str, float]:
    """从 vLLM 服务端 /metrics 接口实时抓取内核 Prometheus 指标"""
    metrics = {
        "gpu_cache_usage_factor": 0.0,
        "num_preemptions_total": 0.0,
        "prefix_cache_hit_rate": 0.0,
    }
    try:
        timeout = aiohttp.ClientTimeout(total=2.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(metrics_url) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    for line in text.splitlines():
                        if line.startswith("#"):
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            name, val = parts[0], parts[1]
                            try:
                                v_float = float(val)
                                # 兼容 vllm:xxx 与 vllm_xxx 及无前缀格式
                                if "gpu_cache_usage_factor" in name or "gpu_cache_usage_pct" in name:
                                    metrics["gpu_cache_usage_factor"] = v_float
                                elif "num_preemptions" in name or "preemption_total" in name:
                                    metrics["num_preemptions_total"] = v_float
                                elif "prefix_cache_hit_rate" in name or "cache_hit_rate" in name:
                                    metrics["prefix_cache_hit_rate"] = v_float
                            except ValueError:
                                pass
    except Exception:
        pass
    return metrics


async def send_single_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    req_id: str
) -> RequestLatencyRecord:
    """发送单次流式请求并高精度记录首字时延 TTFT 与逐词耗时 TPOT"""
    t_start = time.perf_counter()
    t_first_token: Optional[float] = None
    output_tokens_count = 0
    prompt_tokens_est = sum(len(m.get("content", "")) for m in payload.get("messages", [])) // 2

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                return RequestLatencyRecord(
                    request_id=req_id,
                    prompt_tokens=prompt_tokens_est,
                    output_tokens=0,
                    ttft_ms=0.0,
                    total_latency_ms=(time.perf_counter() - t_start) * 1000,
                    tpot_ms=0.0,
                    success=False,
                    error_msg=f"HTTP {resp.status}: {err_text[:100]}"
                )

            async for chunk in resp.content:
                if not chunk:
                    continue
                # 收到首个有效数据包时刻
                if t_first_token is None:
                    t_first_token = time.perf_counter()

                # 解析 SSE 流数据行: data: {...}
                lines = chunk.decode("utf-8", errors="ignore").splitlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            delta_json = json.loads(line[6:])
                            choices = delta_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if delta.get("content"):
                                    output_tokens_count += 1
                        except Exception:
                            pass

        t_end = time.perf_counter()
        if t_first_token is None:
            t_first_token = t_end

        ttft_ms = (t_first_token - t_start) * 1000.0
        total_latency_ms = (t_end - t_start) * 1000.0
        
        # 计算 TPOT = (总时延 - 首字时延) / (输出Token数 - 1)
        if output_tokens_count > 1:
            tpot_ms = ((t_end - t_first_token) / (output_tokens_count - 1)) * 1000.0
        else:
            tpot_ms = (t_end - t_first_token) * 1000.0

        return RequestLatencyRecord(
            request_id=req_id,
            prompt_tokens=prompt_tokens_est,
            output_tokens=max(1, output_tokens_count),
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            tpot_ms=tpot_ms,
            success=True
        )

    except Exception as e:
        return RequestLatencyRecord(
            request_id=req_id,
            prompt_tokens=prompt_tokens_est,
            output_tokens=0,
            ttft_ms=0.0,
            total_latency_ms=(time.perf_counter() - t_start) * 1000,
            tpot_ms=0.0,
            success=False,
            error_msg=str(e)
        )


async def run_concurrency_level_benchmark(
    base_url: str,
    model_name: str,
    concurrency: int,
    total_requests: int,
    prompt_pool: List[List[Dict[str, str]]],
    max_tokens: int = 256,
    mock: bool = False
) -> LevelBenchmarkSummary:
    """在指定并发梯度下执行完整流式压测"""
    chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    metrics_url = f"{base_url.rstrip('/')}/metrics"

    print(f"\n🚀 [并发级别: {concurrency:2d}] 开始压测: 总计 {total_requests} 个流式请求...")

    if mock:
        # Mock 全真仿真生成器 (基于排队论 M/M/m 模型与 GPU 显存饱和曲线)
        return simulate_mock_concurrency_level(concurrency, total_requests)

    # 1. 记录测试前 Prometheus 基线指标
    initial_metrics = await fetch_vllm_prometheus_metrics(metrics_url)
    cache_usage_samples = []

    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=180.0)
    records: List[RequestLatencyRecord] = []

    t_bench_start = time.perf_counter()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def worker(idx: int):
            async with semaphore:
                messages = prompt_pool[idx % len(prompt_pool)]
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "stream": True,
                }
                res = await send_single_streaming_request(
                    session=session,
                    url=chat_url,
                    payload=payload,
                    req_id=f"c{concurrency}_r{idx}"
                )
                records.append(res)

        # 启动后台显存采样任务
        stop_sampling = False
        async def cache_monitor():
            while not stop_sampling:
                m = await fetch_vllm_prometheus_metrics(metrics_url)
                cache_usage_samples.append(m.get("gpu_cache_usage_factor", 0.0) * 100)
                await asyncio.sleep(0.1)

        sampler_task = asyncio.create_task(cache_monitor())
        tasks = [asyncio.create_task(worker(i)) for i in range(total_requests)]
        await asyncio.gather(*tasks)
        stop_sampling = True
        sampler_task.cancel()

    t_bench_end = time.perf_counter()
    duration = max(0.001, t_bench_end - t_bench_start)

    # 2. 抓取测试后指标
    final_metrics = await fetch_vllm_prometheus_metrics(metrics_url)
    preemptions = max(0, int(final_metrics.get("num_preemptions_total", 0) - initial_metrics.get("num_preemptions_total", 0)))
    prefix_hit_rate = final_metrics.get("prefix_cache_hit_rate", 0.0) * 100

    return compute_summary_from_records(
        concurrency=concurrency,
        records=records,
        duration=duration,
        cache_samples=cache_usage_samples,
        preemptions=preemptions,
        prefix_hit_rate=prefix_hit_rate
    )


def simulate_mock_concurrency_level(concurrency: int, total_requests: int) -> LevelBenchmarkSummary:
    """
    全真排队论 Mock 仿真模型：
    - 当并发度较小 (如 1~8) 时，GPU 算力充足，TTFT 极低 (~40ms)，Throughput 随并发线性攀升；
    - 当并发度达到饱和点 (如 16~32) 时，显存带宽与 KV Cache 逼近 85%~95%，TTFT 爬升；
    - 当并发度过载 (>32) 时，KV Cache 爆满，触发 Preemption (抢占重算)，TTFT 呈指数级恶化！
    """
    # 模拟首字延迟 (随着并发增大，排队时间增加)
    base_ttft = 35.0 + 8.5 * concurrency + (math.exp(concurrency / 12.0) if concurrency > 20 else 0)
    ttft_samples = np.random.normal(loc=base_ttft, scale=base_ttft * 0.15, size=total_requests)
    ttft_samples = np.clip(ttft_samples, 20.0, 5000.0)

    # 模拟 TPOT (单 token 解码耗时，批处理增大导致显存搬运延迟微增)
    base_tpot = 12.0 + 0.45 * concurrency + (concurrency * 0.2 if concurrency > 30 else 0)
    tpot_samples = np.random.normal(loc=base_tpot, scale=base_tpot * 0.1, size=total_requests)
    tpot_samples = np.clip(tpot_samples, 8.0, 200.0)

    # 模拟输出 token 数 (汽车客服通常 45 ~ 95 tokens)
    avg_out_tokens = 68.0
    total_output_tokens = int(avg_out_tokens * total_requests)

    # 模拟总吞吐与总测试耗时
    # 吞吐上限随并发饱和 (符合饱和 S 曲线)
    max_hardware_tps = 3200.0
    gen_tps = max_hardware_tps * (1.0 - math.exp(-concurrency / 8.5))
    duration = max(0.5, total_output_tokens / gen_tps)
    rps = total_requests / duration

    # 模拟 KV Cache 占用率 (随并发升高从 15% 上升到 98%)
    cache_peak = min(99.5, 12.0 + concurrency * 2.2 + (math.log(concurrency + 1) * 6.0))
    cache_mean = cache_peak * 0.88
    prefix_hit_rate = 65.0 + min(25.0, concurrency * 0.5)

    # 模拟抢占率：并发超过 32 显存耗尽时产生 Preemption
    preemption_count = 0
    if concurrency >= 32:
        preemption_count = int((concurrency - 28) * (total_requests / 25.0))
    preempt_rate = (preemption_count / total_requests) * 100.0

    return LevelBenchmarkSummary(
        concurrency=concurrency,
        total_requests=total_requests,
        successful_requests=total_requests,
        failed_requests=0,
        duration_seconds=round(duration, 2),
        ttft_p50_ms=round(float(np.percentile(ttft_samples, 50)), 1),
        ttft_p90_ms=round(float(np.percentile(ttft_samples, 90)), 1),
        ttft_p95_ms=round(float(np.percentile(ttft_samples, 95)), 1),
        ttft_p99_ms=round(float(np.percentile(ttft_samples, 99)), 1),
        ttft_mean_ms=round(float(np.mean(ttft_samples)), 1),
        tpot_p50_ms=round(float(np.percentile(tpot_samples, 50)), 2),
        tpot_p90_ms=round(float(np.percentile(tpot_samples, 90)), 2),
        tpot_p95_ms=round(float(np.percentile(tpot_samples, 95)), 2),
        tpot_p99_ms=round(float(np.percentile(tpot_samples, 99)), 2),
        tpot_mean_ms=round(float(np.mean(tpot_samples)), 2),
        generation_throughput_tps=round(gen_tps, 1),
        total_throughput_tps=round(gen_tps * 1.6, 1),
        rps=round(rps, 2),
        gpu_cache_usage_peak_pct=round(cache_peak, 1),
        gpu_cache_usage_mean_pct=round(cache_mean, 1),
        prefix_cache_hit_rate_pct=round(prefix_hit_rate, 1),
        preemption_count=preemption_count,
        preemption_rate_pct=round(preempt_rate, 2),
    )


def compute_summary_from_records(
    concurrency: int,
    records: List[RequestLatencyRecord],
    duration: float,
    cache_samples: List[float],
    preemptions: int,
    prefix_hit_rate: float
) -> LevelBenchmarkSummary:
    """汇总单并发级别下的所有时延分位数与吞吐指标"""
    success_records = [r for r in records if r.success]
    total_reqs = len(records)
    succ_reqs = len(success_records)
    fail_reqs = total_reqs - succ_reqs

    ttfts = [r.ttft_ms for r in success_records] or [0.0]
    tpots = [r.tpot_ms for r in success_records] or [0.0]
    total_out_toks = sum(r.output_tokens for r in success_records)
    total_in_toks = sum(r.prompt_tokens for r in success_records)

    cache_peak = max(cache_samples) if cache_samples else 0.0
    cache_mean = float(np.mean(cache_samples)) if cache_samples else 0.0

    gen_tps = total_out_toks / duration
    tot_tps = (total_in_toks + total_out_toks) / duration
    rps = succ_reqs / duration
    preempt_rate = (preemptions / max(1, total_reqs)) * 100.0

    return LevelBenchmarkSummary(
        concurrency=concurrency,
        total_requests=total_reqs,
        successful_requests=succ_reqs,
        failed_requests=fail_reqs,
        duration_seconds=round(duration, 2),
        ttft_p50_ms=round(float(np.percentile(ttfts, 50)), 1),
        ttft_p90_ms=round(float(np.percentile(ttfts, 90)), 1),
        ttft_p95_ms=round(float(np.percentile(ttfts, 95)), 1),
        ttft_p99_ms=round(float(np.percentile(ttfts, 99)), 1),
        ttft_mean_ms=round(float(np.mean(ttfts)), 1),
        tpot_p50_ms=round(float(np.percentile(tpots, 50)), 2),
        tpot_p90_ms=round(float(np.percentile(tpots, 90)), 2),
        tpot_p95_ms=round(float(np.percentile(tpots, 95)), 2),
        tpot_p99_ms=round(float(np.percentile(tpots, 99)), 2),
        tpot_mean_ms=round(float(np.mean(tpots)), 2),
        generation_throughput_tps=round(gen_tps, 1),
        total_throughput_tps=round(tot_tps, 1),
        rps=round(rps, 2),
        gpu_cache_usage_peak_pct=round(cache_peak, 1),
        gpu_cache_usage_mean_pct=round(cache_mean, 1),
        prefix_cache_hit_rate_pct=round(prefix_hit_rate, 1),
        preemption_count=preemptions,
        preemption_rate_pct=round(preempt_rate, 2),
    )


def print_ascii_summary_table(results: List[LevelBenchmarkSummary]):
    """以清晰结构化的表格输出各梯度核心指标"""
    print("\n" + "=" * 115)
    print("📊【vLLM SFT 汽车客服模型多梯度并发压力测试报告汇总】")
    print("=" * 115)
    header = (
        f"{'并发':>4} | {'RPS':>6} | {'P95 TTFT(ms)':>13} | {'P95 TPOT(ms)':>13} | "
        f"{'生成吞吐(tok/s)':>15} | {'KV Cache峰值':>12} | {'抢占次数':>8} | {'抢占率(%)':>9}"
    )
    print(header)
    print("-" * 115)
    for r in results:
        line = (
            f"{r.concurrency:4d} | {r.rps:6.2f} | {r.ttft_p95_ms:13.1f} | {r.tpot_p95_ms:13.2f} | "
            f"{r.generation_throughput_tps:15.1f} | {r.gpu_cache_usage_peak_pct:11.1f}% | "
            f"{r.preemption_count:8d} | {r.preemption_rate_pct:8.2f}%"
        )
        print(line)
    print("=" * 115)


def generate_svg_dashboard(results: List[LevelBenchmarkSummary], save_path: str):
    """
    零依赖生成极高质量、矢量无损的 6 子图全景看板 (SVG 格式)，
    可在浏览器与 VS Code 中完美展示，完全不受本地第三方图形库限制。
    """
    concs = [r.concurrency for r in results]
    p95_ttfts = [r.ttft_p95_ms for r in results]
    p95_tpots = [r.tpot_p95_ms for r in results]
    tpss = [r.generation_throughput_tps for r in results]
    rpss = [r.rps for r in results]
    caches = [r.gpu_cache_usage_peak_pct for r in results]
    preempts = [r.preemption_rate_pct for r in results]

    width, height = 1200, 800
    card_w, card_h = 360, 220
    margin_x, margin_y = 40, 90
    gap_x, gap_y = 35, 45

    def build_sparkline(data: List[float], w: float, h: float, color: str, y_label: str) -> str:
        min_v = 0.0
        max_v = max(data) * 1.15 if max(data) > 0 else 1.0
        n = len(data)
        dx = (w - 60) / max(1, n - 1)
        points = []
        circles = []
        labels = []
        for i, v in enumerate(data):
            x = 45 + i * dx
            y = h - 35 - ((v - min_v) / (max_v - min_v)) * (h - 60)
            points.append(f"{x:.1f},{y:.1f}")
            circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />')
            labels.append(f'<text x="{x:.1f}" y="{y-8:.1f}" font-size="10" fill="#333" text-anchor="middle">{v:.1f}</text>')

        poly = " ".join(points)
        # 底部 X 轴刻度
        x_ticks = [f'<text x="{45 + i * dx:.1f}" y="{h-15}" font-size="10" fill="#666" text-anchor="middle">c={concs[i]}</text>' for i in range(n)]
        
        return f'''
        <line x1="45" y1="{h-35}" x2="{w-15}" y2="{h-35}" stroke="#ddd" stroke-width="1"/>
        <polyline fill="none" stroke="{color}" stroke-width="3" points="{poly}" />
        {"".join(circles)}
        {"".join(labels)}
        {"".join(x_ticks)}
        <text x="15" y="20" font-size="11" fill="#888">{y_label}</text>
        '''

    cards_meta = [
        ("① P95 首字时延 (TTFT 体验指标)", p95_ttfts, "#2563eb", "ms (越低越好)"),
        ("② P95 单Token耗时 (TPOT 流畅度)", p95_tpots, "#059669", "ms/tok (越低越好)"),
        ("③ 生成吞吐量 (Tokens/s 天花板)", tpss, "#7c3aed", "tok/s (越高越好)"),
        ("④ 业务并发容量 (RPS 请求率)", rpss, "#d97706", "req/s (越高越好)"),
        ("⑤ GPU KV Cache 峰值占用水位", caches, "#dc2626", "% 水位 (超90%预警)"),
        ("⑥ 抢占率 Preemption Rate (失稳告警)", preempts, "#b91c1c", "% 抢占率 (大于0即失稳)"),
    ]

    cards_svg = []
    for idx, (title, data, color, unit) in enumerate(cards_meta):
        row = idx // 3
        col = idx % 3
        x = margin_x + col * (card_w + gap_x)
        y = margin_y + row * (card_h + gap_y)
        content = build_sparkline(data, card_w, card_h, color, unit)
        card = f'''
        <g transform="translate({x}, {y})">
            <rect width="{card_w}" height="{card_h}" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="drop-shadow(0 2px 4px rgba(0,0,0,0.04))"/>
            <text x="16" y="28" font-size="13" font-weight="bold" fill="#1e293b">{title}</text>
            {content}
        </g>
        '''
        cards_svg.append(card)

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
    <rect width="{width}" height="{height}" fill="#f8fafc"/>
    <text x="{width/2}" y="42" font-size="22" font-weight="bold" fill="#0f172a" text-anchor="middle">🚗 vLLM SFT 汽车客服模型多梯度并发压力测试全景看板</text>
    <text x="{width/2}" y="65" font-size="12" fill="#64748b" text-anchor="middle">指标维度：P95 TTFT | P95 TPOT | 生成吞吐 (Tokens/s) | RPS | KV Cache 水位 | 抢占率 (Preemption)</text>
    {"".join(cards_svg)}
</svg>'''

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"📊 [矢量图表] 已成功保存高清全景看板至: {save_path}")


def try_plot_matplotlib_dashboard(results: List[LevelBenchmarkSummary], save_path: str):
    """如果环境中安装了 matplotlib，额外生成高 DPI 的 PNG 位图"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        # 设置支持中文的无衬线字体
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans", "Arial"]
        matplotlib.rcParams["axes.unicode_minus"] = False

        concs = [r.concurrency for r in results]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=200)
        fig.suptitle("vLLM SFT 汽车客服模型多梯度并发压力测试性能基准 (Qwen3-8B-SFT)", fontsize=16, fontweight="bold")

        # 1. P95 TTFT
        ax = axes[0, 0]
        ax.plot(concs, [r.ttft_p95_ms for r in results], marker="o", color="#2563eb", linewidth=2.2, label="P95 TTFT")
        ax.plot(concs, [r.ttft_p50_ms for r in results], marker="s", color="#93c5fd", linestyle="--", label="P50 TTFT")
        ax.set_title("P95/P50 首字延迟 TTFT (ms)")
        ax.set_xlabel("并发数 (Concurrency)")
        ax.set_ylabel("毫秒 (ms)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        # 2. P95 TPOT
        ax = axes[0, 1]
        ax.plot(concs, [r.tpot_p95_ms for r in results], marker="o", color="#059669", linewidth=2.2)
        ax.set_title("P95 单Token生成耗时 TPOT (ms/tok)")
        ax.set_xlabel("并发数 (Concurrency)")
        ax.set_ylabel("毫秒/Token")
        ax.grid(True, linestyle="--", alpha=0.5)

        # 3. Generation Throughput
        ax = axes[0, 2]
        ax.plot(concs, [r.generation_throughput_tps for r in results], marker="^", color="#7c3aed", linewidth=2.2)
        ax.set_title("生成吞吐量 Output TPS (Tokens/s)")
        ax.set_xlabel("并发数 (Concurrency)")
        ax.set_ylabel("Tokens/s")
        ax.grid(True, linestyle="--", alpha=0.5)

        # 4. RPS
        ax = axes[1, 0]
        ax.plot(concs, [r.rps for r in results], marker="D", color="#d97706", linewidth=2.2)
        ax.set_title("业务请求吞吐率 RPS (Requests/s)")
        ax.set_xlabel("并发数 (Concurrency)")
        ax.set_ylabel("Requests/s")
        ax.grid(True, linestyle="--", alpha=0.5)

        # 5. KV Cache Peak Usage
        ax = axes[1, 1]
        ax.plot(concs, [r.gpu_cache_usage_peak_pct for r in results], marker="v", color="#dc2626", linewidth=2.2)
        ax.axhline(y=90.0, color="#f87171", linestyle=":", label="90% 显存警戒线")
        ax.set_title("GPU KV Cache 峰值占用率 (%)")
        ax.set_xlabel("并发数 (Concurrency)")
        ax.set_ylabel("Cache 占用率 (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        # 6. Preemption Rate
        ax = axes[1, 2]
        ax.plot(concs, [r.preemption_rate_pct for r in results], marker="x", color="#b91c1c", linewidth=2.2)
        ax.set_title("抢占率 Preemption Rate (%) - 失稳指标")
        ax.set_xlabel("并发数 (Concurrency)")
        ax.set_ylabel("抢占率 (%)")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"📈 [PNG位图] 已成功保存高清图片至: {save_path}")
    except Exception:
        # 环境暂无 matplotlib 或绘图异常时不阻断流程
        pass


def main():
    parser = argparse.ArgumentParser(description="vLLM SFT 汽车客服模型多梯度并发压力测试")
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000").strip("\"'"),
        help="vLLM OpenAI 兼容服务端根地址 (默认 http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "Qwen3-8B-sft-lora").strip("\"'"),
        help="待压测模型服务名称 (如 Qwen3-8B-sft-lora 或 Qwen3-8B-sft-merged)",
    )
    parser.add_argument(
        "--concurrency_levels",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="并发梯度列表 (例如 1 2 4 8 16 32 64 或 5 10 20 40 60)",
    )
    parser.add_argument(
        "--num_requests_per_level",
        type=int,
        default=30,
        help="每个并发梯度发送的测试请求总数 (默认 30)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="custom_eval/car_assistant_eval.jsonl",
        help="压测输入数据集文件路径",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="单次生成最大 Token 限制",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/benchmark",
        help="评测结果与图表输出目录",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="启用全真排队论 Mock 仿真模式 (无 GPU 或本地开发验证使用)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_pool = load_car_assistant_prompts(args.dataset)

    print("=" * 85)
    print("🚗 vLLM SFT 汽车客服模型多梯度并发压力测试启动")
    print(f"测试模型: {args.model} | 服务地址: {args.base_url}")
    print(f"并发梯度: {args.concurrency_levels} | 每级请求数: {args.num_requests_per_level}")
    print(f"运行模式: {'全真 Mock 仿真压测' if args.mock else '真实 vLLM 线上服务压测'}")
    print(f"结果目录: {args.output_dir}")
    print("=" * 85)

    all_summaries: List[LevelBenchmarkSummary] = []

    for c in args.concurrency_levels:
        summary = asyncio.run(
            run_concurrency_level_benchmark(
                base_url=args.base_url,
                model_name=args.model,
                concurrency=c,
                total_requests=args.num_requests_per_level,
                prompt_pool=prompt_pool,
                max_tokens=args.max_tokens,
                mock=args.mock
            )
        )
        all_summaries.append(summary)

    # 1. 打印控制台表格
    print_ascii_summary_table(all_summaries)

    # 2. 保存结构化 JSON
    json_path = os.path.join(args.output_dir, "stress_test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in all_summaries], f, ensure_ascii=False, indent=2)
    print(f"\n💾 [数据落盘] 完整压测指标已保存至: {json_path}")

    # 3. 绘制并保存可视化图表 (SVG 矢量无损看板 + PNG 位图双重输出)
    svg_path = os.path.join(args.output_dir, "vllm_stress_benchmark.svg")
    generate_svg_dashboard(all_summaries, svg_path)

    png_path = os.path.join(args.output_dir, "vllm_stress_benchmark.png")
    try_plot_matplotlib_dashboard(all_summaries, png_path)

    print(f"✨ 压力测试与指标渲染全部完成！\n")


if __name__ == "__main__":
    main()
