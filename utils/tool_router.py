# -*- coding: utf-8 -*-
"""
工具路由与分发模块 (Tool Router Architecture)
================================================================================
本模块将汽车客服大模型的工具路由机制从训练/评测业务中完全解耦，支持以下工作模式：
1. [router]: 两阶段向量化智能路由 (BGE-M3 粗排 + BGE-Reranker 精排 -> 供给 Top-K 紧凑工具)
2. [full]:   全量工具直接注入 (零神经网络依赖，55+ 原子工具直接紧凑注入 Prompt，100% 稳健可用)
3. [none]:   纯对话模式 (不注入任何工具定义)

具备极致的容错降级能力：当路由权重缺失或加载失败时，自动平滑回退至 [full] 全量模式，保证流程绝对不中断。
================================================================================
"""

import os
import sys
import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.tool import export_openai_schemas, export_compact_schemas, get_tools_by_names, TOOL_REGISTRY

# 默认历史硬编码路径 (仅作兜底回退)
DEFAULT_BGE_M3_PATH = r"D:\ai_models\modelscope_cache\models\models\BAAI--bge-m3\snapshots\master"
DEFAULT_RERANKER_PATH = r"D:\ai_models\modelscope_cache\models\BAAI--bge-reranker-base"


def format_tool_signature_text(tool_schema: Dict[str, Any]) -> str:
    """将工具 schema 转换为富语义紧凑描述文本，专用于向量化检索与精排重打分"""
    func = tool_schema.get("function", tool_schema)
    name = func.get("name", "")
    desc = func.get("description", "")
    params = func.get("parameters", {})
    props = params.get("properties", {})
    required = set(params.get("required", []))

    param_lines = []
    for p_name, p_meta in props.items():
        p_desc = p_meta.get("description", "").replace("[必需] ", "").replace("[可选] ", "").strip()
        req_flag = "必需" if p_name in required else "可选"
        p_type = p_meta.get("type", "string")
        param_str = f"{p_name}({req_flag}, {p_type})"
        enum_vals = p_meta.get("enum")
        if enum_vals:
            param_str += f" 枚举: [{', '.join(str(e) for e in enum_vals)}]"
        if p_desc:
            param_str += f" - {p_desc}"
        param_lines.append(param_str)

    params_text = "; ".join(param_lines) if param_lines else "无参数"
    return f"工具名称: {name} | 功能: {desc} | 入参规范: {params_text}"


def resolve_model_path(target_path: Optional[str], env_key: str, default_rel: str, fallback_hardcoded: str) -> str:
    """智能解析跨平台模型路径：优先使用命令行参数/环境变量/相对路径，回退到历史硬编码"""
    candidates = [
        target_path,
        os.getenv(env_key),
        default_rel,
        fallback_hardcoded,
    ]
    for c in candidates:
        if c:
            cleaned = c.strip("\"'")
            if os.path.exists(cleaned):
                return cleaned
    return (target_path or os.getenv(env_key) or default_rel).strip("\"'")


def get_optimal_device(device_arg: str = "auto") -> str:
    """根据硬件可用性自动判定设备：有 GPU 优先使用 cuda，否则回退使用 cpu"""
    if device_arg and device_arg.lower() in ["cuda", "cpu"]:
        if device_arg.lower() == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
            return "cpu"
        return device_arg.lower()
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class BaseToolRouter(ABC):
    """工具路由器统一抽象基类"""

    @abstractmethod
    def route_tools(self, query: str, history_messages: Optional[List[Dict[str, Any]]] = None, top_k: int = 3) -> List[str]:
        """返回当前轮次推荐的工具名称列表"""
        pass

    @abstractmethod
    def get_compact_tools(self, query: str, history_messages: Optional[List[Dict[str, Any]]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """返回当前轮次供直接注入 Prompt 的紧凑工具 Schema 列表"""
        pass

    def release_memory(self):
        """显式释放模型占用的显存与系统内存 (若有)"""
        pass


class FullToolRouter(BaseToolRouter):
    """
    全量工具直接注入路由器 (零神经网络依赖，绝对稳健模式)
    特点：
    1. 无需下载或加载任何 BGE / Reranker 权重；
    2. 0ms 纯内存操作，直接把全部已注册的原子工具注入给模型；
    3. 适合在工具路由排查、模型权重损坏或极速测试阶段使用。
    """

    def __init__(self):
        self.all_tool_names = list(TOOL_REGISTRY.keys())
        self.cached_compact_schemas = export_compact_schemas(list(TOOL_REGISTRY.values()))
        print(f"📦 [FullToolRouter] 初始化完成：已启用全量工具直接注入模式 (共 {len(self.all_tool_names)} 个原子工具，零模型依赖)")

    def route_tools(self, query: str, history_messages: Optional[List[Dict[str, Any]]] = None, top_k: int = 3) -> List[str]:
        return self.all_tool_names

    def get_compact_tools(self, query: str, history_messages: Optional[List[Dict[str, Any]]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.cached_compact_schemas


class NoToolRouter(BaseToolRouter):
    """空工具路由器 (纯对话模式，不注入任何工具)"""

    def route_tools(self, query: str, history_messages: Optional[List[Dict[str, Any]]] = None, top_k: int = 3) -> List[str]:
        return []

    def get_compact_tools(self, query: str, history_messages: Optional[List[Dict[str, Any]]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        return []


class BGEM3ToolRetriever:
    """BGE-M3 语义向量检索器封装"""

    def __init__(self, model_path: str, device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.device = device
        print(f"📦 正在加载本地 BGE-M3 模型 (设备: {device}): {model_path} ...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        print(f"✅ BGE-M3 模型加载完毕，耗时: {time.time() - t0:.2f} 秒")

    def encode_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        import torch

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                cls_rep = outputs.last_hidden_state[:, 0]
                norm_rep = torch.nn.functional.normalize(cls_rep, p=2, dim=1)
                all_embeddings.append(norm_rep.cpu().numpy())
        return np.vstack(all_embeddings).astype(np.float32)


class BGEReranker:
    """BGE-Reranker 跨注意力重排器封装"""

    def __init__(self, model_path: str, device: str = "cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        print(f"📦 正在加载本地 BGE-Reranker 模型 (设备: {device}): {model_path} ...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()
        print(f"✅ BGE-Reranker 模型加载完毕，耗时: {time.time() - t0:.2f} 秒")

    def rerank(self, query: str, candidate_texts: List[str]) -> np.ndarray:
        import torch

        if not candidate_texts:
            return np.array([], dtype=np.float32)
        pairs = [[query, text] for text in candidate_texts]
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            return scores.cpu().numpy()


def get_or_build_tool_embeddings(
    retriever: BGEM3ToolRetriever,
    cache_dir: str = "experiments/cache",
    force_rebuild: bool = False
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """获取或离线预构建工具向量库"""
    os.makedirs(cache_dir, exist_ok=True)
    npy_path = os.path.join(cache_dir, "tool_bge_m3_embeddings.npy")
    meta_path = os.path.join(cache_dir, "tool_metadata.json")

    raw_schemas = export_openai_schemas()
    tools = [s["function"] for s in raw_schemas]

    if not force_rebuild and os.path.exists(npy_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if len(cached_meta) == len(tools):
                embeddings = np.load(npy_path)
                return cached_meta, embeddings
        except Exception:
            pass

    print(f"🔨 正在为 {len(tools)} 个工具生成 BGE-M3 语义向量...")
    tool_texts = [format_tool_signature_text(t) for t in tools]
    embeddings = retriever.encode_texts(tool_texts)
    np.save(npy_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(tools, f, ensure_ascii=False, indent=2)
    return tools, embeddings


class TwoStageToolRouter(BaseToolRouter):
    """两阶段工具路由器：BGE-M3 向量初筛 (Top-10) + BGE-Reranker 跨注意力精排 (Top-3)"""

    def __init__(
        self,
        m3_path: str = DEFAULT_BGE_M3_PATH,
        reranker_path: str = DEFAULT_RERANKER_PATH,
        device: str = "auto",
        cache_dir: str = "experiments/cache"
    ):
        self.device = get_optimal_device(device)
        self.available = False
        self.m3: Optional[BGEM3ToolRetriever] = None
        self.reranker: Optional[BGEReranker] = None

        try:
            print(f"📦 正在初始化两阶段工具路由器 (运行设备: {self.device})...")
            self.m3 = BGEM3ToolRetriever(model_path=m3_path, device=self.device)
            self.tools, self.embeddings = get_or_build_tool_embeddings(self.m3, cache_dir=cache_dir)
            self.reranker = BGEReranker(model_path=reranker_path, device=self.device)
            self.tool_names = [t.get("function", t).get("name") for t in self.tools]
            self.tool_texts = [format_tool_signature_text(t) for t in self.tools]
            self.tool_name_to_text = {n: txt for n, txt in zip(self.tool_names, self.tool_texts)}
            self.available = True
            print(f"✅ 两阶段工具路由器初始化成功！全量原子工具池: {len(self.tool_names)} 个\n")
        except Exception as e:
            print(f"⚠️ 工具路由器加载失败或权重缺失 ({e})")
            print("🔄 自动触发安全降级：平滑切换为 [FullToolRouter] 全量工具直接注入模式！\n")
            self._fallback_router = FullToolRouter()

    def route_tools(self, query: str, history_messages: Optional[List[Dict[str, Any]]] = None, top_k: int = 3) -> List[str]:
        if not self.available:
            return self._fallback_router.route_tools(query, history_messages, top_k)

        query_to_embed = query
        if history_messages and len(history_messages) > 1:
            first_user_msg = ""
            for m in history_messages:
                if m.get("role") == "user" and m.get("content"):
                    first_user_msg = m.get("content")
                    break
            prev_content = history_messages[-1].get("content") or "" if history_messages else ""
            query_to_embed = f"核心诉求: {first_user_msg[:60]} | 上文: {prev_content[:40]} | 当前提问: {query}"

        # 1. 粗排：BGE-M3 向量相似度初筛 Top-10
        q_vec = self.m3.encode_texts([query_to_embed])[0]
        scores = np.dot(self.embeddings, q_vec)
        top10_idx = np.argsort(scores)[::-1][:10]
        stage1_names = [self.tool_names[i] for i in top10_idx]

        # 2. 精排：BGE-Reranker 跨注意力深度重排
        cand_texts = [self.tool_name_to_text[name] for name in stage1_names]
        rerank_scores = self.reranker.rerank(query=query_to_embed, candidate_texts=cand_texts)
        rerank_order = np.argsort(rerank_scores)[::-1]
        final_names = [stage1_names[i] for i in rerank_order][:top_k]
        return final_names

    def get_compact_tools(self, query: str, history_messages: Optional[List[Dict[str, Any]]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.available:
            return self._fallback_router.get_compact_tools(query, history_messages, top_k)
        names = self.route_tools(query, history_messages, top_k=top_k)
        return export_compact_schemas(get_tools_by_names(names))

    def release_memory(self):
        """释放路由模型占用的 GPU 显存"""
        if self.m3 is not None:
            del self.m3
            self.m3 = None
        if self.reranker is not None:
            del self.reranker
            self.reranker = None
        try:
            import gc
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            print("🧹 两阶段路由模型显存已彻底清空。")
        except Exception:
            pass


def get_tool_router(
    mode: str = "router",
    bge_m3_path: Optional[str] = None,
    reranker_path: Optional[str] = None,
    device: str = "auto",
) -> BaseToolRouter:
    """
    统一工厂函数：按需创建工具路由器
    :param mode: 模式名称
                 - "router": 智能两阶段路由 (BGE-M3 + Reranker)，若权重缺失自动回退为 "full"
                 - "full":   全量工具直接注入 (零神经网络依赖，55+ 全部直接注入)
                 - "none":   纯对话，不注入任何工具
    :param bge_m3_path: BGE-M3 模型路径
    :param reranker_path: Reranker 模型路径
    :param device: 运行设备 ("auto", "cuda", "cpu")
    """
    mode = (mode or "router").strip().lower()

    if mode == "none":
        return NoToolRouter()
    elif mode == "full":
        return FullToolRouter()
    else:  # mode == "router" or "auto"
        m3_actual = resolve_model_path(
            bge_m3_path,
            "BGE_M3_PATH",
            "model/BAAI/bge-m3",
            DEFAULT_BGE_M3_PATH,
        )
        reranker_actual = resolve_model_path(
            reranker_path,
            "RERANKER_PATH",
            "model/BAAI/bge-reranker-base",
            DEFAULT_RERANKER_PATH,
        )
        return TwoStageToolRouter(
            m3_path=m3_actual,
            reranker_path=reranker_actual,
            device=device,
        )
