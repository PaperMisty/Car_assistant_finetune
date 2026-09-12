import os
import sys
from modelscope import snapshot_download

sys.stdout.reconfigure(encoding="utf-8")

model_id = "BAAI/bge-reranker-large"
cache_dir = r"D:\ai_models\modelscope_cache"

print(f"🚀 开始下载模型 {model_id} 至 {cache_dir} ...")
model_dir = snapshot_download(model_id, cache_dir=cache_dir)
print(f"✅ 下载完成！模型本地绝对路径: {model_dir}")
