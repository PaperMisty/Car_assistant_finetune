import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from modelscope import snapshot_download

model_id = "BAAI/bge-reranker-base"
local_dir = r"D:\ai_models\modelscope_cache\models\BAAI--bge-reranker-base"

print(f"🚀 开始通过 Python + ModelScope 下载 {model_id} ...")
print(f"📁 目标存储路径: {local_dir}")
print("⚡ 策略: 仅下载 model.safetensors 核心权重与分词配置 (约 1.1GB)，自动跳过 bin / onnx 重复文件...")

t0 = time.time()
try:
    downloaded_dir = snapshot_download(
        model_id=model_id,
        local_dir=local_dir,
        ignore_file_pattern=["*.bin*", "*.onnx*", "onnx/*"]
    )
    cost = time.time() - t0
    print(f"\n🎉 恭喜！下载完成，耗时: {cost:.1f} 秒")
    print(f"✅ 模型本地绝对路径: {downloaded_dir}")
    
    # 列出目录文件与大小
    print("\n📂 下载文件清单:")
    for f in os.listdir(downloaded_dir):
        fp = os.path.join(downloaded_dir, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp) / (1024 * 1024)
            print(f"   • {f}: {sz:.2f} MB")
except Exception as e:
    print(f"\n❌ 下载过程中出错: {e}")
    sys.exit(1)
