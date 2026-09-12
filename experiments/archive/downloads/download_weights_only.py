import os
import sys
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

target_dir = r"D:\ai_models\modelscope_cache\models\BAAI--bge-reranker-base"
target_file = os.path.join(target_dir, "model.safetensors")
temp_file = os.path.join(target_dir, "model.safetensors.downloading")

url = "https://www.modelscope.cn/api/v1/models/BAAI/bge-reranker-base/repo?Revision=master&FilePath=model.safetensors"

print(f"🎯 专项目标: 下载唯一缺失的核心权重 model.safetensors")
print(f"📁 目标目录: {target_dir}")

# 1. 检查已有的断点大小
downloaded_bytes = 0
if os.path.exists(temp_file):
    downloaded_bytes = os.path.getsize(temp_file)
    print(f"🔄 检测到未完成的临时文件，已下载: {downloaded_bytes / (1024*1024):.2f} MB，将执行断点续传...")

headers = {"User-Agent": "Mozilla/5.0"}
# 探测总大小
head_resp = requests.head(url, headers=headers, timeout=15)
total_bytes = int(head_resp.headers.get("Content-Length", 0))
print(f"📦 权重文件总大小: {total_bytes / (1024*1024):.2f} MB ({total_bytes} 字节)")

if downloaded_bytes > 0:
    headers["Range"] = f"bytes={downloaded_bytes}-"

chunk_size = 1024 * 1024  # 1MB 块
start_time = time.time()
last_print_time = start_time
last_bytes = downloaded_bytes

try:
    resp = requests.get(url, headers=headers, stream=True, timeout=30)
    if resp.status_code not in (200, 206):
        raise RuntimeError(f"HTTP 响应异常: {resp.status_code}")

    mode = "ab" if downloaded_bytes > 0 else "wb"
    with open(temp_file, mode) as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            downloaded_bytes += len(chunk)
            
            now = time.time()
            if now - last_print_time >= 5:  # 每 5 秒更新一次进度
                speed_mb = (downloaded_bytes - last_bytes) / (1024 * 1024) / (now - last_print_time)
                pct = downloaded_bytes / total_bytes * 100 if total_bytes else 0
                print(f"⏳ 进度: {downloaded_bytes/(1024*1024):.1f}/{total_bytes/(1024*1024):.1f} MB ({pct:.1f}%) | 速度: {speed_mb:.2f} MB/s")
                last_print_time = now
                last_bytes = downloaded_bytes

    # 下载完毕，重命名为正式文件名
    if os.path.exists(target_file):
        os.remove(target_file)
    os.rename(temp_file, target_file)
    cost = time.time() - start_time
    print(f"\n🎉 核心权重 model.safetensors 下载 100% 完成！耗时: {cost:.1f} 秒")
    print(f"✅ 文件落盘: {target_file} (大小: {os.path.getsize(target_file)/(1024*1024):.2f} MB)")

except Exception as e:
    print(f"\n❌ 下载中断: {e}")
    print(f"ℹ️ 已下载部分已保存在 {temp_file}，下次重新运行本脚本可继续断点续传。")
    sys.exit(1)
