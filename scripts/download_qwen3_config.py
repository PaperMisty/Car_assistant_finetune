from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-8B",
    cache_dir="./model/Qwen",
    # 忽略所有 safetensors, bin, pt, pth 等体积巨大的权重文件
    ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf"],
)
print("模型配置文件下载完成，保存路径：", model_dir)
