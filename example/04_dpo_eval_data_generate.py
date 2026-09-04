import os
import json
from datasets import load_dataset

# =========================
# 配置
# =========================
# 本地 ultrafeedback_binarized 数据集
DATASET_PATH = "./data/ultrafeedback_binarized"

# EvalScope 自定义 QA 数据集目录
OUTPUT_DIR = "./custom_eval/text/qa"

# subset 名称
SUBSET_NAME = "ultrafeedback"

# 最终 EvalScope 数据文件
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"{SUBSET_NAME}.jsonl"
)

DATA_SIZE = 50
SEED = 42

def get_eval_data():
    """
    从 ultrafeedback_binarized 的 test_prefs 中
    随机提取 DATA_SIZE 条数据，
    构造成 EvalScope general_qa 所需的答案评估集。
    """

    # 1. 加载数据集
    dataset = load_dataset(DATASET_PATH)["test_prefs"]

    print("原始数据量:", len(dataset))
    print("原始字段:", dataset.column_names)

    # 2. 固定随机种子并随机抽取 500 条
    dataset = dataset.shuffle(seed=SEED)

    data_size = min(DATA_SIZE, len(dataset))
    # 从数据集里面选取固定条数的数据出来
    dataset = dataset.select(range(data_size))

    # 3. 创建 EvalScope 所需目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. 保存为 JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for item in dataset:

            eval_item = {
                "query": item["prompt"],
                "response":"dummy response"
            }

            f.write(
                json.dumps(eval_item, ensure_ascii=False) + "\n"
            )

    print(f"\n评估集数据量: {data_size}")
    print(f"评估集已保存到: {OUTPUT_FILE}")

    # 5. 打印第一条数据检查
    print("\n第一条数据:")
    print({
        "query": dataset[0]["prompt"]
    })

if __name__ == "__main__":
    get_eval_data()
