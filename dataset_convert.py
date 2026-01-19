from datasets import load_dataset
from tqdm import tqdm
import json

# 1️⃣ 选择要加载的 Hugging Face 数据集
# 示例：wikitext-103, openwebtext, alpaca, etc.
dataset_name = "/mnt/lustre/share_data/datasets/Nemotron-Pretraining-Dataset-sample"
subset = "Nemotron-CC-High-Quality"  # 可改成具体子集名
split = "train"  # "train" / "validation" / "test"

print(f"Loading {dataset_name}/{subset} split={split} ...")
ds = load_dataset(dataset_name, subset, split=split)

# 2️⃣ 选择文本字段（不同数据集字段名不一样）
# 你可以用 print(ds.column_names) 查看
text_field = "text"

# 3️⃣ 导出为 Megatron 预处理脚本可读的 JSONL（每行一个 {"text": "..."}）
output_jsonl = f"{dataset_name}/{subset}/{split}.jsonl"

print(f"Writing JSONL to {output_jsonl} ...")
with open(output_jsonl, "w", encoding="utf-8") as f:
    for item in tqdm(ds):
        text = item.get(text_field, "")
        if isinstance(text, list):
            text = " ".join([t for t in text if isinstance(t, str)])
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            continue
        # Megatron 预处理脚本 tools/preprocess_data.py 读取 {"text": "..."}
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

print(f"✅ Done! Saved to {output_jsonl}")
