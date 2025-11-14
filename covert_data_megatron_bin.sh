python3 tools/preprocess_data.py \
    --input /mnt/lustre/share_data/datasets/Nemotron-Pretraining-Dataset-sample/Nemotron-CC-High-Quality/train.jsonl \
    --output-prefix /mnt/lustre/share_data/datasets/Nemotron-Pretraining-Dataset-sample/Nemotron-CC-High-Quality \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /mnt/lustre/share_data/models/NVIDIA-Nemotron-Nano-12B-v2-Base \
    --append-eod \
    --workers 10 \
    --log-interval 1 \


