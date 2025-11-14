
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/lustre/share_data/models/NVIDIA-Nemotron-Nano-12B-v2-Base")
tokenizer.save_pretrained("/mnt/lustre/share_data/models/NVIDIA-Nemotron-Nano-12B-v2-Base-tokenizer")