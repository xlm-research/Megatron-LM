#!/usr/bin/env bash
set -euo pipefail

# ===== 基础环境与路径（按需修改） =====
WORKDIR="/mnt/lustre/yangbo1/Megatron-LM"
cd "$WORKDIR"

DATA_PATH="/mnt/lustre/share_data/datasets/Nemotron-Pretraining-Dataset-sample/Nemotron-CC-High-Quality/train_text_document"
TOKENIZER_PATH="/mnt/lustre/share_data/models/NVIDIA-Nemotron-Nano-12B-v2-Base"  # 目录内应包含 tokenizer.model 或兼容文件

# 输出目录
CKPT_DIR="${WORKDIR}/checkpoints/nemotron_nano_12B_v2"
DATACACHE_DIR="${WORKDIR}/data-cache/nemotron_nano_12B_v2"
TENSORBOARD_DIR="${WORKDIR}/tensorboard/nemotron_nano_12B_v2"
mkdir -p "$CKPT_DIR" "$DATACACHE_DIR" "$TENSORBOARD_DIR"

# ===== 分布式配置（按需修改） =====
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}

# ===== 必要环境变量 =====
export CUDA_DEVICE_MAX_CONNECTIONS=1   # YAML: sequence_parallel: true -> 需要设置为 1
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}


export export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export  CUDNN_LOGERR_DBG=1 
export CUDNN_LOGDEST_DBG=stderr




MODE_CONFIG=(
    --num-layers 9 #62
    --hidden-size 5120
    --ffn-hidden-size 20480
    --num-attention-heads 40
    --num-query-groups 8
    --kv-channels 128
    --seq-length  2048 #8192 
    --max-position-embeddings 131072

    # Hybrid (Mamba + Attention)
    --is-hybrid-model
    # --hybrid-override-pattern "M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M-"
    --hybrid-override-pattern "M-M*-M-M-"
    --hybrid-attention-ratio 0.0
    --hybrid-mlp-ratio 0.0

    # Mamba-specific
    --mamba-num-heads 128
    --mamba-head-dim 80
    --mamba-state-dim 128
    --mamba-num-groups 8

    # 训练精度/归一化/融合/注意力实现
    --normalization RMSNorm
    --attention-backend flash # eager|flash|fa2(=fused)|auto(默认)
    --disable-bias-linear
    --hidden-dropout 0.0
    --attention-dropout 0.0

    --position-embedding-type none
    # 词表
    --make-vocab-size-divisible-by 128
)


TRAIN_CONFIG=(
# 训练 batch / 迭代配置（来自 YAML）
    --global-batch-size 10 #768 
    --micro-batch-size 1
    --train-iters 100
    --eval-interval 100
    --eval-iters 1

    --bf16
    --fp8-format hybrid
    --reuse-grad-buf-for-mxfp8-param-ag
    --fp8-param-gather 
    --transformer-impl transformer_engine
    
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
)

DATA_CONFIG=(
    --data-path ${DATA_PATH}
    --tokenizer-type HuggingFaceTokenizer 
    --tokenizer-model ${TOKENIZER_PATH}
    # --split 99,1,0
)


OPTIMIZER_CONFIG=(
    --optimizer adam
    --lr 3.0e-4
    --min-lr 0.0
    --lr-decay-style cosine
    --lr-warmup-iters 0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1.0e-8
    --weight-decay 0.0
)

PARELLEL_CONFIG=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --distributed-backend nccl
    --overlap-grad-reduce
    --accumulate-allreduce-grads-in-fp32
    --use-distributed-optimizer

    --cross-entropy-loss-fusion

)


CKPT_RESUME_CONFIG=(
    --ckpt-format torch_dist
    --dist-ckpt-strictness log_all
    --ckpt-assume-constant-structure 
    --save ${CKPT_DIR}
    --save-interval 1000
    #--load
)


LOG_CONFIG=(
    --tensorboard-dir ${TENSORBOARD_DIR}
    --log-timers-to-tensorboard
    --log-throughput 
    --log-interval 1
)

CMD=(
  torchrun
    --nnodes ${NNODES}
    --nproc-per-node ${GPUS_PER_NODE}
    --node-rank ${NODE_RANK}
    --master-addr ${MASTER_ADDR}
    --master-port ${MASTER_PORT}
    pretrain_mamba.py
    ${MODE_CONFIG[@]}
    ${DATA_CONFIG[@]}
    ${TRAIN_CONFIG[@]}
    ${OPTIMIZER_CONFIG[@]}
    ${PARELLEL_CONFIG[@]}
    ${CKPT_RESUME_CONFIG[@]}
    ${LOG_CONFIG[@]}
)

echo "${CMD[@]}"
"${CMD[@]}"


