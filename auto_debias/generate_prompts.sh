MODEL_VERSION=bert-base-uncased
MODEL_NAME=bert
SEED=42
BATCH_SIZE=1000
DEBIAS_TYPE=gender
RUN_NO=run00
DATA_DIR=./data/
MAX_PROMPT_LEN=5
TOP_K=100
SAVE_DIR=./prompts/
NUM_WORKERS=4
LR=2e-5
ACCUMULATE_GRAD_BATCHES=1
GRADIENT_CLIP_VAL=1.0
MAX_EPOCHS=100
OUTPUT_DIR=./out/
GPUS=2
PATIENCE=10
CKPT_PATH=./ckpts/
PRECISION=16

python generate_prompts.py \
    --model_version bert-base-uncased \
    --model_name bert \
    --seed 42 \
    --batch_size 1000 \
    --debias_type gender \
    --run_name run00 \
    --data_dir ./data/ \
    --max_prompt_len 5 \
    --top_k 100 \
    --output_dir ./prompts/ \
    --num_workers 4 \
    --lr 2e-5 \
    --accumulate_grad_batches 1 \
    --gradient_clip_val 1.0 \
    --max_epochs 100 \
    --output_dir ./out/ \
    --gpus 2 \
    --patience 10 \
    --precision 16
