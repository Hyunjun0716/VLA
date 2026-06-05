#!/bin/bash
# True E2E Subgoal Training — 완전 처음부터(base Llava-Pythia-1.3B)
#
# Usage:
#   bash scripts/train_e2e_subgoal.sh \
#       --base_model ~/models/Llava-Pythia-1.3B \
#       --output_dir ~/experiments/tinyvla_e2e_mt50 \
#       [--max_steps 30000] [--batch_size 1] [--grad_accum 16]

set -e

BASE_MODEL="$HOME/models/Llava-Pythia-1.3B"
OUTPUT_DIR="$HOME/experiments/tinyvla_e2e_mt50"
TASK_CONFIG="metaworld_mt50"
MAX_STEPS=10000
BATCH_SIZE=4
GRAD_ACCUM=64

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --base_model)   BASE_MODEL="$2";   shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --task_config)  TASK_CONFIG="$2";  shift 2 ;;
        --max_steps)    MAX_STEPS="$2";    shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
        --grad_accum)   GRAD_ACCUM="$2";   shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

BASE_MODEL=$(eval echo "$BASE_MODEL")
OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "===== E2E Subgoal Training ====="
echo "BASE_MODEL:   $BASE_MODEL"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "TASK_CONFIG:  $TASK_CONFIG"
echo "MAX_STEPS:    $MAX_STEPS"
echo "EFF_BATCH:    $((BATCH_SIZE * GRAD_ACCUM))"
echo "================================"

python3 "$SCRIPT_DIR/train_tinyvla_e2e.py" \
    --base_model    "$BASE_MODEL"   \
    --output_dir    "$OUTPUT_DIR"   \
    --task_config_name "$TASK_CONFIG" \
    --max_steps     "$MAX_STEPS"    \
    --batch_size    "$BATCH_SIZE"   \
    --grad_accum    "$GRAD_ACCUM"   \
    --lr_lora       1e-5            \
    --lr_action     1e-4            \
    --lr_subgoal    1e-4            \
    --subgoal_weight 0.1            \
    --delta_min     32              \
    --delta_max     32              \
    --eval_every    1000            \
    --save_every    2000            \
    --log_every     50              \
    --lora_r        64              \
    --lora_alpha    256             \
    --lora_module   "vit llm"
