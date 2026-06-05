#!/bin/bash
# ============================================================
# TinyVLA MetaWorld MT50 학습 스크립트 (단일 RTX 5090, 32GB)
#
# 논문 설정 (8x GPU):
#   per_device_batch=32, num_gpus=8, accum=1 → effective_batch=256
#
# 단일 GPU 동등 설정:
#   per_device_batch=8,  num_gpus=1, accum=32 → effective_batch=256 ✓
#
# 사용법:
#   bash scripts/train_metaworld.sh
# ============================================================

# ── 사전 다운로드 필요한 VLM 경로 ────────────────────────────
# 처음 실행 전 아래 명령으로 다운로드:
#   huggingface-cli download lesjie/Llava-Pythia-400M --local-dir ~/models/Llava-Pythia-400M
#   huggingface-cli download lesjie/Llava-Pythia-1.3B --local-dir ~/models/Llava-Pythia-1.3B
#
# TinyVLA-S (400M) - 빠른 학습/실험용
# VLM_PATH=~/models/Llava-Pythia-400M
# TinyVLA-B (700M) - 중간 성능/속도
# VLM_PATH=/home/jun/models/Llava-Pythia-700M
# TinyVLA-H (1.3B) - 논문 최고 성능
VLM_PATH=~/models/Llava-Pythia-1.3B

# ── 출력 경로 ─────────────────────────────────────────────
OUTPUT=~/experiments/tinyvla_metaworld_camera_coner2

mkdir -p $OUTPUT
cp ./scripts/train_metaworld.sh $OUTPUT

deepspeed --master_port 29601 --num_gpus=1 ./train_tinyvla.py \
  --deepspeed scripts/zero2.json \
  \
  `# ── LoRA 설정 (논문과 동일) ──` \
  --lora_enable True \
  --lora_module 'vit llm' \
  --lora_r 64 \
  --lora_alpha 256 \
  \
  `# ── MetaWorld 데이터 설정 (state_dim=4: EEF xyz+gripper) ──` \
  --task_name "metaworld_mt50" \
  --action_dim 4 \
  --state_dim 4 \
  \
  `# ── 모델 설정 ──` \
  --load_pretrain False \
  --model_name_or_path $VLM_PATH \
  --version v0 \
  --pretrain_image_size 224 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  \
  `# ── 학습 설정 ──` \
  --bf16 True \
  --tf32 True \
  --output_dir $OUTPUT \
  --max_steps 10000 \
  `# 논문: 32×8=256 유효배치 → 단일GPU: 8×32=256 동일` \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --non_lora_lr 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.005 \
  --lr_scheduler_type "cosine" \
  --gradient_checkpointing True \
  \
  `# ── 저장/로깅 설정 ──` \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 5 \
  --logging_steps 10 \
  --model_max_length 2048 \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  \
  `# ── Action head 설정 (논문과 동일) ──` \
  --action_head_type droid_diffusion \
  --concat "token_cat" \
  \
  --seed 42 \
  \
  `# ── 모니터링 ──` \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log

# 체크포인트에 preprocessor_config.json 복사 (inference에 필요)
for dir in "$OUTPUT"/*/; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp llava-pythia/preprocessor_config.json "$dir"
    fi
done

echo "학습 완료! 결과: $OUTPUT"
