#!/bin/bash
# 기본 TinyVLA + Subgoal TinyVLA를 5개 시드 × 10 rollout으로 평가
#
# Usage:
#   bash scripts/run_eval_all.sh \
#       --checkpoint ~/experiments/tinyvla_metaworld_mt50_H/checkpoint-10000 \
#       --base_model ~/models/Llava-Pythia-1.3B \
#       --subgoal_ckpt ~/experiments/tinyvla_metaworld_mt50_H/subgoal_diffuser/subgoal_diffuser_best.pt \
#       --action_head_ckpt ~/experiments/tinyvla_metaworld_mt50_H/action_head_subgoal/action_head_best.pt \
#       --output_dir ~/experiments/tinyvla_metaworld_mt50_H/eval_all

set -e

# ── 인자 파싱 ─────────────────────────────────────────────
CHECKPOINT=""
BASE_MODEL=""
SUBGOAL_CKPT=""
ACTION_HEAD_CKPT=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)      CHECKPOINT="$2";      shift 2 ;;
        --base_model)      BASE_MODEL="$2";      shift 2 ;;
        --subgoal_ckpt)    SUBGOAL_CKPT="$2";    shift 2 ;;
        --action_head_ckpt) ACTION_HEAD_CKPT="$2"; shift 2 ;;
        --output_dir)      OUTPUT_DIR="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# 필수 인자 확인
for var in CHECKPOINT BASE_MODEL SUBGOAL_CKPT ACTION_HEAD_CKPT OUTPUT_DIR; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: --$(echo $var | tr '[:upper:]' '[:lower:]' | tr '_' '-') 가 필요합니다."
        exit 1
    fi
done

CHECKPOINT=$(eval echo "$CHECKPOINT")
BASE_MODEL=$(eval echo "$BASE_MODEL")
SUBGOAL_CKPT=$(eval echo "$SUBGOAL_CKPT")
ACTION_HEAD_CKPT=$(eval echo "$ACTION_HEAD_CKPT")
OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")

SEEDS=(40 42 44)
NUM_ROLLOUTS=5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run_eval_all.log"
echo "===== 평가 시작: $(date) =====" | tee -a "$LOG_FILE"
echo "CHECKPOINT:      $CHECKPOINT"   | tee -a "$LOG_FILE"
echo "BASE_MODEL:      $BASE_MODEL"   | tee -a "$LOG_FILE"
echo "SUBGOAL_CKPT:    $SUBGOAL_CKPT" | tee -a "$LOG_FILE"
echo "ACTION_HEAD:     $ACTION_HEAD_CKPT" | tee -a "$LOG_FILE"
echo "OUTPUT_DIR:      $OUTPUT_DIR"   | tee -a "$LOG_FILE"
echo "SEEDS:           ${SEEDS[*]}"   | tee -a "$LOG_FILE"
echo "NUM_ROLLOUTS:    $NUM_ROLLOUTS" | tee -a "$LOG_FILE"
echo ""                               | tee -a "$LOG_FILE"

# ── 평가 루프 ─────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
    # # ── Baseline (subgoal 없음) ──────────────────────────
    # BASELINE_VIDEO_DIR="$OUTPUT_DIR/baseline/seed_${SEED}/videos"
    # BASELINE_JSON="$OUTPUT_DIR/baseline/seed_${SEED}/eval_results.json"
    # mkdir -p "$OUTPUT_DIR/baseline/seed_${SEED}"

    # echo "[$(date +%H:%M:%S)] BASELINE seed=$SEED 시작..." | tee -a "$LOG_FILE"
    # python "$SCRIPT_DIR/eval_metaworld.py" \
    #     --checkpoint "$CHECKPOINT" \
    #     --base_model "$BASE_MODEL" \
    #     --num_rollouts "$NUM_ROLLOUTS" \
    #     --seed "$SEED" \
    #     --video_dir "$BASELINE_VIDEO_DIR" \
    #     2>&1 | tee -a "$LOG_FILE"

    # # eval_metaworld.py는 checkpoint 상위에 저장 → 원하는 위치로 복사
    # SRC_JSON="$(dirname "$CHECKPOINT")/eval_results.json"
    # if [[ -f "$SRC_JSON" ]]; then
    #     cp "$SRC_JSON" "$BASELINE_JSON"
    # fi
    # echo "[$(date +%H:%M:%S)] BASELINE seed=$SEED 완료" | tee -a "$LOG_FILE"

    # ── Subgoal ──────────────────────────────────────────
    SUBGOAL_VIDEO_DIR="$OUTPUT_DIR/subgoal/seed_${SEED}/videos"
    SUBGOAL_JSON="$OUTPUT_DIR/subgoal/seed_${SEED}/eval_results_subgoal.json"
    mkdir -p "$OUTPUT_DIR/subgoal/seed_${SEED}"

    echo "[$(date +%H:%M:%S)] SUBGOAL seed=$SEED 시작..." | tee -a "$LOG_FILE"
    python "$SCRIPT_DIR/eval_metaworld_subgoal.py" \
        --checkpoint "$CHECKPOINT" \
        --base_model "$BASE_MODEL" \
        --subgoal_ckpt "$SUBGOAL_CKPT" \
        --action_head_ckpt "$ACTION_HEAD_CKPT" \
        --num_rollouts "$NUM_ROLLOUTS" \
        --seed "$SEED" \
        --video_dir "$SUBGOAL_VIDEO_DIR" \
        2>&1 | tee -a "$LOG_FILE"

    SRC_JSON_SG="$(dirname "$CHECKPOINT")/eval_results_subgoal.json"
    if [[ -f "$SRC_JSON_SG" ]]; then
        cp "$SRC_JSON_SG" "$SUBGOAL_JSON"
    fi
    echo "[$(date +%H:%M:%S)] SUBGOAL seed=$SEED 완료" | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
done

# # ── 최종 집계 ─────────────────────────────────────────────
# echo "===== 전체 결과 집계 =====" | tee -a "$LOG_FILE"
# python - "$OUTPUT_DIR" "${SEEDS[@]}" << 'PYEOF' 2>&1 | tee -a "$LOG_FILE"
# import sys, json, os

# output_dir = sys.argv[1]
# seeds = [int(s) for s in sys.argv[2:]]

# DIFF_KEYS = ['easy', 'medium', 'hard', 'very_hard', 'overall']

# def aggregate(mode, json_name):
#     agg = {d: {'success': 0, 'total': 0} for d in DIFF_KEYS}
#     found_seeds = []
#     for seed in seeds:
#         path = os.path.join(output_dir, mode, f'seed_{seed}', json_name)
#         if not os.path.exists(path):
#             print(f"  [WARN] 없음: {path}")
#             continue
#         with open(path) as f:
#             data = json.load(f)
#         summary = data.get('_summary', {})
#         for d in DIFF_KEYS:
#             if d in summary:
#                 agg[d]['success'] += summary[d]['success']
#                 agg[d]['total']   += summary[d]['total']
#         found_seeds.append(seed)

#     print(f"\n{'─'*50}")
#     print(f"[{mode.upper()}]  seeds={found_seeds}")
#     print(f"{'난이도':<12} {'성공/전체':>12}  {'성공률':>8}")
#     print(f"{'─'*50}")
#     for d in DIFF_KEYS:
#         s = agg[d]
#         rate = s['success'] / s['total'] * 100 if s['total'] > 0 else 0
#         print(f"  {d:<12} {s['success']:>5}/{s['total']:<5}  {rate:6.1f}%")

#     return agg

# base_agg = aggregate('baseline', 'eval_results.json')
# sg_agg   = aggregate('subgoal',  'eval_results_subgoal.json')

# # 비교 출력
# print(f"\n{'─'*60}")
# print(f"{'비교 (Subgoal - Baseline)':}")
# print(f"{'난이도':<12} {'Baseline':>10}  {'Subgoal':>10}  {'Δ':>8}")
# print(f"{'─'*60}")
# for d in DIFF_KEYS:
#     b = base_agg[d]
#     s = sg_agg[d]
#     b_rate = b['success'] / b['total'] * 100 if b['total'] > 0 else 0
#     s_rate = s['success'] / s['total'] * 100 if s['total'] > 0 else 0
#     delta  = s_rate - b_rate
#     sign   = '+' if delta >= 0 else ''
#     print(f"  {d:<12} {b_rate:9.1f}%  {s_rate:9.1f}%  {sign}{delta:6.1f}%")

# # 집계 결과 JSON 저장
# final = {
#     'seeds': seeds,
#     'baseline': {d: {'rate': base_agg[d]['success']/base_agg[d]['total'] if base_agg[d]['total']>0 else 0,
#                      **base_agg[d]} for d in DIFF_KEYS},
#     'subgoal':  {d: {'rate': sg_agg[d]['success']/sg_agg[d]['total'] if sg_agg[d]['total']>0 else 0,
#                      **sg_agg[d]} for d in DIFF_KEYS},
# }
# out_path = os.path.join(output_dir, 'final_comparison.json')
# with open(out_path, 'w') as f:
#     json.dump(final, f, indent=2)
# print(f"\n집계 결과 저장: {out_path}")
# PYEOF

# echo "===== 평가 완료: $(date) =====" | tee -a "$LOG_FILE"
