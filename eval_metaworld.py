#!/usr/bin/env python3
"""
MetaWorld MT50 평가 스크립트
- 학습된 TinyVLA 체크포인트로 50개 태스크 평가
- 태스크당 N회 롤아웃, 모든 영상 저장 (success/fail 구분)
- Easy / Medium / Hard / VeryHard 난이도별 성공률 리포트

Usage:
    python eval_metaworld.py \
        --checkpoint ~/experiments/tinyvla_metaworld_mt50/checkpoint-10000 \
        --base_model ~/models/Llava-Pythia-700M \
        --num_rollouts 5 \
        --video_dir ~/experiments/tinyvla_metaworld_mt50/eval_videos

    # 특정 태스크만 테스트:
    python eval_metaworld.py ... --tasks reach-v3 push-v3
"""

import os
import sys
import argparse
import pickle
import json

import cv2
import numpy as np
import torch
import mujoco as _mj
import metaworld
import imageio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_real_franka import llava_pythia_act_policy
from collect_metaworld_data import TASK_LANGUAGE
from aloha_scripts.constants import METAWORLD_TASKS

# ──────────────────────────────────────────────────────────
# 난이도 분류 (MetaWorld 논문 Yu et al. 2019 기준)
# Easy(28) / Medium(11) / Hard(6) / VeryHard(5) = 50
# ──────────────────────────────────────────────────────────
DIFFICULTY_MAP = {
    'easy': [
        'button-press-v3', 'button-press-topdown-v3',
        'button-press-topdown-wall-v3', 'button-press-wall-v3',
        'coffee-button-v3', 'dial-turn-v3', 'door-close-v3', 'door-lock-v3',
        'door-open-v3', 'door-unlock-v3', 'drawer-close-v3', 'drawer-open-v3',
        'faucet-close-v3', 'faucet-open-v3', 'handle-press-v3', 'handle-press-side-v3',
        'handle-pull-v3', 'handle-pull-side-v3', 'lever-pull-v3', 'plate-slide-v3',
        'plate-slide-back-v3', 'plate-slide-back-side-v3', 'plate-slide-side-v3',
        'reach-v3', 'reach-wall-v3',
        'window-close-v3', 'window-open-v3', 'peg-unplug-side-v3',
    ],
    'medium': [
        'basketball-v3', 'bin-picking-v3', 'box-close-v3', 'coffee-pull-v3',
        'coffee-push-v3', 'hammer-v3', 'peg-insert-side-v3', 'push-wall-v3',
        'soccer-v3', 'sweep-v3', 'sweep-into-v3',
    ],
    'hard': [
        'assembly-v3', 'hand-insert-v3', 'pick-out-of-hole-v3',
        'pick-place-v3', 'push-v3', 'push-back-v3',
    ],
    'very_hard': [
        'shelf-place-v3', 'disassemble-v3', 'stick-pull-v3',
        'stick-push-v3', 'pick-place-wall-v3',
    ],
}

TASK_DIFFICULTY = {}
for diff, tasks in DIFFICULTY_MAP.items():
    for t in tasks:
        TASK_DIFFICULTY[t] = diff

IMG_SIZE   = 224
VIDEO_SIZE = 480   # 영상 저장 해상도 (원본 렌더 크기)
CHUNK_SIZE = 16
ACTION_DIM = 4
MAX_STEPS  = 500


# ──────────────────────────────────────────────────────────
# 단일 롤아웃
# ──────────────────────────────────────────────────────────
def run_rollout(env, task, policy, stats, raw_lang, debug=False):
    """
    단일 롤아웃 수행.

    Returns:
        success (bool): 태스크 성공 여부
        frames  (list): RGB uint8 프레임 리스트
    """
    env.set_task(task)
    env._freeze_rand_vec = True  # 매 rollout마다 물체 위치 랜덤화
    obs, _ = env.reset()

    post_process = (
        lambda a: ((a + 1) / 2)
        * (stats['action_max'] - stats['action_min'])
        + stats['action_min']
    )

    # Temporal aggregation 버퍼 (max_steps, max_steps+chunk, action_dim)
    all_time_actions = torch.zeros(
        [MAX_STEPS, MAX_STEPS + CHUNK_SIZE, ACTION_DIM],
        dtype=torch.float32,
    ).cuda()

    frames  = []
    success = False
    corner2_cam_id = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_CAMERA, 'corner2')
    env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]      # corner2 논문 카메라 위치
    env.mujoco_renderer.camera_id = corner2_cam_id

    with torch.inference_mode():
        for t in range(MAX_STEPS):
            # ── 이미지 ──────────────────────────────────────
            img_raw = env.mujoco_renderer.render(render_mode='rgb_array')  # corner2 카메라
            img_video = cv2.rotate(img_raw, cv2.ROTATE_180)               # 영상용 180° 회전
            frames.append(img_video.astype(np.uint8))

            img = cv2.resize(img_raw, (IMG_SIZE, IMG_SIZE))     # (224,224,3) 학습 데이터와 동일한 cv2.bilinear resize

            # process_batch_to_llava 는 (2,C,H,W) 를 받아 chunk 로 분리
            img_t = torch.from_numpy(img / 255.0).float()
            img_t = img_t.permute(2, 0, 1).unsqueeze(0)        # (1,3,224,224)
            curr_image = torch.cat([img_t, img_t], dim=0)       # (2,3,224,224)

            # ── 로봇 상태 정규화 ────────────────────────────
            qpos      = obs[:4].astype(np.float32)   # EEF xyz + gripper (pure proprioception)
            qpos_norm = (qpos - stats['qpos_mean']) / stats['qpos_std']
            robot_state = torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)

            # ── 정책 쿼리 ───────────────────────────────────
            batch       = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
            all_actions = policy.policy(**batch, eval=True)  # (1, CHUNK_SIZE, ACTION_DIM)

            # ── Temporal aggregation ─────────────────────────
            # 실제로 기록된 행만 사용: query timestep k ≤ t < k+CHUNK_SIZE
            all_time_actions[[t], t:t + CHUNK_SIZE] = all_actions.float()
            start_k = max(0, t - CHUNK_SIZE + 1)
            actions_for_curr_step = all_time_actions[start_k:t + 1, t]  # (min(t+1,CHUNK_SIZE), ACTION_DIM)

            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(1).float()
            raw_action  = (actions_for_curr_step * exp_weights).sum(dim=0)

            # ── 역정규화 & 클리핑 ────────────────────────────
            action = post_process(raw_action.cpu().numpy())  # (4,)

            if debug and t < 5:
                print(f'  [t={t}] raw_model={all_actions[0,0].float().cpu().numpy().round(3)}'
                      f'  denorm={action.round(3)}'
                      f'  qpos={qpos.round(3)}')

            # ── 환경 스텝 ────────────────────────────────────
            obs, _, terminated, truncated, info = env.step(action)

            if info.get('success', False):
                success = True
                break

            if terminated or truncated:
                break

    return success, frames


# ──────────────────────────────────────────────────────────
# 단일 태스크 평가
# ──────────────────────────────────────────────────────────
def evaluate_task(task_name, mt50, policy, stats, num_rollouts, video_dir, debug=False):
    env_cls   = mt50.train_classes[task_name]
    env       = env_cls(render_mode='rgb_array')
    task_list = [t for t in mt50.train_tasks if t.env_name == task_name]
    raw_lang  = TASK_LANGUAGE.get(
        task_name, task_name.replace('-v3', '').replace('-', ' ')
    )

    task_video_dir = os.path.join(video_dir, task_name)
    os.makedirs(task_video_dir, exist_ok=True)

    successes = 0
    for rollout_id in range(num_rollouts):
        task = task_list[rollout_id % len(task_list)]

        success, frames = run_rollout(env, task, policy, stats, raw_lang,
                                      debug=(debug and rollout_id == 0))

        if success:
            successes += 1

        # 영상 저장 (success / fail 구분)
        result_tag = 'success' if success else 'fail'
        video_path = os.path.join(
            task_video_dir,
            f'rollout_{rollout_id:02d}_{result_tag}.mp4',
        )
        imageio.mimsave(video_path, frames, fps=30)

    env.close()
    return successes, num_rollouts


# ──────────────────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────────────────
def print_results(results):
    print('\n' + '=' * 60)
    print(f'{"Task":<35} {"Diff":<10} {"Success"}')
    print('-' * 60)

    diff_stats = {d: {'success': 0, 'total': 0} for d in DIFFICULTY_MAP}

    for task_name, (suc, tot) in results.items():
        diff = TASK_DIFFICULTY.get(task_name, 'unknown')
        pct  = suc / tot * 100 if tot > 0 else 0
        print(f'{task_name:<35} [{diff:<9}]  {suc}/{tot} ({pct:5.1f}%)')
        if diff in diff_stats:
            diff_stats[diff]['success'] += suc
            diff_stats[diff]['total']   += tot

    print('=' * 60)
    print(f'\n{"난이도별 요약":}')
    print('-' * 40)
    overall_suc, overall_tot = 0, 0
    label_map = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard', 'very_hard': 'VeryHard'}
    for diff_key in ['easy', 'medium', 'hard', 'very_hard']:
        s = diff_stats[diff_key]
        pct = s['success'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f'  {label_map[diff_key]:<10}: {s["success"]}/{s["total"]} ({pct:.1f}%)')
        overall_suc += s['success']
        overall_tot += s['total']

    overall_pct = overall_suc / overall_tot * 100 if overall_tot > 0 else 0
    print(f'  {"Overall":<10}: {overall_suc}/{overall_tot} ({overall_pct:.1f}%)')
    print('=' * 60)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='MetaWorld MT50 TinyVLA 평가')
    parser.add_argument('--checkpoint',   required=True,
                        help='학습된 체크포인트 경로 (예: ~/experiments/.../checkpoint-10000)')
    parser.add_argument('--base_model',   required=True,
                        help='기본 VLM 경로 (예: ~/models/Llava-Pythia-700M)')
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='태스크당 롤아웃 횟수 (기본: 5)')
    parser.add_argument('--video_dir',    default=None,
                        help='영상 저장 디렉토리 (기본: checkpoint 상위/eval_videos)')
    parser.add_argument('--tasks',        nargs='+', default=None,
                        help='평가할 태스크 목록 (기본: 전체 50개)')
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--debug',        action='store_true',
                        help='첫 번째 롤아웃에서 액션 값 출력')
    args = parser.parse_args()

    # 경로 정규화
    checkpoint = os.path.expanduser(args.checkpoint)
    base_model = os.path.expanduser(args.base_model)

    if args.video_dir is None:
        video_dir = os.path.join(os.path.dirname(checkpoint), 'eval_videos/seed_42')
    else:
        video_dir = os.path.expanduser(args.video_dir)
    os.makedirs(video_dir, exist_ok=True)

    # 평가할 태스크 목록
    task_list = args.tasks if args.tasks else METAWORLD_TASKS

    print(f'체크포인트 : {checkpoint}')
    print(f'기본 모델  : {base_model}')
    print(f'영상 저장  : {video_dir}')
    print(f'롤아웃/태스크: {args.num_rollouts}')
    print(f'평가 태스크: {len(task_list)}개')

    # ── 정책 로딩 ────────────────────────────────────────
    policy_config = {
        'model_path':       checkpoint,
        'model_base':       base_model,
        'enable_lora':      True,
        'conv_mode':        'pythia',
        'action_head':      'droid_diffusion',
        'action_head_type': 'droid_diffusion',
        'chunk_size':       CHUNK_SIZE,
        'action_dim':       ACTION_DIM,
    }
    print('\n모델 로딩 중...')
    policy = llava_pythia_act_policy(policy_config)
    policy.policy.eval()
    print('모델 로딩 완료')

    # ── 데이터셋 통계 로딩 ───────────────────────────────
    stats_path = os.path.join(os.path.dirname(checkpoint), 'dataset_stats.pkl')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f'dataset_stats.pkl 를 찾을 수 없습니다: {stats_path}\n'
            '학습 완료 후 output_dir 에 자동 저장됩니다.'
        )
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    print(f'통계 파일 로딩: {stats_path}')

    # ── MetaWorld 환경 초기화 ─────────────────────────────
    print('\nMetaWorld MT50 초기화 중...')
    mt50 = metaworld.MT50(seed=args.seed)

    # ── 평가 루프 ─────────────────────────────────────────
    results = {}
    for task_name in tqdm(task_list, desc='태스크 평가'):
        if task_name not in mt50.train_classes:
            print(f'[WARN] {task_name} 이 MT50에 없습니다. 스킵.')
            continue

        suc, tot = evaluate_task(
            task_name, mt50, policy, stats,
            num_rollouts=args.num_rollouts,
            video_dir=video_dir,
            debug=args.debug,
        )
        results[task_name] = (suc, tot)
        pct = suc / tot * 100
        diff = TASK_DIFFICULTY.get(task_name, '?')
        tqdm.write(f'  {task_name:<35} [{diff}]  {suc}/{tot} ({pct:.0f}%)')

    # ── 결과 출력 & 저장 ─────────────────────────────────
    print_results(results)

    # JSON 저장
    results_path = os.path.join(os.path.dirname(checkpoint), 'eval_results.json')

    # 태스크별 결과
    json_results = {k: {'success': v[0], 'total': v[1], 'rate': v[0]/v[1]}
                    for k, v in results.items()}

    # 난이도별 집계
    diff_stats = {d: {'success': 0, 'total': 0} for d in DIFFICULTY_MAP}
    for task_name, (suc, tot) in results.items():
        diff = TASK_DIFFICULTY.get(task_name)
        if diff in diff_stats:
            diff_stats[diff]['success'] += suc
            diff_stats[diff]['total']   += tot
    summary = {}
    overall_suc, overall_tot = 0, 0
    for diff_key in ['easy', 'medium', 'hard', 'very_hard']:
        s = diff_stats[diff_key]
        summary[diff_key] = {
            'success': s['success'],
            'total':   s['total'],
            'rate':    s['success'] / s['total'] if s['total'] > 0 else 0.0,
        }
        overall_suc += s['success']
        overall_tot += s['total']
    summary['overall'] = {
        'success': overall_suc,
        'total':   overall_tot,
        'rate':    overall_suc / overall_tot if overall_tot > 0 else 0.0,
    }

    json_results['_summary'] = summary
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'\n결과 저장: {results_path}')


if __name__ == '__main__':
    main()
