#!/usr/bin/env python3
"""
MetaWorld MT50 평가 스크립트 - Subgoal Conditioning 버전

VLM → z_t → SubgoalDiffuserMLP.sample() → g_t
→ ConditionalUnet1DWithSubgoal(hidden_states, qpos, g_t) → actions

Usage:
    python eval_metaworld_subgoal.py \
        --checkpoint ~/experiments/tinyvla_metaworld_mt50_H/checkpoint-10000 \
        --base_model ~/models/Llava-Pythia-1.3B \
        --subgoal_ckpt ~/experiments/tinyvla_metaworld_mt50_H/subgoal_diffuser/subgoal_diffuser_best.pt \
        --action_head_ckpt ~/experiments/tinyvla_metaworld_mt50_H/action_head_subgoal/best_unet.pt \
        --num_rollouts 5 \
        --video_dir ~/experiments/tinyvla_metaworld_mt50_H/eval_videos_subgoal
"""

import os
import sys
import argparse
import pickle
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mujoco as _mj
import metaworld
import imageio
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_real_franka import llava_pythia_act_policy
from collect_metaworld_data import TASK_LANGUAGE
from aloha_scripts.constants import METAWORLD_TASKS
from policy_heads.models.subgoal_diffuser import SubgoalDiffuserMLP, GlobalLatentExtractor
from policy_heads.models.droid_unet_diffusion_subgoal import (
    ConditionalUnet1DWithSubgoal, build_subgoal_unet_from_checkpoint
)

# eval_metaworld.py와 동일한 난이도 분류
DIFFICULTY_MAP = {
    'easy': [
        'reach-v3', 'push-v3', 'pick-place-v3', 'door-open-v3',
        'drawer-open-v3', 'drawer-close-v3',
        'button-press-topdown-v3', 'button-press-topdown-wall-v3',
        'button-press-v3', 'button-press-wall-v3',
        'push-back-v3', 'push-wall-v3', 'reach-wall-v3', 'pick-place-wall-v3',
        'coffee-button-v3', 'coffee-push-v3',
        'faucet-open-v3', 'faucet-close-v3',
        'plate-slide-v3', 'plate-slide-side-v3',
        'plate-slide-back-v3', 'plate-slide-back-side-v3',
        'handle-press-v3', 'handle-press-side-v3',
        'lever-pull-v3', 'window-open-v3', 'window-close-v3', 'door-close-v3',
    ],
    'medium': [
        'basketball-v3', 'dial-turn-v3', 'sweep-v3', 'sweep-into-v3',
        'soccer-v3', 'stick-push-v3',
        'handle-pull-v3', 'handle-pull-side-v3',
        'hammer-v3', 'peg-unplug-side-v3', 'shelf-place-v3',
    ],
    'hard': [
        'coffee-pull-v3', 'disassemble-v3',
        'door-lock-v3', 'door-unlock-v3',
        'bin-picking-v3', 'box-close-v3',
    ],
    'very_hard': [
        'assembly-v3', 'stick-pull-v3', 'pick-out-of-hole-v3',
        'hand-insert-v3', 'peg-insert-side-v3',
    ],
}

TASK_DIFFICULTY = {}
for diff, tasks in DIFFICULTY_MAP.items():
    for t in tasks:
        TASK_DIFFICULTY[t] = diff

IMG_SIZE   = 224
VIDEO_SIZE = 480
CHUNK_SIZE = 16
ACTION_DIM = 4
MAX_STEPS  = 200
SUBGOAL_DIM = 2048
NUM_INFERENCE_TIMESTEPS = 5


def get_hidden_states(vlm_model, batch_i):
    """VLM backbone forward → hidden_states (float32)."""
    input_ids_mm, attention_mask_mm, past_kv, inputs_embeds, _ = \
        vlm_model.prepare_inputs_labels_for_multimodal(
            batch_i["input_ids"], batch_i["attention_mask"],
            None, None, batch_i["images"],
            images_r=batch_i["images_r"],
            visual_concat=vlm_model.visual_concat,
            states=batch_i["states"],
        )
    out = vlm_model.get_model()(
        input_ids=input_ids_mm,
        attention_mask=attention_mask_mm,
        past_key_values=past_kv,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return out[0].float()  # (1, seq_len, hidden_size)


def predict_actions(new_unet, noise_scheduler, hidden_states, states, subgoal):
    """DDIM denoising loop → predicted actions (1, CHUNK_SIZE, ACTION_DIM)."""
    B = hidden_states.shape[0]
    noisy_action = torch.randn((B, CHUNK_SIZE, ACTION_DIM), device=hidden_states.device)
    noise_scheduler.set_timesteps(NUM_INFERENCE_TIMESTEPS)

    for k in noise_scheduler.timesteps:
        noise_pred = new_unet(
            noisy_action, k,
            global_cond=hidden_states,
            states=states,
            subgoal=subgoal,
        )
        noisy_action = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=noisy_action,
        ).prev_sample

    return noisy_action  # (B, CHUNK_SIZE, ACTION_DIM)


def run_rollout(env, task, vlm_policy, new_unet, noise_scheduler,
                extractor, subgoal_model, stats, raw_lang, debug=False):
    env.set_task(task)
    obs, _ = env.reset()

    post_process = (
        lambda a: ((a + 1) / 2)
        * (stats['action_max'] - stats['action_min'])
        + stats['action_min']
    )

    all_time_actions = torch.zeros(
        [MAX_STEPS, MAX_STEPS + CHUNK_SIZE, ACTION_DIM],
        dtype=torch.float32,
    ).cuda()

    frames  = []
    success = False
    corner_cam_id  = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_CAMERA, 'corner')
    default_cam_id = env.mujoco_renderer.camera_id
    vlm_model = vlm_policy.policy

    with torch.inference_mode():
        for t in range(MAX_STEPS):
            # ── 이미지 렌더링 ───────────────────────────────
            img_raw = env.render()
            env.mujoco_renderer.camera_id = corner_cam_id
            img_video = env.mujoco_renderer.render(render_mode='rgb_array')
            env.mujoco_renderer.camera_id = default_cam_id
            img_video = cv2.rotate(img_video, cv2.ROTATE_180)
            frames.append(img_video.astype(np.uint8))

            img = cv2.resize(img_raw, (IMG_SIZE, IMG_SIZE))
            img_t = torch.from_numpy(img / 255.0).float()
            img_t = img_t.permute(2, 0, 1).unsqueeze(0)       # (1,3,224,224)
            curr_image = torch.cat([img_t, img_t], dim=0)      # (2,3,224,224)

            # ── 로봇 상태 정규화 ────────────────────────────
            qpos      = obs[:4].astype(np.float32)   # EEF xyz + gripper (pure proprioception)
            qpos_norm = (qpos - stats['qpos_mean']) / stats['qpos_std']
            robot_state = torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)  # (1,4)

            # ── VLM → hidden_states ─────────────────────────
            batch_i = vlm_policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
            hidden_states = get_hidden_states(vlm_model, batch_i)  # (1, seq_len, 2048)

            # ── z_t → g_t (subgoal) ────────────────────────
            z_t = extractor(hidden_states)       # (1, 2048)
            g_t = subgoal_model.sample(z_t)      # (1, 2048)

            # ── ConditionalUnet1DWithSubgoal → actions ──────
            all_actions = predict_actions(
                new_unet, noise_scheduler,
                hidden_states, robot_state.float(), g_t.float(),
            )  # (1, CHUNK_SIZE, ACTION_DIM)

            # ── Temporal aggregation ─────────────────────────
            all_time_actions[[t], t:t + CHUNK_SIZE] = all_actions.float()
            start_k = max(0, t - CHUNK_SIZE + 1)
            actions_for_curr_step = all_time_actions[start_k:t + 1, t]

            k_exp = 0.01
            exp_weights = np.exp(-k_exp * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(1).float()
            raw_action  = (actions_for_curr_step * exp_weights).sum(dim=0)

            # ── 역정규화 & 클리핑 ────────────────────────────
            action = post_process(raw_action.cpu().numpy())
            action = np.clip(action, -1.0, 1.0)

            if debug and t < 5:
                print(f'  [t={t}] action={action.round(3)}  qpos={qpos.round(3)}')

            obs, _, terminated, truncated, info = env.step(action)

            if info.get('success', False):
                success = True
                break

            if terminated or truncated:
                break

    return success, frames


def evaluate_task(task_name, mt50, vlm_policy, new_unet, noise_scheduler,
                  extractor, subgoal_model, stats, num_rollouts, video_dir, debug=False):
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

        success, frames = run_rollout(
            env, task, vlm_policy, new_unet, noise_scheduler,
            extractor, subgoal_model, stats, raw_lang,
            debug=(debug and rollout_id == 0),
        )

        if success:
            successes += 1

        result_tag = 'success' if success else 'fail'
        video_path = os.path.join(
            task_video_dir, f'rollout_{rollout_id:02d}_{result_tag}.mp4'
        )
        imageio.mimsave(video_path, frames, fps=30)

    env.close()
    return successes, num_rollouts


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
    print(f'\n난이도별 요약')
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


def main():
    parser = argparse.ArgumentParser(description='MetaWorld MT50 TinyVLA+Subgoal 평가')
    parser.add_argument('--checkpoint',      required=True,
                        help='학습된 TinyVLA 체크포인트 경로')
    parser.add_argument('--base_model',      required=True,
                        help='기본 VLM 경로 (예: ~/models/Llava-Pythia-1.3B)')
    parser.add_argument('--subgoal_ckpt',    required=True,
                        help='SubgoalDiffuserMLP 체크포인트 경로 (.pt)')
    parser.add_argument('--action_head_ckpt', required=True,
                        help='ConditionalUnet1DWithSubgoal 체크포인트 경로 (.pt)')
    parser.add_argument('--num_rollouts',    type=int, default=5)
    parser.add_argument('--video_dir',       default=None)
    parser.add_argument('--tasks',           nargs='+', default=None)
    parser.add_argument('--seed',            type=int, default=44)
    parser.add_argument('--debug',           action='store_true')
    args = parser.parse_args()

    checkpoint       = os.path.expanduser(args.checkpoint)
    base_model       = os.path.expanduser(args.base_model)
    subgoal_ckpt     = os.path.expanduser(args.subgoal_ckpt)
    action_head_ckpt = os.path.expanduser(args.action_head_ckpt)

    if args.video_dir is None:
        video_dir = os.path.join(os.path.dirname(checkpoint), 'eval_videos_subgoal/seed_44')
    else:
        video_dir = os.path.expanduser(args.video_dir)
    os.makedirs(video_dir, exist_ok=True)

    task_list = args.tasks if args.tasks else METAWORLD_TASKS

    print(f'체크포인트       : {checkpoint}')
    print(f'SubgoalDiffuser  : {subgoal_ckpt}')
    print(f'ActionHead       : {action_head_ckpt}')
    print(f'영상 저장        : {video_dir}')
    print(f'롤아웃/태스크    : {args.num_rollouts}')
    print(f'평가 태스크      : {len(task_list)}개')

    # ── 1. VLM 로딩 (frozen) ─────────────────────────────
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
    print('\n[1/4] VLM 로딩 중...')
    vlm_policy = llava_pythia_act_policy(policy_config)
    vlm_model  = vlm_policy.policy
    vlm_model.eval()

    # ── 2. SubgoalDiffuserMLP 로딩 ───────────────────────
    print('[2/4] SubgoalDiffuserMLP 로딩 중...')
    subgoal_model = SubgoalDiffuserMLP(latent_dim=SUBGOAL_DIM).cuda().float()
    ckpt = torch.load(subgoal_ckpt, map_location='cuda')
    state_dict = ckpt.get('model_state_dict', ckpt)
    subgoal_model.load_state_dict(state_dict)
    subgoal_model.eval()

    # ── 3. GlobalLatentExtractor ─────────────────────────
    extractor = GlobalLatentExtractor().cuda().float()
    extractor.eval()

    # ── 4. ConditionalUnet1DWithSubgoal 로딩 ─────────────
    print('[3/4] ConditionalUnet1DWithSubgoal 로딩 중...')
    new_unet = build_subgoal_unet_from_checkpoint(
        vlm_model.embed_out, subgoal_dim=SUBGOAL_DIM
    ).cuda().float()
    for m in new_unet.modules():
        if hasattr(m, 'dtype') and m.dtype == torch.bfloat16:
            m.dtype = torch.float32
    unet_state = torch.load(action_head_ckpt, map_location='cuda')
    new_unet.load_state_dict(unet_state)
    new_unet.eval()

    # ── 5. Noise Scheduler ───────────────────────────────
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon',
    )

    # ── 6. 데이터셋 통계 로딩 ────────────────────────────
    stats_path = os.path.join(os.path.dirname(checkpoint), 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    print(f'[4/4] 통계 파일 로딩: {stats_path}')

    # ── 7. MetaWorld 환경 초기화 ─────────────────────────
    print('\nMetaWorld MT50 초기화 중...')
    mt50 = metaworld.MT50(seed=args.seed)

    # ── 8. 평가 루프 ─────────────────────────────────────
    results = {}
    for task_name in tqdm(task_list, desc='태스크 평가'):
        if task_name not in mt50.train_classes:
            print(f'[WARN] {task_name} 이 MT50에 없습니다. 스킵.')
            continue

        suc, tot = evaluate_task(
            task_name, mt50, vlm_policy, new_unet, noise_scheduler,
            extractor, subgoal_model, stats,
            num_rollouts=args.num_rollouts,
            video_dir=video_dir,
            debug=args.debug,
        )
        results[task_name] = (suc, tot)
        pct  = suc / tot * 100
        diff = TASK_DIFFICULTY.get(task_name, '?')
        tqdm.write(f'  {task_name:<35} [{diff}]  {suc}/{tot} ({pct:.0f}%)')

    # ── 결과 출력 & 저장 ─────────────────────────────────
    print_results(results)

    results_path = os.path.join(os.path.dirname(checkpoint), 'eval_results_subgoal.json')

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
