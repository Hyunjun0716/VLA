#!/usr/bin/env python3
"""
MetaWorld MT50 데이터 수집 스크립트
- 50개 task × 50개 demonstration
- TinyVLA h5py 포맷으로 저장 (action_dim=4, state_dim=4)

Usage:
    python collect_metaworld_data.py --save_dir /home/jun/data/metaworld --num_demos 50
"""

import os
import argparse
import warnings
import numpy as np
import h5py
import metaworld
import metaworld.policies as policies
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Task -> Language description 매핑
# ──────────────────────────────────────────────
TASK_LANGUAGE = {
    'assembly-v3':                  'pick up the wrench and insert it onto the peg',
    'basketball-v3':                'dunk the basketball into the basket',
    'bin-picking-v3':               'pick up the block and place it into the bin',
    'box-close-v3':                 'grab the lid and put it on the box',
    'button-press-topdown-v3':      'press the button from the top',
    'button-press-topdown-wall-v3': 'press the button from the top over the wall',
    'button-press-v3':              'press the button',
    'button-press-wall-v3':         'press the button over the wall',
    'coffee-button-v3':             'press the coffee machine button',
    'coffee-pull-v3':               'pull the coffee mug to the goal',
    'coffee-push-v3':               'push the coffee mug to the goal',
    'dial-turn-v3':                 'turn the dial',
    'disassemble-v3':               'pick the nut off the peg',
    'door-close-v3':                'close the door',
    'door-lock-v3':                 'lock the door by rotating the lock clockwise',
    'door-open-v3':                 'open the door',
    'door-unlock-v3':               'unlock the door by rotating the lock',
    'hand-insert-v3':               'push the object into the hole',
    'drawer-close-v3':              'push the drawer closed',
    'drawer-open-v3':               'open the drawer',
    'faucet-open-v3':               'turn the faucet counterclockwise to open it',
    'faucet-close-v3':              'turn the faucet clockwise to close it',
    'hammer-v3':                    'hammer the nail into the wall',
    'handle-press-side-v3':         'press the handle down from the side',
    'handle-press-v3':              'press the handle down',
    'handle-pull-side-v3':          'pull up the handle from the side',
    'handle-pull-v3':               'pull up the handle',
    'lever-pull-v3':                'pull the lever down',
    'pick-place-wall-v3':           'pick up the puck and place it at the goal over the wall',
    'pick-out-of-hole-v3':          'pick the puck out of the hole and place it at the goal',
    'pick-place-v3':                'pick up the puck and place it at the goal',
    'plate-slide-v3':               'slide the plate to the goal',
    'plate-slide-side-v3':          'slide the plate sideways to the goal',
    'plate-slide-back-v3':          'slide the plate back to the goal',
    'plate-slide-back-side-v3':     'slide the plate back and sideways to the goal',
    'peg-insert-side-v3':           'insert the peg into the hole from the side',
    'peg-unplug-side-v3':           'unplug the peg from the side hole',
    'soccer-v3':                    'kick the soccer ball into the goal',
    'stick-push-v3':                'use the stick to push the object to the goal',
    'stick-pull-v3':                'use the stick to pull the object to the goal',
    'push-v3':                      'push the puck to the goal',
    'push-wall-v3':                 'push the puck to the goal around the wall',
    'push-back-v3':                 'push the puck back to the goal',
    'reach-v3':                     'reach the goal position',
    'reach-wall-v3':                'reach the goal position over the wall',
    'shelf-place-v3':               'pick up the object and place it on the shelf',
    'sweep-into-v3':                'sweep the object into the hole',
    'sweep-v3':                     'sweep the object to the goal',
    'window-open-v3':               'slide the window open',
    'window-close-v3':              'slide the window closed',
}


def get_policy(task_name: str):
    """task 이름 → scripted expert policy 인스턴스 반환"""
    # 자동 변환 규칙과 실제 클래스명이 다른 예외 케이스
    POLICY_NAME_OVERRIDES = {
        'peg-insert-side-v3': 'SawyerPegInsertionSideV3Policy',
    }
    if task_name in POLICY_NAME_OVERRIDES:
        class_name = POLICY_NAME_OVERRIDES[task_name]
    else:
        # 예: 'door-open-v3' → 'SawyerDoorOpenV3Policy'
        parts = task_name.replace('-v3', '').split('-')
        class_name = 'Sawyer' + ''.join(p.capitalize() for p in parts) + 'V3Policy'
    policy_cls = getattr(policies, class_name)
    return policy_cls()


def collect_episode(env, policy, task, max_steps: int, img_size: int):
    """
    한 에피소드 수집. 성공한 경우에만 데이터 반환, 실패 시 None 반환.

    반환 형태 (TinyVLA h5py 포맷):
        actions : (T, 4)  — delta xyz + gripper
        qpos    : (T, 7)  — EEF xyz + gripper + goal/obj xyz  (obs[:7])
        qvel    : (T, 7)  — zeros (MetaWorld는 joint velocity 미제공)
        images  : (T, img_size, img_size, 3)
    """
    import cv2

    env.set_task(task)
    env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]      # corner2 논문 카메라 위치
    obs, _ = env.reset()

    import mujoco as _mj
    corner2_cam_id = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_CAMERA, 'corner2')
    orig_cam_id = env.mujoco_renderer.camera_id
    env.mujoco_renderer.camera_id = corner2_cam_id

    actions, qpos_list, qvel_list, images = [], [], [], []
    success = False

    for _ in range(max_steps):
        img = env.mujoco_renderer.render(render_mode='rgb_array')  # corner2 카메라
        if img_size != img.shape[0]:
            img = cv2.resize(img, (img_size, img_size))

        action = policy.get_action(obs)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # obs 저장은 action 실행 전
        images.append(img.astype(np.uint8))
        qpos_list.append(obs[:7].astype(np.float32))   # EEF xyz + gripper + goal/obj xyz
        qvel_list.append(np.zeros(7, dtype=np.float32))
        actions.append(action)

        obs, _, terminated, truncated, info = env.step(action)

        if info.get('success', False):
            success = True
            break

        if terminated or truncated:
            break

    if not success:
        return None

    return {
        'actions': np.array(actions, dtype=np.float32),   # (T, 4)
        'qpos':    np.array(qpos_list, dtype=np.float32), # (T, 7)
        'qvel':    np.array(qvel_list, dtype=np.float32), # (T, 7)
        'images':  np.array(images, dtype=np.uint8),      # (T, H, W, 3)
    }


def save_episode(data: dict, filepath: str, language: str):
    """TinyVLA h5py 포맷으로 저장"""
    with h5py.File(filepath, 'w') as root:
        root.attrs['sim'] = True        # datasets.py의 is_sim 분기에 사용
        root.attrs['compress'] = False

        root.create_dataset('action', data=data['actions'])  # (T, 4)

        # language_raw: 가변 길이 문자열
        dt = h5py.special_dtype(vlen=str)
        root.create_dataset('language_raw',
                            data=np.array([language], dtype=object),
                            dtype=dt)

        obs_grp = root.create_group('observations')
        obs_grp.create_dataset('qpos', data=data['qpos'])   # (T, 4)
        obs_grp.create_dataset('qvel', data=data['qvel'])   # (T, 4)

        img_grp = obs_grp.create_group('images')
        img_grp.create_dataset('front', data=data['images'],  # (T, H, W, 3)
                               compression='gzip', compression_opts=4)


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    mt50 = metaworld.MT50(seed=args.seed)
    task_names = list(mt50.train_classes.keys())

    print(f"\n{'='*60}")
    print(f"MetaWorld MT50 데이터 수집")
    print(f"  태스크 수  : {len(task_names)}")
    print(f"  데모 수/태스크: {args.num_demos}")
    print(f"  저장 경로  : {args.save_dir}")
    print(f"  이미지 크기: {args.img_size}x{args.img_size}")
    print(f"  최대 스텝  : {args.max_steps}")
    print(f"{'='*60}\n")

    for task_name in tqdm(task_names, desc='전체 진행'):
        task_dir = os.path.join(args.save_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        # 이미 충분한 demos가 있으면 스킵
        existing = len([f for f in os.listdir(task_dir) if f.endswith('.hdf5')])
        if existing >= args.num_demos:
            tqdm.write(f'[SKIP] {task_name}: {existing}개 이미 존재')
            continue

        env_cls = mt50.train_classes[task_name]
        env = env_cls(render_mode='rgb_array')
        task_list = [t for t in mt50.train_tasks if t.env_name == task_name]
        policy = get_policy(task_name)
        language = TASK_LANGUAGE.get(task_name,
                                     task_name.replace('-v3', '').replace('-', ' '))

        collected = existing
        attempts = 0
        max_attempts = args.num_demos * 20  # 최대 시도 횟수

        inner_bar = tqdm(total=args.num_demos, initial=collected,
                         desc=f'  {task_name}', leave=False)

        while collected < args.num_demos and attempts < max_attempts:
            # task_list를 순환하며 다양한 goal 위치 사용
            task = task_list[attempts % len(task_list)]
            episode = collect_episode(env, policy, task, args.max_steps, args.img_size)
            attempts += 1

            if episode is not None:
                filepath = os.path.join(task_dir, f'episode_{collected:03d}.hdf5')
                save_episode(episode, filepath, language)
                collected += 1
                inner_bar.update(1)

        inner_bar.close()
        env.close()

        status = 'DONE' if collected >= args.num_demos else 'FAIL'
        success_rate = collected / attempts * 100 if attempts > 0 else 0
        tqdm.write(
            f'[{status}] {task_name}: {collected}/{args.num_demos}개 수집 '
            f'(시도: {attempts}, 성공률: {success_rate:.1f}%)'
        )

    print(f"\n완료! 데이터 저장 경로: {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MetaWorld MT50 데이터 수집')
    parser.add_argument('--save_dir',  type=str, default='/home/jun/data/metaworld/max_step_200',
                        help='데이터 저장 경로')
    parser.add_argument('--num_demos', type=int, default=50,
                        help='태스크당 수집할 데모 수')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='에피소드 최대 스텝 수')
    parser.add_argument('--img_size',  type=int, default=224,
                        help='저장할 이미지 크기 (224 권장)')
    parser.add_argument('--seed',      type=int, default=42,
                        help='MetaWorld 랜덤 시드')
    args = parser.parse_args()
    main(args)
