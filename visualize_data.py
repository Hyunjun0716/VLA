"""
MetaWorld 수집 데이터 시각화
Usage:
  # 특정 태스크, 특정 에피소드
  python3 visualize_data.py --task assembly-v3 --episode 0

  # 특정 태스크, 전체 에피소드
  python3 visualize_data.py --task assembly-v3 --all

  # 전체 태스크, 에피소드 0번만
  python3 visualize_data.py --all_tasks
"""

import argparse
import os
import h5py
import numpy as np
import cv2

DATA_DIR = '/home/jun/data/metaworld/max_steps_500_imgsize_224'
OUTPUT_DIR = '/home/jun/data/max_steps_500/videos'


def save_video(frames, save_path, fps=30):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f'저장: {save_path}')


def visualize_episode(task, ep_idx, output_dir):
    ep_path = os.path.join(DATA_DIR, task, f'episode_{ep_idx:03d}.hdf5')
    if not os.path.exists(ep_path):
        print(f'파일 없음: {ep_path}')
        return

    with h5py.File(ep_path, 'r') as f:
        frames = f['observations/images/front'][:]  # (T, H, W, 3)
        lang = f['language_raw'][0].decode() if f['language_raw'].dtype.kind == 'S' else str(f['language_raw'][0])

    print(f'{task} ep{ep_idx}: {len(frames)} steps | "{lang}"')

    save_path = os.path.join(output_dir, task, f'ep{ep_idx:03d}.mp4')
    save_video(frames, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--all', action='store_true', help='해당 태스크의 전체 에피소드')
    parser.add_argument('--all_tasks', action='store_true', help='전체 태스크 ep0번만')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.all_tasks:
        tasks = sorted(os.listdir(DATA_DIR))
        tasks = [t for t in tasks if os.path.isdir(os.path.join(DATA_DIR, t)) and t != 'videos']
        for task in tasks:
            visualize_episode(task, 0, args.output_dir)

    elif args.task and args.all:
        task_dir = os.path.join(DATA_DIR, args.task)
        eps = sorted([f for f in os.listdir(task_dir) if f.endswith('.hdf5')])
        for i, _ in enumerate(eps):
            visualize_episode(args.task, i, args.output_dir)

    elif args.task:
        visualize_episode(args.task, args.episode, args.output_dir)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
