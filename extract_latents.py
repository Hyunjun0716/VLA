"""
Offline latent extraction for Subgoal Diffuser training.

Loads a trained TinyVLA checkpoint (VLM + LoRA merged) and extracts
z = L2_norm(mean_pool(hidden_states)) for every frame in every episode.

Output:
    output_dir/TASK_NAME/episode_N_latents.pt  ->  (T, 2048) float32

Usage:
    python extract_latents.py \
        --model_path ~/experiments/tinyvla_metaworld_mt50_H/checkpoint-10000 \
        --model_base ~/models/Llava-Pythia-1.3B \
        --data_dir /home/jun/data/metaworld \
        --output_dir ~/experiments/tinyvla_metaworld_mt50_H/latents
"""

import os
import argparse
import glob
import h5py
import torch
import cv2
from tqdm import tqdm

# TinyVLA imports
from eval_real_franka import llava_pythia_act_policy
from policy_heads.models.subgoal_diffuser import GlobalLatentExtractor
from aloha_scripts.constants import TASK_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VLM latents from MetaWorld MT50 episodes")
    parser.add_argument("--model_path", type=str,
                        default=os.path.expanduser("~/experiments/tinyvla_metaworld_mt50_H/checkpoint-10000"),
                        help="Path to trained checkpoint")
    parser.add_argument("--model_base", type=str,
                        default=os.path.expanduser("~/models/Llava-Pythia-1.3B"),
                        help="Path to base model for LoRA merge")
    parser.add_argument("--data_dir", type=str,
                        default="/home/jun/data/metaworld",
                        help="Root directory of MetaWorld HDF5 data")
    parser.add_argument("--task_config_name", type=str, default="metaworld_mt50",
                        help="Use task list from TASK_CONFIGS. "
                             "Set empty string to scan --data_dir directly.")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser("~/experiments/tinyvla_metaworld_mt50_H/latents"),
                        help="Where to save extracted latents")
    parser.add_argument("--im_size", type=int, default=224,
                        help="Image resize dimension (224 for MetaWorld)")
    return parser.parse_args()


def validate_model_files(model_path):
    """Fail early with a clear message if LoRA companion files are missing."""
    if "checkpoint" in os.path.basename(model_path):
        model_root = os.path.dirname(model_path)
    else:
        model_root = model_path

    required = [
        os.path.join(model_path, "adapter_config.json"),
        os.path.join(model_path, "adapter_model.safetensors"),
        os.path.join(model_path, "preprocessor_config.json"),
        os.path.join(model_root, "config.json"),
        os.path.join(model_root, "non_lora_trainables.bin"),
    ]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        miss = "\n  - " + "\n  - ".join(missing)
        raise FileNotFoundError(
            "Missing model companion files for LoRA merge:\n"
            f"{miss}\n"
            "Make sure checkpoint and model root files are from the same run."
        )


def load_vlm(model_path, model_base):
    """Load VLM with LoRA merged, ready for inference."""
    if "checkpoint" in os.path.basename(model_path):
        model_root = os.path.dirname(model_path)
    else:
        model_root = model_path

    policy_config = {
        "model_path": model_path,
        "model_base": model_base,
        "enable_lora": True,
        "conv_mode": "pythia",
        "action_head": "droid_diffusion",
        "action_head_type": "droid_diffusion",
        "camera_obs_keys": ["front_image", "front_image"],   # MetaWorld: 단일 카메라 복제
        "config_path": model_root,
    }

    policy = llava_pythia_act_policy(policy_config)
    policy.policy.eval()
    return policy


def prepare_image(front_img, im_size):
    """
    MetaWorld 단일 카메라 이미지를 VLM 입력 형식으로 변환.
    front 이미지를 복제해 (1, 2, 3, H, W) 텐서로 반환.

    Args:
        front_img: (H, W, 3) uint8 numpy array
        im_size: target image size (square)

    Returns:
        curr_image: (1, 2, 3, im_size, im_size) float tensor on CUDA
    """
    if front_img.shape[0] != im_size or front_img.shape[1] != im_size:
        front_img = cv2.resize(front_img, (im_size, im_size))

    # HWC -> CHW, normalize to [0, 1]
    img_tensor = torch.from_numpy(front_img.transpose(2, 0, 1) / 255.0).float()

    # 단일 카메라 → 2채널 복제: (2, 3, H, W)
    curr_image = torch.stack([img_tensor, img_tensor], dim=0).cuda()

    # Add batch dim: (1, 2, 3, H, W)
    return curr_image.unsqueeze(0)


def extract_hidden_states(policy, curr_image, robot_state, raw_lang):
    """
    Run VLM forward pass and return hidden_states (before action head).

    Returns:
        hidden_states: (1, seq_len, 2048)
    """
    batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)

    model = policy.policy
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    images = batch["images"]
    images_r = batch["images_r"]
    states = batch["states"]

    # Prepare multimodal inputs
    input_ids_mm, attention_mask_mm, past_key_values, inputs_embeds, _ = \
        model.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, None, None, images,
            images_r=images_r, visual_concat=model.visual_concat, states=states
        )

    # Run backbone LLM
    outputs = model.get_model()(
        input_ids=input_ids_mm,
        attention_mask=attention_mask_mm,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    return outputs[0]  # hidden_states: (1, seq_len, 2048)


def main():
    args = parse_args()

    print("=" * 60)
    print("  Latent Extraction for Subgoal Diffuser (MetaWorld MT50)")
    print(f"  Model:  {args.model_path}")
    print(f"  Data:   {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    # Load model
    print("\n[1/2] Loading VLM...")
    validate_model_files(args.model_path)
    policy = load_vlm(args.model_path, args.model_base)
    extractor = GlobalLatentExtractor()

    # 로봇 상태: 정규화 0 더미 (시각+언어 잠재 벡터만 필요), state_dim=7
    dummy_state = torch.zeros(1, 7).cuda().to(dtype=policy.policy.dtype)

    # Build task list
    task_items = []
    if args.task_config_name:
        if args.task_config_name not in TASK_CONFIGS:
            raise KeyError(f"task_config_name '{args.task_config_name}' not found in TASK_CONFIGS")
        cfg_task_dirs = TASK_CONFIGS[args.task_config_name]["dataset_dir"]
        for task_path in cfg_task_dirs:
            if os.path.isdir(task_path):
                task_items.append((os.path.basename(task_path), task_path))
            else:
                print(f"[WARN] missing task directory, skip: {task_path}")
        print(f"\n[2/2] Extracting latents from {len(task_items)} tasks "
              f"(TASK_CONFIGS['{args.task_config_name}'])...")
    else:
        task_dirs = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
        ])
        task_items = [(d, os.path.join(args.data_dir, d)) for d in task_dirs]
        print(f"\n[2/2] Extracting latents from {len(task_items)} tasks (--data_dir scan)...")

    total_episodes = 0
    total_frames = 0

    for task_name, task_path in tqdm(task_items, desc="Tasks"):
        episode_files = sorted(glob.glob(os.path.join(task_path, "episode_*.hdf5")))

        if not episode_files:
            continue

        out_task_dir = os.path.join(args.output_dir, task_name)
        os.makedirs(out_task_dir, exist_ok=True)

        for ep_file in tqdm(episode_files, desc=f"  {task_name[:40]}", leave=False):
            ep_name = os.path.splitext(os.path.basename(ep_file))[0]
            out_path = os.path.join(out_task_dir, f"{ep_name}_latents.pt")

            if os.path.exists(out_path):
                continue

            with h5py.File(ep_file, "r") as root:
                raw_lang   = root["language_raw"][0].decode("utf-8")
                t_steps    = root["/observations/qpos"].shape[0]
                front_imgs = root["/observations/images/front"][()]  # (T, 224, 224, 3)

            # Extract z for each frame
            latents = []
            with torch.no_grad():
                for t in range(t_steps):
                    curr_image    = prepare_image(front_imgs[t], args.im_size)
                    hidden_states = extract_hidden_states(policy, curr_image, dummy_state, raw_lang)
                    z = extractor(hidden_states)  # (1, 2048)
                    latents.append(z.cpu().float())

            # Stack and save: (T, 2048)
            latents = torch.cat(latents, dim=0)
            torch.save(latents, out_path)

            total_episodes += 1
            total_frames += t_steps

    print(f"\nDone! Extracted {total_episodes} episodes, {total_frames} frames")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
