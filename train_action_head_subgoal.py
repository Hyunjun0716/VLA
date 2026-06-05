"""
Action Head Fine-tuning with Subgoal Conditioning.

VLM(LoRA) + SubgoalDiffuserMLP 동결,
ConditionalUnet1DWithSubgoal(action head)만 재학습.

학습 데이터 흐름:
  HDF5(image, qpos, action, lang) + latents(z_future)
    → VLM (frozen) → hidden_states
    → ConditionalUnet1DWithSubgoal(hidden_states, qpos, z_future) → diffusion loss

평가 시(eval_metaworld_subgoal.py):
  VLM → z_t → SubgoalDiffuser.sample() → g_t → ConditionalUnet1DWithSubgoal

Usage:
    python train_action_head_subgoal.py \
        --checkpoint ~/experiments/tinyvla_metaworld_mt50_H/checkpoint-10000 \
        --base_model ~/models/Llava-Pythia-1.3B \
        --subgoal_ckpt ~/experiments/tinyvla_metaworld_mt50_H/subgoal_diffuser/subgoal_diffuser_best.pt \
        --data_dir /home/jun/data/metaworld \
        --latent_dir ~/experiments/tinyvla_metaworld_mt50_H/latents \
        --output_dir ~/experiments/tinyvla_metaworld_mt50_H/action_head_subgoal
"""

import os
import glob
import argparse
import json
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
import cv2

from eval_real_franka import llava_pythia_act_policy
from policy_heads.models.subgoal_diffuser import SubgoalDiffuserMLP, GlobalLatentExtractor
from policy_heads.models.droid_unet_diffusion_subgoal import (
    ConditionalUnet1DWithSubgoal, build_subgoal_unet_from_checkpoint
)
from aloha_scripts.constants import TASK_CONFIGS

IMG_SIZE   = 224
CHUNK_SIZE = 16
ACTION_DIM = 4
STATE_DIM  = 4
SUBGOAL_DIM = 2048
DELTA      = 32


class MetaWorldSubgoalDataset(Dataset):
    """
    (image, qpos, action_chunk, language, z_future) 튜플 반환.

    HDF5 파일에서 이미지/상태/액션/언어를 읽고,
    같은 에피소드의 latent 파일에서 z_future = z_{t+DELTA}를 읽음.
    """

    def __init__(self, data_dir, latent_dir, task_config_name="metaworld_mt50",
                 chunk_size=CHUNK_SIZE, delta=DELTA, img_size=IMG_SIZE):
        self.chunk_size = chunk_size
        self.delta = delta
        self.img_size = img_size

        # 태스크 디렉토리 목록
        cfg = TASK_CONFIGS[task_config_name]
        task_dirs = cfg["dataset_dir"]

        self.samples = []   # (hdf5_path, latent_path, t)
        for task_path in task_dirs:
            task_name = os.path.basename(task_path)
            ep_files = sorted(glob.glob(os.path.join(task_path, "episode_*.hdf5")))
            for ep_file in ep_files:
                ep_name = os.path.splitext(os.path.basename(ep_file))[0]
                lat_file = os.path.join(latent_dir, task_name, f"{ep_name}_latents.pt")
                if not os.path.exists(lat_file):
                    continue
                with h5py.File(ep_file, "r") as f:
                    T = f["/observations/qpos"].shape[0]
                for t in range(T - chunk_size):   # chunk가 끝까지 가능한 t만
                    self.samples.append((ep_file, lat_file, t))

        print(f"MetaWorldSubgoalDataset: {len(self.samples)} samples "
              f"from {len(task_dirs)} tasks")

        # 에피소드 캐시 (latent는 작으므로 캐시)
        self._lat_cache = {}

    def _load_latents(self, lat_file):
        if lat_file not in self._lat_cache:
            self._lat_cache[lat_file] = torch.load(
                lat_file, map_location='cpu', weights_only=True
            )
        return self._lat_cache[lat_file]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_file, lat_file, t = self.samples[idx]

        with h5py.File(ep_file, "r") as f:
            raw_lang = f["language_raw"][0].decode("utf-8")
            img      = f["/observations/images/front"][t]   # (224, 224, 3) uint8
            qpos     = f["/observations/qpos"][t][:4]        # (4,) EEF xyz + gripper
            T        = f["/observations/qpos"].shape[0]
            # action chunk: t ~ t+chunk_size (경계에서 마지막 액션 반복)
            end = min(t + self.chunk_size, T)
            actions = f["/action"][t:end]                   # (chunk, 4)

        # 짧은 경우 패딩
        if len(actions) < self.chunk_size:
            pad = np.tile(actions[-1:], (self.chunk_size - len(actions), 1))
            actions = np.concatenate([actions, pad], axis=0)

        # 이미지 전처리: HWC uint8 → CHW float [0,1]
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        # latent: z_{t+delta}  (teacher forcing)
        latents = self._load_latents(lat_file)
        T_lat   = latents.shape[0]
        z_future = latents[min(t + self.delta, T_lat - 1)]  # (2048,)

        return {
            "image":    img_t,                                           # (3, 224, 224)
            "qpos":     torch.from_numpy(qpos.astype(np.float32)),      # (7,)
            "actions":  torch.from_numpy(actions.astype(np.float32)),   # (16, 4)
            "lang":     raw_lang,
            "z_future": z_future,                                        # (2048,)
        }


def collate_fn(batch):
    """lang는 문자열이므로 별도 처리."""
    images   = torch.stack([b["image"]    for b in batch])
    qpos     = torch.stack([b["qpos"]     for b in batch])
    actions  = torch.stack([b["actions"]  for b in batch])
    z_future = torch.stack([b["z_future"] for b in batch])
    langs    = [b["lang"] for b in batch]
    return images, qpos, actions, langs, z_future


def parse_args():
    parser = argparse.ArgumentParser(description="Action Head Fine-tuning with Subgoal")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="TinyVLA checkpoint (e.g., .../checkpoint-10000)")
    parser.add_argument("--base_model", type=str,
                        default=os.path.expanduser("~/models/Llava-Pythia-1.3B"))
    parser.add_argument("--subgoal_ckpt", type=str, required=True,
                        help="SubgoalDiffuserMLP checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default="/home/jun/data/metaworld")
    parser.add_argument("--latent_dir", type=str,
                        default=os.path.expanduser(
                            "~/experiments/tinyvla_metaworld_mt50_H/latents"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser(
                            "~/experiments/tinyvla_metaworld_mt50_H/action_head_subgoal_delta32"))
    parser.add_argument("--task_config_name", type=str, default="metaworld_mt50")
    parser.add_argument("--delta",        type=int,   default=32,
                        help="Temporal gap for z_future (must match subgoal diffuser training)")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_steps",    type=int,   default=10000)
    parser.add_argument("--eval_every",   type=int,   default=1000)
    parser.add_argument("--save_every",   type=int,   default=2000)
    parser.add_argument("--log_every",    type=int,   default=50)
    parser.add_argument("--val_ratio",    type=float, default=0.05)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--wandb_project",  type=str, default="action-head-subgoal-metaworld")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def compute_diffusion_loss(unet, noise_scheduler, hidden_states, qpos, actions, z_future):
    """
    Diffusion MSE loss.
    actions: (B, T, 4) ground-truth action chunk
    """
    B = actions.shape[0]
    noise      = torch.randn_like(actions)
    timesteps  = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (B,), device=actions.device
    ).long()
    noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

    noise_pred = unet(
        noisy_actions.to(hidden_states.dtype),
        timesteps,
        global_cond=hidden_states,
        states=qpos,
        subgoal=z_future,
    )
    return F.mse_loss(noise_pred.float(), noise.float())


def evaluate(unet, noise_scheduler, val_loader, vlm_policy, device, max_batches=20):
    unet.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for batch_idx, (images, qpos, actions, langs, z_future) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            images   = images.to(device)
            qpos     = qpos.to(device)
            actions  = actions.to(device)
            z_future = z_future.to(device)

            # VLM → hidden_states (frozen)
            img_pair = torch.stack([images, images], dim=1)  # (B,2,3,224,224)
            hidden_states_list = []
            for i in range(images.shape[0]):
                pair_i = img_pair[i]                         # (2,3,224,224)
                state_i = qpos[i:i+1]
                batch_i = vlm_policy.process_batch_to_llava(pair_i, state_i, langs[i])
                vlm_model = vlm_policy.policy
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
                hidden_states_list.append(out[0])
            max_len = max(hs.shape[1] for hs in hidden_states_list)
            padded = [F.pad(hs, (0, 0, 0, max_len - hs.shape[1])) for hs in hidden_states_list]
            hidden_states = torch.cat(padded, dim=0).float()

            loss = compute_diffusion_loss(
                unet, noise_scheduler, hidden_states,
                qpos.float(), actions.float(), z_future.float(),
            )
            total_loss += loss.item() * images.shape[0]
            count      += images.shape[0]
    unet.train()
    return total_loss / count if count > 0 else float('inf')


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda")

    # ── 1. VLM 로드 (전체 동결) ───────────────────────────────
    print("[1/4] Loading VLM (frozen)...")
    policy_config = {
        "model_path":       args.checkpoint,
        "model_base":       args.base_model,
        "enable_lora":      True,
        "conv_mode":        "pythia",
        "action_head":      "droid_diffusion",
        "action_head_type": "droid_diffusion",
        "camera_obs_keys":  ["front_image", "front_image"],
        "config_path":      os.path.dirname(args.checkpoint),
    }
    vlm_policy = llava_pythia_act_policy(policy_config)
    vlm_model  = vlm_policy.policy

    # VLM 전체 동결
    for p in vlm_model.parameters():
        p.requires_grad_(False)
    vlm_model.eval()

    # noise_scheduler 재사용
    noise_scheduler = vlm_model.noise_scheduler

    # ── 2. SubgoalDiffuser 로드 (동결) ───────────────────────
    print("[2/4] Loading SubgoalDiffuser (frozen)...")
    subgoal_diffuser = SubgoalDiffuserMLP(latent_dim=SUBGOAL_DIM).to(device)
    subgoal_diffuser.load_state_dict(
        torch.load(args.subgoal_ckpt, map_location=device, weights_only=True)
    )
    subgoal_diffuser.eval()
    for p in subgoal_diffuser.parameters():
        p.requires_grad_(False)

    # ── 3. ConditionalUnet1DWithSubgoal 구성 ─────────────────
    print("[3/4] Building ConditionalUnet1DWithSubgoal...")
    new_unet = build_subgoal_unet_from_checkpoint(
        vlm_model.embed_out, subgoal_dim=SUBGOAL_DIM
    ).to(device).float()  # VLM은 bfloat16이지만 action head는 float32로 학습
    # SinusoidalPosEmb.dtype 속성도 float32로 갱신 (가중치만 변환해도 이 속성은 그대로 남음)
    for m in new_unet.modules():
        if hasattr(m, 'dtype') and m.dtype == torch.bfloat16:
            m.dtype = torch.float32
    new_unet.train()

    trainable = sum(p.numel() for p in new_unet.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable / 1e6:.2f}M")

    # ── 4. 데이터셋 ──────────────────────────────────────────
    print("[4/4] Loading dataset...")
    full_dataset = MetaWorldSubgoalDataset(
        data_dir=args.data_dir,
        latent_dir=args.latent_dir,
        task_config_name=args.task_config_name,
        delta=args.delta,
    )
    val_size   = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn)
    print(f"  Train: {train_size}, Val: {val_size}")

    # ── Optimizer / Scheduler ────────────────────────────────
    optimizer = AdamW(new_unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)

    # ── Logging ──────────────────────────────────────────────
    log_dir = os.path.join(args.output_dir, "log")
    writer  = SummaryWriter(log_dir=log_dir)
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    print(f"\nTraining action head for {args.max_steps} steps...")
    best_val_loss = float('inf')
    global_step   = 0

    pbar = tqdm(total=args.max_steps, desc="Training")

    while global_step < args.max_steps:
        for images, qpos, actions, langs, z_future in train_loader:
            if global_step >= args.max_steps:
                break

            images   = images.to(device)
            qpos     = qpos.to(device)
            actions  = actions.to(device)
            z_future = z_future.to(device)

            # VLM forward (no grad)
            with torch.no_grad():
                img_pair = torch.stack([images, images], dim=1)  # (B,2,3,224,224)
                hidden_states_list = []
                for i in range(images.shape[0]):
                    pair_i  = img_pair[i]
                    state_i = qpos[i:i+1]
                    batch_i = vlm_policy.process_batch_to_llava(pair_i, state_i, langs[i])
                    # LlavaPythia backbone forward (embed_out 호출 없이 hidden_states만 추출)
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
                    hs = out[0]  # (1, seq_len, hidden_size)
                    hidden_states_list.append(hs)
                # 언어 토큰 길이가 샘플마다 다를 수 있으므로 max seq_len에 맞춰 패딩
                max_len = max(hs.shape[1] for hs in hidden_states_list)
                padded = [F.pad(hs, (0, 0, 0, max_len - hs.shape[1])) for hs in hidden_states_list]
                hidden_states = torch.cat(padded, dim=0)

            # new_unet은 float32이므로 모든 입력을 float32로 통일
            hidden_states = hidden_states.float()
            loss = compute_diffusion_loss(
                new_unet, noise_scheduler, hidden_states,
                qpos.float(), actions.float(), z_future.float(),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(new_unet.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.6f}")

            if global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar("loss/train", loss.item(), global_step)
                writer.add_scalar("lr", lr, global_step)
                wandb.log({"loss/train": loss.item(), "lr": lr}, step=global_step)

            if global_step % args.eval_every == 0:
                val_loss = evaluate(new_unet, noise_scheduler, val_loader, vlm_policy, device)
                writer.add_scalar("loss/val", val_loss, global_step)
                wandb.log({"loss/val": val_loss}, step=global_step)
                print(f"\n  [step {global_step}] val_loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(new_unet.state_dict(),
                               os.path.join(args.output_dir, f"action_head_best_delta{args.delta}.pt"))
                    print(f"  -> New best (val_loss={val_loss:.4f})")

            if global_step % args.save_every == 0:
                torch.save(new_unet.state_dict(),
                           os.path.join(args.output_dir,
                                        f"action_head_step{global_step}_delta{args.delta}.pt"))

    pbar.close()
    torch.save(new_unet.state_dict(),
               os.path.join(args.output_dir, f"action_head_final_delta{args.delta}.pt"))

    writer.flush()
    writer.close()
    wandb.finish()

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
