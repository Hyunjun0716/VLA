"""
train_tinyvla_e2e.py — True End-to-End Subgoal Training from Base Model

완전 처음부터(base Llava-Pythia-1.3B) E2E 학습:
  - MetaWorld 체크포인트 없음 (--checkpoint 불필요)
  - SubgoalDiffuserMLP 체크포인트 없음 (랜덤 초기화)
  - ConditionalUnet1DWithSubgoal 랜덤 초기화
  - image[t+delta]를 VLM에 통과시켜 z_future 실시간 계산
  - g_t = SubgoalDiffuserMLP.sample(z_t)  (teacher forcing 없음)
  - total_loss = action_loss + 0.1 * subgoal_loss

Usage:
    python train_tinyvla_e2e.py \
        --base_model ~/models/Llava-Pythia-1.3B \
        --output_dir ~/experiments/tinyvla_e2e_mt50 \
        --task_config_name metaworld_mt50 \
        --max_steps 10000 \
        --batch_size 32 --grad_accum 8
"""

import os
import sys
import glob
import json
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import get_model_name_from_path
from llava_pythia.model.language_model.pythia.configuration_llava_pythia import LlavaPythiaConfig
from peft import LoraConfig, get_peft_model

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from policy_heads.models.subgoal_diffuser import SubgoalDiffuserMLP, GlobalLatentExtractor
from policy_heads.models.droid_unet_diffusion_subgoal import ConditionalUnet1DWithSubgoal
from aloha_scripts.constants import TASK_CONFIGS

# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE  = 16
ACTION_DIM  = 4
STATE_DIM   = 4
HIDDEN_SIZE = 2048
DELTA       = 32       # z_future = z_{t+DELTA}
IMG_SIZE    = 224
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: returns image[t] + image[t+delta] (both CLIP-processed)
# ─────────────────────────────────────────────────────────────────────────────
class MetaWorldE2EDataset(Dataset):
    """
    EpisodicDataset보다 단순한 커스텀 데이터셋.
    image[t]와 image[t+delta]를 모두 반환.
    z_future는 학습 스텝에서 VLM으로 실시간 계산.
    """

    def __init__(self, task_config_name="metaworld_mt50",
                 chunk_size=CHUNK_SIZE, delta_min=32, delta_max=32,
                 image_processor=None, tokenizer=None, conv_mode="pythia"):
        from llava_pythia.conversation import conv_templates
        from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava_pythia.mm_utils import tokenizer_image_token

        self.chunk_size = chunk_size
        self.delta_min  = delta_min
        self.delta_max  = delta_max
        self.image_processor = image_processor
        self.tokenizer  = tokenizer
        self.conv_mode  = conv_mode
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._conv_templates = conv_templates
        self._tok_image_token = tokenizer_image_token

        cfg = TASK_CONFIGS[task_config_name]
        self.samples = []  # (hdf5_path, t, T)
        for task_path in cfg["dataset_dir"]:
            for ep_file in sorted(glob.glob(os.path.join(task_path, "episode_*.hdf5"))):
                with h5py.File(ep_file, "r") as f:
                    T = f["/observations/qpos"].shape[0]
                # 최소 delta_min 만큼 미래 프레임이 존재하는 timestep만 포함
                for t in range(T - chunk_size):
                    if t + delta_min < T:
                        self.samples.append((ep_file, t, T))

        print(f"[E2EDataset] {len(self.samples)} samples, delta~U[{delta_min},{delta_max}], "
              f"tasks={len(cfg['dataset_dir'])}")

    def __len__(self):
        return len(self.samples)

    def _process_image(self, img_hwc):
        """HWC uint8 → CLIP-processed (3, H, W) tensor."""
        img = img_hwc.astype(np.float32) / 255.0   # (H, W, 3)
        img_t = self.image_processor.preprocess(
            img[None],  # (1, H, W, 3)
            return_tensors='pt',
            do_normalize=True,
            do_rescale=False,
            do_center_crop=False,
        )['pixel_values'][0]  # (3, H, W)
        return img_t

    def _tokenize(self, raw_lang):
        """Tokenize language instruction → input_ids."""
        from llava_pythia.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        conv = self._conv_templates[self.conv_mode].copy()
        inp  = self._DEFAULT_IMAGE_TOKEN + "\n" + raw_lang
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " <|endoftext|>"
        input_ids = self._tok_image_token(
            prompt, self.tokenizer,
            self._IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).squeeze(0)  # (seq,)
        return input_ids

    def __getitem__(self, idx):
        ep_file, t, T = self.samples[idx]
        # 매 샘플마다 delta를 랜덤하게 샘플링 → 다양한 시간 거리의 subgoal 학습
        delta = np.random.randint(self.delta_min, self.delta_max + 1)
        with h5py.File(ep_file, "r") as f:
            raw_lang   = f["language_raw"][0].decode("utf-8")
            img_t      = f["/observations/images/front"][t]
            img_future = f["/observations/images/front"][min(t + delta, T - 1)]
            qpos       = f["/observations/qpos"][t][:4].astype(np.float32)   # (4,) EEF xyz + gripper
            end        = min(t + self.chunk_size, T)
            actions    = f["/action"][t:end].astype(np.float32)           # (chunk, 4)

        if len(actions) < self.chunk_size:
            pad     = np.tile(actions[-1:], (self.chunk_size - len(actions), 1))
            actions = np.concatenate([actions, pad], axis=0)

        img_t_tensor      = self._process_image(img_t)       # (3, H, W)
        img_future_tensor = self._process_image(img_future)  # (3, H, W)
        input_ids         = self._tokenize(raw_lang)         # (seq,)

        return {
            "input_ids":    input_ids,
            "images":       img_t_tensor,      # (3, H, W) CLIP-normalized
            "images_r":     img_t_tensor,      # same (single camera)
            "image_future": img_future_tensor, # (3, H, W) CLIP-normalized
            "state":        torch.from_numpy(qpos),
            "actions":      torch.from_numpy(actions),
        }


def e2e_collate_fn(batch, pad_token_id=1):
    """Custom collate: pad input_ids + stack tensors."""
    input_ids = [b["input_ids"] for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                 padding_value=pad_token_id)
    return {
        "input_ids":     input_ids,
        "attention_mask": input_ids.ne(pad_token_id),
        "images":        torch.stack([b["images"]       for b in batch]),
        "images_r":      torch.stack([b["images_r"]     for b in batch]),
        "image_future":  torch.stack([b["image_future"] for b in batch]),
        "state":         torch.stack([b["state"]        for b in batch]),
        "actions":       torch.stack([b["actions"]      for b in batch]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VLM hidden states helper
# ─────────────────────────────────────────────────────────────────────────────
def get_hidden_states(vlm, batch, images_key="images", visual_concat="None"):
    """
    Batch-level VLM forward → hidden_states (B, seq_len, hidden) float32.
    vlm: LlavaPythiaForCausalLM (or PeftModel with attribute delegation).
    images are converted to model dtype (bfloat16) before encoding.
    """
    dev   = next(vlm.parameters()).device
    dtype = next(vlm.parameters()).dtype   # typically bfloat16

    # Convert images to model dtype — required by bfloat16 vision tower
    images = batch[images_key].to(dev, dtype=dtype)
    states = batch["state"].to(dev, dtype=dtype)
    # MetaWorld는 카메라 1개(front)만 사용 → images_r=None
    # images_r이 있으면 visual_concat='token_cat'이어야 하는데 base 모델은 지원 안함

    input_ids_mm, attn_mask, past_kv, inputs_embeds, _ = \
        vlm.prepare_inputs_labels_for_multimodal(
            batch["input_ids"].to(dev),
            batch["attention_mask"].to(dev),
            None, None,
            images,
            images_r=None,
            visual_concat=visual_concat,
            states=states,
        )
    out = vlm.get_model()(
        input_ids=input_ids_mm,
        attention_mask=attn_mask,
        past_key_values=past_kv,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return out[0].float()   # (B, seq_len, hidden) float32


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion loss with subgoal
# ─────────────────────────────────────────────────────────────────────────────
def compute_diffusion_loss(unet, noise_scheduler, hidden_states, qpos, actions, g_t):
    B = actions.shape[0]
    noise     = torch.randn_like(actions)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (B,), device=actions.device
    ).long()
    noisy = noise_scheduler.add_noise(actions, noise, timesteps)
    noise_pred = unet(
        noisy.to(hidden_states.dtype),
        timesteps,
        global_cond=hidden_states,
        states=qpos,
        subgoal=g_t,
    )
    return F.mse_loss(noise_pred.float(), noise.float())


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("E2E TinyVLA Subgoal Training (from scratch)")
    p.add_argument("--base_model",    type=str,
                   default=os.path.expanduser("~/models/Llava-Pythia-1.3B"))
    p.add_argument("--output_dir",    type=str,
                   default=os.path.expanduser("~/experiments/tinyvla_e2e_mt50"))
    p.add_argument("--task_config_name", type=str, default="metaworld_mt50")

    # LoRA
    p.add_argument("--lora_r",       type=int,   default=64)
    p.add_argument("--lora_alpha",   type=int,   default=256)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_module",  type=str,   default="vit llm",
                   help="LoRA target: 'vit llm', 'llm', 'vit', etc.")

    # LR
    p.add_argument("--lr_lora",    type=float, default=1e-5)
    p.add_argument("--lr_action",  type=float, default=1e-4)
    p.add_argument("--lr_subgoal", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Training
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--grad_accum",     type=int,   default=8)
    p.add_argument("--max_steps",      type=int,   default=10000)
    p.add_argument("--eval_every",     type=int,   default=1000)
    p.add_argument("--save_every",     type=int,   default=2000)
    p.add_argument("--log_every",      type=int,   default=50)
    p.add_argument("--val_ratio",      type=float, default=0.05)
    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--seed",           type=int,   default=42)

    # E2E specific
    p.add_argument("--delta_min",      type=int,   default=8)
    p.add_argument("--delta_max",      type=int,   default=32)
    p.add_argument("--subgoal_weight", type=float, default=0.1)
    p.add_argument("--conv_mode",      type=str,   default="pythia")

    # Logging
    p.add_argument("--wandb_project",  type=str, default="tinyvla-e2e-subgoal")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb",       action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(vlm, new_unet, subgoal_model, extractor, noise_scheduler,
             val_loader, device, visual_concat="None", max_batches=20):
    new_unet.eval()
    subgoal_model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            hs_t      = get_hidden_states(vlm, batch, "images",       visual_concat)
            hs_future = get_hidden_states(vlm, batch, "image_future", visual_concat)
            z_t      = extractor(hs_t)
            z_future = extractor(hs_future)
            sg_loss  = subgoal_model.compute_loss(z_t, z_future)
            g_t      = subgoal_model.sample(z_t)
            ac_loss  = compute_diffusion_loss(
                new_unet, noise_scheduler,
                hs_t, batch["state"].float(), batch["actions"].float(), g_t.float()
            )
            total_loss += (ac_loss + 0.1 * sg_loss).item() * hs_t.shape[0]
            count += hs_t.shape[0]
    new_unet.train()
    subgoal_model.train()
    return total_loss / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda")
    base_model = os.path.expanduser(args.base_model)

    # ── 1. Load base VLM ────────────────────────────────────────────────────
    print("[1/5] Loading base VLM from:", base_model)
    model_name = get_model_name_from_path(base_model)
    tokenizer, vlm, image_processor, _ = load_pretrained_model(
        model_path=base_model,
        model_base=None,
        model_name=model_name,
        load_8bit=False, load_4bit=False,
    )
    tokenizer.pad_token_id = 1

    # ── 2. Replace embed_out with ConditionalUnet1DWithSubgoal ──────────────
    print("[2/5] Replacing embed_out with ConditionalUnet1DWithSubgoal...")
    vlm.head_type = "droid_diffusion"
    # config.concat이 Python None일 수 있음 (base 모델) → 문자열 "None"으로 변환
    visual_concat = getattr(vlm, "visual_concat", None) or "None"
    vlm.visual_concat = visual_concat
    new_unet = ConditionalUnet1DWithSubgoal(
        input_dim=ACTION_DIM,
        global_cond_dim=HIDDEN_SIZE,
        subgoal_dim=HIDDEN_SIZE,
        state_dim=STATE_DIM,
    )
    vlm.embed_out = new_unet
    vlm.noise_scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon',
    )
    vlm.num_queries = CHUNK_SIZE
    vlm.num_inference_timesteps = 10
    # proj_to_action is Identity for droid_diffusion; ensure it exists
    if not hasattr(vlm, "proj_to_action"):
        vlm.proj_to_action = nn.Identity()

    # ── 3. Apply LoRA ────────────────────────────────────────────────────────
    print("[3/5] Applying LoRA...")
    from llava_pythia.llava_pythia_utils import find_all_linear_names
    lora_targets = find_all_linear_names(vlm, print, args.lora_module)
    print(f"  LoRA target modules ({len(lora_targets)}): {lora_targets[:5]} ...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_targets,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    vlm = get_peft_model(vlm, lora_config)
    vlm.embed_out.requires_grad_(True)  # action head is NOT a LoRA target; enable manually
    if hasattr(vlm, "proj_to_action"):
        vlm.proj_to_action.requires_grad_(True)

    # move to device + cast
    vlm = vlm.to(device, dtype=torch.bfloat16)
    # embed_out stays float32 (diffusion head works in fp32)
    vlm.embed_out.float()
    vlm.embed_out.to(device)
    for m in vlm.embed_out.modules():
        if hasattr(m, 'dtype') and m.dtype == torch.bfloat16:
            m.dtype = torch.float32

    lora_params = sum(p.numel() for n, p in vlm.named_parameters()
                      if 'lora_' in n and p.requires_grad)
    print(f"  VLM LoRA trainable params: {lora_params/1e6:.2f}M")
    print(f"  ActionHead params:          {sum(p.numel() for p in vlm.embed_out.parameters())/1e6:.2f}M")

    # ── 4. SubgoalDiffuserMLP + Extractor (fresh) ────────────────────────────
    print("[4/5] Initializing SubgoalDiffuserMLP and GlobalLatentExtractor...")
    subgoal_model = SubgoalDiffuserMLP(latent_dim=HIDDEN_SIZE).to(device).float()
    extractor     = GlobalLatentExtractor().to(device).float()

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon',
    )

    # ── 5. Dataset & DataLoader ──────────────────────────────────────────────
    print("[5/5] Loading dataset...")
    full_ds = MetaWorldE2EDataset(
        task_config_name=args.task_config_name,
        chunk_size=CHUNK_SIZE,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        image_processor=image_processor,
        tokenizer=tokenizer,
        conv_mode=args.conv_mode,
    )
    val_size   = max(1, int(len(full_ds) * args.val_ratio))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"  train={train_size}, val={val_size}")

    collate_fn = lambda b: e2e_collate_fn(b, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=False,
    )

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    lora_param_list   = [p for n, p in vlm.named_parameters() if 'lora_' in n and p.requires_grad]
    action_param_list = list(vlm.embed_out.parameters())
    optimizer = AdamW([
        {"params": lora_param_list,            "lr": args.lr_lora},
        {"params": action_param_list,          "lr": args.lr_action},
        {"params": subgoal_model.parameters(), "lr": args.lr_subgoal},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-6)

    # ── Logging ──────────────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # ── Training Loop ────────────────────────────────────────────────────────
    # global_step counts optimizer steps (same as HuggingFace Trainer max_steps)
    # iter_step counts raw forward passes (iter_step = global_step * grad_accum)
    print(f"\nStarting E2E training for {args.max_steps} optimizer steps "
          f"(effective batch = {args.batch_size} × {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum})")

    best_val_loss = float('inf')
    global_step   = 0   # optimizer steps
    iter_step     = 0   # raw iterations
    optimizer.zero_grad()

    all_params = (lora_param_list + action_param_list +
                  list(subgoal_model.parameters()))

    pbar      = tqdm(total=args.max_steps, desc="E2E Training")
    train_it  = iter(train_loader)

    # running averages for logging (reset every log_every optimizer steps)
    running_ac = running_sg = 0.0

    while global_step < args.max_steps:
        # fetch next batch
        try:
            batch = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            batch = next(train_it)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # VLM forward ×2
        hs_t      = get_hidden_states(vlm, batch, "images",       visual_concat)
        hs_future = get_hidden_states(vlm, batch, "image_future", visual_concat)

        # Latent extraction
        z_t      = extractor(hs_t)       # (B, 2048)
        z_future = extractor(hs_future)  # (B, 2048)

        # SubgoalDiffuser loss
        subgoal_loss = subgoal_model.compute_loss(z_t, z_future)

        # g_t: no teacher forcing
        with torch.no_grad():
            g_t = subgoal_model.sample(z_t)  # (B, 2048)

        # Action loss
        action_loss = compute_diffusion_loss(
            vlm.embed_out, noise_scheduler,
            hs_t, batch["state"].float(), batch["actions"].float(), g_t.float(),
        )

        total_loss = action_loss + args.subgoal_weight * subgoal_loss
        (total_loss / args.grad_accum).backward()

        running_ac += action_loss.item()
        running_sg += subgoal_loss.item()
        iter_step  += 1

        # optimizer step every grad_accum raw iterations
        if iter_step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            pbar.update(1)
            pbar.set_postfix(
                ac=f"{running_ac / args.grad_accum:.4f}",
                sg=f"{running_sg / args.grad_accum:.4f}",
                step=global_step,
            )

            if global_step % args.log_every == 0:
                log_dict = {
                    "loss/action":  running_ac / args.grad_accum,
                    "loss/subgoal": running_sg / args.grad_accum,
                    "loss/total":   (running_ac + args.subgoal_weight * running_sg) / args.grad_accum,
                    "lr/lora":      optimizer.param_groups[0]['lr'],
                    "lr/action":    optimizer.param_groups[1]['lr'],
                }
                if not args.no_wandb:
                    wandb.log(log_dict, step=global_step)

            running_ac = running_sg = 0.0  # reset running averages

            if global_step % args.eval_every == 0:
                val_loss = evaluate(vlm, vlm.embed_out, subgoal_model, extractor,
                                    noise_scheduler, val_loader, device, visual_concat)
                print(f"\n  [step {global_step}] val_loss={val_loss:.4f}")
                if not args.no_wandb:
                    wandb.log({"loss/val": val_loss}, step=global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(vlm, subgoal_model, args.output_dir, tag="best")
                    print(f"  -> Best checkpoint saved (val={val_loss:.4f})")

            if global_step % args.save_every == 0:
                _save_checkpoint(vlm, subgoal_model, args.output_dir,
                                 tag=f"step{global_step}")

    pbar.close()
    _save_checkpoint(vlm, subgoal_model, args.output_dir, tag="final")
    print(f"\n학습 완료. best_val_loss={best_val_loss:.4f}")
    print(f"저장 경로: {args.output_dir}")
    if not args.no_wandb:
        wandb.finish()


def _save_checkpoint(vlm, subgoal_model, output_dir, tag="best"):
    """Save LoRA weights + ActionHead + SubgoalDiffuserMLP."""
    # LoRA weights
    lora_state = {n: p.cpu() for n, p in vlm.named_parameters() if 'lora_' in n}
    torch.save(lora_state, os.path.join(output_dir, f"lora_weights_{tag}.pt"))
    # ActionHead (ConditionalUnet1DWithSubgoal)
    torch.save(vlm.embed_out.state_dict(),
               os.path.join(output_dir, f"action_head_{tag}.pt"))
    # SubgoalDiffuserMLP
    torch.save(subgoal_model.state_dict(),
               os.path.join(output_dir, f"subgoal_model_{tag}.pt"))


if __name__ == "__main__":
    main()
