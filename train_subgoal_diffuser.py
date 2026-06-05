"""
Standalone training script for the Subgoal Diffuser.

Trains a lightweight MLP-based DDIM diffusion model to predict
future latent z_{t+Δ} from current latent z_t.

Input: Pre-extracted latents from extract_latents.py
       (latents/TASK/episode_*_latents.pt)

Usage:
    python train_subgoal_diffuser.py \
        --latent_dir ~/experiments/tinyvla_metaworld_mt50_H/latents \
        --output_dir ~/experiments/tinyvla_metaworld_mt50_H/subgoal_diffuser \
        --delta 16 \
        --max_steps 30000
"""

import os
import glob
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from policy_heads.models.subgoal_diffuser import SubgoalDiffuserMLP


class LatentPairDataset(Dataset):
    """
    Dataset of (z_t, z_{t+Δ}) pairs loaded from pre-extracted latent files.

    Each .pt file contains a tensor of shape (T, D) representing the
    per-frame latent vectors for one episode.
    """

    def __init__(self, latent_dir, delta=16):
        self.delta = delta
        self.latent_files = sorted(glob.glob(
            os.path.join(latent_dir, "**", "*_latents.pt"), recursive=True
        ))

        if len(self.latent_files) == 0:
            raise FileNotFoundError(f"No *_latents.pt files found in {latent_dir}")

        self.index = []
        self.episode_lengths = []
        for file_idx, path in enumerate(self.latent_files):
            T = torch.load(path, map_location='cpu', weights_only=True).shape[0]
            self.episode_lengths.append(T)
            for t in range(T):
                self.index.append((file_idx, t))

        print(f"LatentPairDataset: {len(self.latent_files)} episodes, "
              f"{len(self.index)} pairs, delta={delta}")

        self._cache = {}

    def _load_episode(self, file_idx):
        if file_idx not in self._cache:
            self._cache[file_idx] = torch.load(
                self.latent_files[file_idx], map_location='cpu', weights_only=True
            )
        return self._cache[file_idx]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, t = self.index[idx]
        latents = self._load_episode(file_idx)
        T = latents.shape[0]
        z_t = latents[t]
        z_future = latents[min(t + self.delta, T - 1)]
        return z_t, z_future


def parse_args():
    parser = argparse.ArgumentParser(description="Train Subgoal Diffuser")
    parser.add_argument("--latent_dir", type=str,
                        default=os.path.expanduser("~/experiments/tinyvla_metaworld_mt50_H/latents"),
                        help="Directory with extracted latents")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser("~/experiments/tinyvla_metaworld_mt50_H/subgoal_diffuser_delta32"),
                        help="Where to save trained model")
    parser.add_argument("--delta", type=int, default=32,
                        help="Temporal gap for future latent (default: 16 = chunk_size)")
    parser.add_argument("--latent_dim", type=int, default=2048,
                        help="Latent dimension (must match VLM hidden_size)")
    parser.add_argument("--time_emb_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=30000,
                        help="Total training steps (default: 30000 ≈ 16 epochs)")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Fraction of data for validation")
    parser.add_argument("--eval_every_steps", type=int, default=1000,
                        help="Run validation every N steps")
    parser.add_argument("--save_every_steps", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--log_every_steps", type=int, default=100,
                        help="Log batch loss every N steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="subgoal-diffuser-metaworld",
                        help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name (default: auto)")
    return parser.parse_args()


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for z_t, z_future in val_loader:
            z_t, z_future = z_t.to(device), z_future.to(device)
            loss = model.compute_loss(z_t, z_future)
            total_loss += loss.item() * z_t.shape[0]
            count += z_t.shape[0]
    model.train()
    return total_loss / count if count > 0 else float('inf')


def test_sample_quality(model, val_loader, device):
    model.eval()
    similarities = []
    with torch.no_grad():
        for z_t, z_future in val_loader:
            z_t, z_future = z_t.to(device), z_future.to(device)
            g_t = model.sample(z_t)
            sim = F.cosine_similarity(g_t, z_future, dim=-1)
            similarities.extend(sim.cpu().tolist())
            if len(similarities) >= 1000:
                break
    model.train()
    sims = torch.tensor(similarities)
    return sims.mean().item(), sims.std().item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    print("Loading latent pairs...")
    full_dataset = LatentPairDataset(args.latent_dir, delta=args.delta)

    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Train: {train_size}, Val: {val_size}")

    # Model
    model = SubgoalDiffuserMLP(
        latent_dim=args.latent_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SubgoalDiffuserMLP: {num_params / 1e6:.2f}M parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)

    # Logging
    log_dir = os.path.join(args.output_dir, "log")
    writer = SummaryWriter(log_dir=log_dir)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )

    print(f"\nStarting training for {args.max_steps} steps...")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")

    best_val_loss = float('inf')
    training_log = []
    global_step = 0
    model.train()

    pbar = tqdm(total=args.max_steps, desc="Training")

    while global_step < args.max_steps:
        for z_t, z_future in train_loader:
            if global_step >= args.max_steps:
                break

            z_t, z_future = z_t.to(device), z_future.to(device)
            loss = model.compute_loss(z_t, z_future)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

            # Step 로그
            if global_step % args.log_every_steps == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar("loss/train_step", loss.item(), global_step)
                writer.add_scalar("lr", lr, global_step)
                wandb.log({"loss/train_step": loss.item(), "lr": lr}, step=global_step)

            # 검증
            if global_step % args.eval_every_steps == 0:
                val_loss = evaluate(model, val_loader, device)
                writer.add_scalar("loss/val", val_loss, global_step)
                wandb.log({"loss/val": val_loss}, step=global_step)
                print(f"\n  [step {global_step}] val_loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(),
                               os.path.join(args.output_dir, f"subgoal_diffuser_best_delta{args.delta}.pt"))
                    print(f"  -> New best (val_loss={val_loss:.4f})")

                training_log.append({"step": global_step, "val_loss": val_loss})

            # 체크포인트 저장
            if global_step % args.save_every_steps == 0:
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, f"subgoal_diffuser_step{global_step}_delta{args.delta}.pt"))

                mean_sim, std_sim = test_sample_quality(model, val_loader, device)
                writer.add_scalar("quality/cosine_sim_mean", mean_sim, global_step)
                writer.add_scalar("quality/cosine_sim_std", std_sim, global_step)
                wandb.log({"quality/cosine_sim_mean": mean_sim,
                           "quality/cosine_sim_std": std_sim}, step=global_step)
                print(f"  [step {global_step}] cosine_sim={mean_sim:.4f} +/- {std_sim:.4f}")

    pbar.close()

    # 최종 저장
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, f"subgoal_diffuser_final_delta{args.delta}.pt"))

    mean_sim, std_sim = test_sample_quality(model, val_loader, device)
    writer.add_scalar("quality/cosine_sim_mean", mean_sim, global_step)
    writer.flush()
    writer.close()
    wandb.log({"quality/cosine_sim_mean": mean_sim, "quality/cosine_sim_std": std_sim},
              step=global_step)
    wandb.finish()

    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Final cosine_sim: {mean_sim:.4f} +/- {std_sim:.4f}")
    print(f"  Saved to: {args.output_dir}")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
