# Plan: True End-to-End Subgoal Training

## Context

Teacher Forcing Gap 문제를 근본적으로 해결하기 위한 진정한 E2E 학습.
- **기존 문제**: 사전 추출된 z_future 사용 → VLM 가중치 변경 시 z 불일치, ActionHead가 완벽한 subgoal에만 의존
- **해결**: image[t+delta]를 매 스텝 VLM에 통과시켜 z_future 실시간 계산 + ActionHead에 예측된 g_t 사용 (teacher forcing 제거)

---

## 전체 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                    E2E Training Loop                         │
│                                                             │
│  Dataset                                                    │
│  ┌──────────────┐                                          │
│  │ episode.hdf5 │──→ image[t]      ─────────────────────┐ │
│  │              │──→ image[t+16]   ──────────────────┐  │ │
│  │              │──→ qpos[t]                          │  │ │
│  │              │──→ actions[t:t+16]                  │  │ │
│  │              │──→ lang                             │  │ │
│  └──────────────┘                                     │  │ │
│                                                       │  │ │
│  ┌─────────────────────────────┐                     │  │ │
│  │   VLM (Pythia-1.3B + LoRA) │◄────────────────────┘  │ │
│  │   [LoRA trainable, lr=1e-5] │◄───────────────────────┘ │
│  └──────────────┬──────────────┘                          │
│                 │ hidden_states_t    hidden_states_future  │
│                 │ (B, seq, 2048)     (B, seq, 2048)        │
│                 ▼                         ▼               │
│  ┌──────────────────────┐   ┌──────────────────────┐      │
│  │ GlobalLatentExtractor│   │ GlobalLatentExtractor│      │
│  │  mean_pool + L2_norm │   │  mean_pool + L2_norm │      │
│  └──────────┬───────────┘   └──────────┬───────────┘      │
│             │ z_t (B,2048)             │ z_future (B,2048) │
│             │                          │                   │
│             │         ┌────────────────┘                   │
│             ▼         ▼                                    │
│  ┌──────────────────────────────────┐                      │
│  │      SubgoalDiffuserMLP          │                      │
│  │  compute_loss(z_t, z_future)     │──→ subgoal_loss      │
│  │  [trainable, lr=1e-4]            │                      │
│  └──────────┬───────────────────────┘                      │
│             │ sample(z_t)  [no_grad]                       │
│             │ g_t (B, 2048)                                │
│             ▼                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │   ConditionalUnet1DWithSubgoal               │          │
│  │   combine = Linear(2048+7+2048, 2048)        │          │
│  │   [combine 랜덤초기화, 나머지 embed_out 복사] │          │
│  │   [trainable, lr=1e-4]                       │          │
│  │                                              │          │
│  │   input: noisy_actions, timestep             │          │
│  │   cond:  hidden_states_t + qpos + g_t        │──→ action_loss │
│  └──────────────────────────────────────────────┘          │
│                                                             │
│  total_loss = action_loss + 0.1 × subgoal_loss             │
│  optimizer.step() → VLM LoRA + SubgoalDiffuser + ActionHead │
└─────────────────────────────────────────────────────────────┘
```

---

## 코드

### 1. 데이터셋 (latent 파일 불필요)

```python
class MetaWorldE2EDataset(Dataset):
    """image[t]와 image[t+delta] 모두 반환. latent 파일 불필요."""

    def __init__(self, data_dir, task_config_name="metaworld_mt50",
                 chunk_size=16, delta=16, img_size=224):
        self.delta = delta
        self.chunk_size = chunk_size
        self.samples = []  # (hdf5_path, t)

        cfg = TASK_CONFIGS[task_config_name]
        for task_path in cfg["dataset_dir"]:
            for ep_file in sorted(glob.glob(os.path.join(task_path, "episode_*.hdf5"))):
                with h5py.File(ep_file, "r") as f:
                    T = f["/observations/qpos"].shape[0]
                for t in range(T - chunk_size):
                    if t + delta < T:   # future 프레임 존재 보장
                        self.samples.append((ep_file, t))

    def __getitem__(self, idx):
        ep_file, t = self.samples[idx]
        with h5py.File(ep_file, "r") as f:
            T          = f["/observations/qpos"].shape[0]
            raw_lang   = f["language_raw"][0].decode("utf-8")
            img_t      = f["/observations/images/front"][t]                        # (H,W,3)
            img_future = f["/observations/images/front"][min(t+self.delta, T-1)]   # (H,W,3)
            qpos       = f["/observations/qpos"][t]                                # (7,)
            end        = min(t + self.chunk_size, T)
            actions    = f["/action"][t:end]                                       # (chunk,4)

        if len(actions) < self.chunk_size:
            pad = np.tile(actions[-1:], (self.chunk_size - len(actions), 1))
            actions = np.concatenate([actions, pad], axis=0)

        to_tensor = lambda img: torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1)
        return {
            "image_t":      to_tensor(img_t),      # (3,224,224)
            "image_future": to_tensor(img_future),  # (3,224,224)
            "qpos":    torch.from_numpy(qpos.astype(np.float32)),     # (7,)
            "actions": torch.from_numpy(actions.astype(np.float32)),  # (16,4)
            "lang":    raw_lang,
        }
```

### 2. VLM LoRA 파라미터 활성화

```python
# VLM 로드
vlm_policy = llava_pythia_act_policy(policy_config)
vlm_model  = vlm_policy.policy

# LoRA만 trainable, 나머지 동결
for name, p in vlm_model.named_parameters():
    if 'lora_' in name:
        p.requires_grad_(True)
    else:
        p.requires_grad_(False)
for p in vlm_model.embed_out.parameters():
    p.requires_grad_(False)   # 기존 ActionHead 동결

# 메모리 절약
vlm_model.get_model().gradient_checkpointing_enable()
```

### 3. SubgoalDiffuser & ActionHead 초기화

```python
# SubgoalDiffuser: 기존 ckpt에서 시작 (이미 MetaWorld z 표현을 학습함)
subgoal_model = SubgoalDiffuserMLP(latent_dim=2048).cuda().float()
subgoal_model.load_state_dict(torch.load(args.subgoal_ckpt, map_location='cuda'))
subgoal_model.train()

# GlobalLatentExtractor (파라미터 없음, 고정)
extractor = GlobalLatentExtractor().cuda().float()

# ActionHead: combine 랜덤초기화, 나머지는 embed_out 복사
new_unet = build_subgoal_unet_from_checkpoint(
    vlm_model.embed_out, subgoal_dim=2048
).cuda().float()
for m in new_unet.modules():
    if hasattr(m, 'dtype') and m.dtype == torch.bfloat16:
        m.dtype = torch.float32
new_unet.train()
```

### 4. 옵티마이저 (파라미터 그룹)

```python
optimizer = AdamW([
    {
        'params': [p for n, p in vlm_model.named_parameters()
                   if 'lora_' in n and p.requires_grad],
        'lr': 1e-5,    # VLM LoRA: 낮은 lr (기존 표현 보존)
    },
    {
        'params': subgoal_model.parameters(),
        'lr': 1e-4,    # SubgoalDiffuser
    },
    {
        'params': new_unet.parameters(),
        'lr': 1e-4,    # ActionHead
    },
], weight_decay=1e-4)
```

### 5. 학습 루프 핵심

```python
def get_hidden_states(vlm_model, vlm_policy, images, qpos_batch, langs):
    """(B,3,H,W) 이미지 배치 → hidden_states (B, max_len, 2048) float32"""
    hs_list = []
    for i in range(images.shape[0]):
        pair_i  = torch.stack([images[i], images[i]], dim=0)  # (2,3,H,W)
        state_i = qpos_batch[i:i+1]
        batch_i = vlm_policy.process_batch_to_llava(pair_i, state_i, langs[i])
        input_ids_mm, attn_mask, past_kv, inputs_embeds, _ = \
            vlm_model.prepare_inputs_labels_for_multimodal(
                batch_i["input_ids"], batch_i["attention_mask"],
                None, None, batch_i["images"],
                images_r=batch_i["images_r"],
                visual_concat=vlm_model.visual_concat,
                states=batch_i["states"],
            )
        out = vlm_model.get_model()(
            input_ids=input_ids_mm, attention_mask=attn_mask,
            past_key_values=past_kv, inputs_embeds=inputs_embeds,
            use_cache=False, return_dict=True,
        )
        hs_list.append(out[0])  # (1, seq_len, 2048) bfloat16
    max_len = max(h.shape[1] for h in hs_list)
    padded  = [F.pad(h, (0, 0, 0, max_len - h.shape[1])) for h in hs_list]
    return torch.cat(padded, dim=0).float()  # (B, max_len, 2048) float32

# --- 학습 스텝 ---
images_t      = images_t.to(device)
images_future = images_future.to(device)
qpos          = qpos.to(device)
actions       = actions.to(device)

# VLM forward ×2 (LoRA grad 활성화)
hs_t      = get_hidden_states(vlm_model, vlm_policy, images_t,      qpos, langs)
hs_future = get_hidden_states(vlm_model, vlm_policy, images_future,  qpos, langs)

# latent 추출
z_t      = extractor(hs_t)       # (B, 2048)
z_future = extractor(hs_future)   # (B, 2048)

# SubgoalDiffuser loss
subgoal_loss = subgoal_model.compute_loss(z_t, z_future)

# g_t: teacher forcing 없이 예측값 사용
with torch.no_grad():
    g_t = subgoal_model.sample(z_t)  # (B, 2048)

# ActionHead loss
action_loss = compute_diffusion_loss(
    new_unet, noise_scheduler,
    hs_t, qpos.float(), actions.float(), g_t.float(),
)

total_loss = action_loss + 0.1 * subgoal_loss

# Gradient accumulation
(total_loss / args.grad_accum).backward()
if (global_step + 1) % args.grad_accum == 0:
    torch.nn.utils.clip_grad_norm_(
        [p for g in optimizer.param_groups for p in g['params']], 1.0
    )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### 6. 저장

```python
torch.save(new_unet.state_dict(),
           os.path.join(args.output_dir, "new_unet_best.pt"))
torch.save(subgoal_model.state_dict(),
           os.path.join(args.output_dir, "subgoal_model_best.pt"))
# LoRA 가중치 저장
lora_state = {n: p for n, p in vlm_model.named_parameters() if 'lora_' in n}
torch.save(lora_state, os.path.join(args.output_dir, "lora_weights.pt"))
```

---

## 실행

```bash
python train_e2e_subgoal.py \
    --checkpoint ~/experiments/tinyvla_metaworld_mt50_H/checkpoint-10000 \
    --base_model ~/models/Llava-Pythia-1.3B \
    --subgoal_ckpt ~/experiments/tinyvla_metaworld_mt50_H/subgoal_diffuser/subgoal_diffuser_best.pt \
    --data_dir /home/jun/data/metaworld \
    --output_dir ~/experiments/tinyvla_metaworld_mt50_H/e2e_subgoal \
    --batch_size 1 \
    --grad_accum 32 \
    --max_steps 20000
```

---

## 기존 파일 수정

없음. `train_action_head_subgoal.py`의 `compute_diffusion_loss` import해서 재사용.

---

## 검증

1. `loss/subgoal`과 `loss/action` 모두 감소 확인 (wandb)
2. 완료 후 `eval_metaworld_subgoal.py` 실행
   - `--action_head_ckpt new_unet_best.pt`
   - `--subgoal_ckpt subgoal_model_best.pt`
3. baseline vs 기존 subgoal(teacher forcing) vs E2E subgoal 성능 비교
