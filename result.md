# TinyVLA + SubgoalDiffuser: 4-Stage Pipeline

## 전체 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FULL PIPELINE OVERVIEW                           │
│                                                                         │
│  MetaWorld MT50 Dataset (HDF5)                                          │
│  image(224×224), qpos(4), action(4), language                           │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────┐ │
│  │   Stage 1   │──▶│   Stage 2   │──▶│   Stage 3   │──▶│  Stage 4   │ │
│  │ TinyVLA     │   │ Extract z   │   │ Subgoal     │   │ ActionHead │ │
│  │ Training    │   │ Latents     │   │ Diffuser    │   │ Training   │ │
│  │             │   │             │   │ Training    │   │            │ │
│  │train_tinyvla│   │extract_     │   │train_subgoal│   │train_action│ │
│  │   .py       │   │latents.py   │   │_diffuser.py │   │_head_      │ │
│  │             │   │             │   │             │   │subgoal.py  │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └────────────┘ │
│         │                 │                 │                  │        │
│    checkpoint          latents/         subgoal_           action_     │
│    -10000              *.pt             diffuser_          head_       │
│                                         best.pt            best.pt     │
│                                                                         │
│                              EVALUATION                                 │
│                    eval_metaworld_subgoal.py                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: TinyVLA 학습

### 목적
Base VLM(Llava-Pythia-1.3B)을 MetaWorld MT50 로봇 조작 태스크에 맞게 fine-tuning.
Action head(ConditionalUnet1D)가 diffusion policy로 action chunk를 생성하도록 학습.

### 코드
- **스크립트**: `scripts/train_metaworld.sh`
- **학습 코드**: `train_tinyvla.py`
- **데이터 로더**: `data_utils/datasets.py`

### 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           STAGE 1: TinyVLA Training                     │
│                                                                         │
│  Input (per timestep t):                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  image[t]    │  │  qpos[t]     │  │  language    │                  │
│  │ (224×224×3)  │  │    (4,)      │  │  instruction │                  │
│  │ front camera │  │EEF xyz+grip  │  │  raw text    │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                           │
│         ▼                 │                 ▼                           │
│  ┌──────────────┐         │        ┌──────────────┐                    │
│  │ CLIP Vision  │         │        │  Tokenizer   │                    │
│  │    Tower     │         │        │  + Embed     │                    │
│  │ (ViT-L/14)  │         │        └──────┬───────┘                    │
│  └──────┬───────┘         │               │                            │
│         │                 │               │                            │
│         └────────────┬────┘               │                            │
│                      ▼                    ▼                            │
│              ┌───────────────────────────────────┐                     │
│              │   prepare_inputs_labels_for_       │                     │
│              │       multimodal()                 │                     │
│              │  [img_tokens + lang_tokens +        │                     │
│              │   state_tokens concatenated]        │                     │
│              └──────────────┬────────────────────┘                     │
│                             │                                           │
│                             ▼                                           │
│              ┌───────────────────────────────────┐                     │
│              │      GPT-NeoX Backbone            │                     │
│              │     (Pythia-1.3B + LoRA)          │                     │
│              │                                   │                     │
│              │  LoRA: r=64, α=256               │                     │
│              │  Target: vit + llm linear layers  │                     │
│              └──────────────┬────────────────────┘                     │
│                             │                                           │
│                             ▼                                           │
│              ┌───────────────────────────────────┐                     │
│              │    hidden_states                  │                     │
│              │    (B, seq_len, 2048)             │                     │
│              └──────────────┬────────────────────┘                     │
│                             │                                           │
│                             ▼                                           │
│              ┌───────────────────────────────────┐                     │
│              │    ConditionalUnet1D               │                     │
│              │    (embed_out / ActionHead)        │                     │
│              │                                   │                     │
│              │  combine = Linear(2048+4, 2048)   │                     │
│              │  global_cond = pool(hidden_states) │                     │
│              │  cond = combine([cond, qpos])     │                     │
│              └──────────────┬────────────────────┘                     │
│                             │                                           │
│                             ▼                                           │
│              ┌───────────────────────────────────┐                     │
│              │    Diffusion Loss                  │                     │
│              │    MSE(noise_pred, noise)          │                     │
│              │    action chunk (16, 4)            │                     │
│              └───────────────────────────────────┘                     │
│                                                                         │
│  설정: batch=32, grad_accum=8 → effective_batch=256                     │
│        max_steps=10000, lr=2e-4 (LoRA), lr=2e-5 (non-LoRA)             │
│        deepspeed ZeRO-2                                                 │
│                                                                         │
│  출력: ~/experiments/tinyvla_metaworld_mt50_H/checkpoint-10000          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 2: VLM에서 z Latent 추출

### 목적
Stage 1에서 학습된 VLM으로 각 에피소드의 모든 프레임을 처리하여
global latent vector z_t를 추출. Stage 3 학습의 입력 데이터로 사용.

### 코드
- **스크립트**: `extract_latents.py`
- **핵심 모듈**: `GlobalLatentExtractor` (`policy_heads/models/subgoal_diffuser.py`)

### 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: Latent Extraction                           │
│                                                                         │
│  For each episode, for each frame t:                                    │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐                                    │
│  │  image[t]    │  │  language    │                                    │
│  │ (224×224×3)  │  │  instruction │                                    │
│  └──────┬───────┘  └──────┬───────┘                                    │
│         │                 │                                             │
│         ▼                 ▼                                             │
│  ┌───────────────────────────────────┐                                 │
│  │   TinyVLA VLM (frozen)           │                                 │
│  │   Stage 1 checkpoint             │                                 │
│  └──────────────┬────────────────────┘                                 │
│                 │                                                       │
│                 ▼                                                       │
│  ┌───────────────────────────────────┐                                 │
│  │    hidden_states                  │                                 │
│  │    (1, seq_len, 2048)            │                                 │
│  └──────────────┬────────────────────┘                                 │
│                 │                                                       │
│                 ▼                                                       │
│  ┌───────────────────────────────────┐                                 │
│  │   GlobalLatentExtractor           │                                 │
│  │                                   │                                 │
│  │   1) mean_pool(hidden_states)     │                                 │
│  │      (seq_len 차원 평균)           │                                 │
│  │      → (1, 2048)                  │                                 │
│  │                                   │                                 │
│  │   2) L2_normalize(z)              │                                 │
│  │      → z_t ∈ unit sphere (2048)  │                                 │
│  └──────────────┬────────────────────┘                                 │
│                 │                                                       │
│                 ▼                                                       │
│  ┌───────────────────────────────────┐                                 │
│  │    episode_*_latents.pt           │                                 │
│  │    shape: (T, 2048)               │                                 │
│  │    T = 에피소드 총 프레임 수        │                                 │
│  └───────────────────────────────────┘                                 │
│                                                                         │
│  출력: ~/experiments/tinyvla_metaworld_mt50_H/latents/                  │
│        TASK_NAME/episode_*_latents.pt                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 3: SubgoalDiffuser 학습

### 목적
현재 latent z_t로부터 delta 스텝 후의 미래 latent z_{t+delta}를
예측하는 diffusion 모델 학습. 평가 시 g_t = sample(z_t) 로 사용.

### 코드
- **학습 코드**: `train_subgoal_diffuser.py`
- **모델**: `SubgoalDiffuserMLP` (`policy_heads/models/subgoal_diffuser.py`)

### 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  STAGE 3: SubgoalDiffuser Training                      │
│                                                                         │
│  학습 데이터 (latent pairs):                                             │
│  ┌──────────────────┐      ┌───────────────────┐                       │
│  │   z_t  (2048)    │      │ z_{t+Δ}  (2048)  │  Δ=32 (default)      │
│  │ 현재 latent      │      │ 미래 latent (GT)  │                       │
│  └────────┬─────────┘      └─────────┬─────────┘                       │
│           │                          │                                  │
│           │        학습 시            │                                  │
│           │   ┌───────────────────┐  │                                  │
│           │   │  DDIM Forward     │  │                                  │
│           │   │  (add noise)      │  │                                  │
│           │   │  x_k = noise(z_Δ,│  │                                  │
│           │   │         timestep k)│  │                                  │
│           │   └────────┬──────────┘  │                                  │
│           │            │             │                                  │
│           ▼            ▼             │                                  │
│  ┌────────────────────────────────┐  │                                  │
│  │     SubgoalDiffuserMLP         │  │                                  │
│  │                                │  │                                  │
│  │  Input: [x_k(2048), t_emb(256),│  │                                  │
│  │          z_t(2048)]            │  │                                  │
│  │          concat → (4352)       │  │                                  │
│  │                                │  │                                  │
│  │  FC: 4352 → 4096  (hidden_dim) │  │                                  │
│  │  SiLU                          │  │                                  │
│  │  FC: 4096 → 4096               │  │                                  │
│  │  SiLU                          │  │                                  │
│  │  FC: 4096 → 2048               │  │                                  │
│  │                                │  │                                  │
│  │  output: noise_pred (2048)     │  │                                  │
│  └───────────────┬────────────────┘  │                                  │
│                  │                   │                                  │
│                  ▼                   ▼                                  │
│         ┌────────────────────────────────┐                             │
│         │   Loss = MSE(noise_pred, noise) │                             │
│         └────────────────────────────────┘                             │
│                                                                         │
│  추론 시 (eval):                                                         │
│  z_t ──▶ DDIM 역방향 10스텝 ──▶ g_t (L2-normalized, 2048)              │
│                                                                         │
│  설정: batch=256, max_steps=30000, lr=1e-4                              │
│        delta=32 (32스텝 후 상태 예측)                                    │
│                                                                         │
│  출력: subgoal_diffuser_best_delta32.pt                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 4: ActionHead 학습 (Subgoal Conditioning)

### 목적
Stage 1의 VLM(frozen) + Stage 3의 SubgoalDiffuser(frozen)를 사용하여
Subgoal 조건부 ActionHead(ConditionalUnet1DWithSubgoal)만 재학습.
Teacher forcing: 학습 시 ground truth z_{t+Δ}를 subgoal로 사용.

### 코드
- **학습 코드**: `train_action_head_subgoal.py`
- **모델**: `ConditionalUnet1DWithSubgoal` (`policy_heads/models/droid_unet_diffusion_subgoal.py`)

### 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│               STAGE 4: ActionHead Training (with Subgoal)               │
│                                                                         │
│  Input:                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ image[t] │  │ qpos[t]  │  │  language    │  │ z_{t+Δ} (2048) │   │
│  │(224×224) │  │  (4,)    │  │              │  │ from latent file │   │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └────────┬─────────┘   │
│       │             │               │                    │             │
│       ▼             │               ▼                    │             │
│  ┌──────────────────────────────────────┐                │             │
│  │      TinyVLA VLM (FROZEN)            │                │             │
│  │      Stage 1 checkpoint              │                │             │
│  └──────────────────┬───────────────────┘                │             │
│                     │                                     │             │
│                     ▼                                     │             │
│        hidden_states (B, seq_len, 2048)                   │             │
│                     │                                     │             │
│                     ▼                                     ▼             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │            ConditionalUnet1DWithSubgoal                          │  │
│  │                                                                  │  │
│  │  1) global_cond = mean_pool(hidden_states)  → (B, 2048)         │  │
│  │                                                                  │  │
│  │  2) combine = Linear(2048 + 4 + 2048, 2048)                     │  │
│  │     cond = combine([global_cond, qpos, z_{t+Δ}])                │  │
│  │          =  combine([   2048   +  4  +  2048 ])                  │  │
│  │          → (B, 2048)                                             │  │
│  │                                                                  │  │
│  │  3) UNet1D denoising (ConditionalResidualBlock1D ×N)            │  │
│  │     noisy_actions → noise_pred                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                     │                                                   │
│                     ▼                                                   │
│         Loss = MSE(noise_pred, noise)                                   │
│         target: action_chunk[t:t+16] (16, 4)                           │
│                                                                         │
│  주의 - Teacher Forcing Gap:                                             │
│  학습: z_{t+Δ} = ground truth latent (완벽한 미래 정보)                  │
│  평가: z_{t+Δ} ≈ SubgoalDiffuser.sample(z_t) (예측값, 불완전)           │
│  → 분포 불일치로 성능 저하 가능                                           │
│                                                                         │
│  설정: batch=32, max_steps=10000, lr=1e-4                               │
│        VLM frozen, SubgoalDiffuser frozen, ActionHead만 학습             │
│                                                                         │
│  출력: action_head_best_delta32.pt                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 평가 파이프라인

### 코드
- `eval_metaworld_subgoal.py` (subgoal 버전)
- `eval_metaworld.py` (베이스라인)
- `scripts/run_eval_all.sh` (두 평가 모두 실행)

### 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION PIPELINE                             │
│                                                                         │
│  MetaWorld Environment (실시간)                                          │
│         │                                                               │
│         ▼ obs at timestep t                                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  image[t] (224×224×3)                                            │  │
│  │  obs[:4]  → qpos (EEF xyz + gripper)   ← 순수 proprioception     │  │
│  └──────────────────┬─────────────────────────────────────────────┘  │
│                     │                                                   │
│                     ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                TinyVLA VLM (frozen)                               │  │
│  └──────────────────┬─────────────────────────────────────────────┘  │
│                     │                                                   │
│                     ▼                                                   │
│        hidden_states (1, seq_len, 2048)                                 │
│                     │                                                   │
│           ┌─────────┴──────────┐                                       │
│           ▼                    ▼                                        │
│  ┌─────────────────┐  ┌─────────────────────────┐                      │
│  │GlobalLatent     │  │  ActionHead는            │                      │
│  │Extractor        │  │  hidden_states를 직접    │                      │
│  │mean_pool+L2norm │  │  사용                    │                      │
│  └────────┬────────┘  └─────────────────────────┘                      │
│           │                    │                                        │
│           ▼                    │                                        │
│      z_t (2048)                │                                        │
│           │                    │                                        │
│           ▼                    │                                        │
│  ┌─────────────────┐           │                                        │
│  │SubgoalDiffuser  │           │                                        │
│  │.sample(z_t)     │           │                                        │
│  │ 10 DDIM steps   │           │                                        │
│  └────────┬────────┘           │                                        │
│           │                    │                                        │
│           ▼                    │                                        │
│       g_t (2048)               │                                        │
│           │                    │                                        │
│           └──────────┬─────────┘                                        │
│                      ▼                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │   ConditionalUnet1DWithSubgoal                                    │  │
│  │   combine([hidden_states_pool, qpos, g_t])                       │  │
│  │   → DDIM denoising 10 steps                                      │  │
│  └──────────────────┬─────────────────────────────────────────────┘  │
│                     │                                                   │
│                     ▼                                                   │
│        action_chunk (16, 4)  → 실행                                     │
│        Temporal aggregation (exp weighted)                              │
│                     │                                                   │
│                     ▼                                                   │
│        env.step(action)  → 다음 obs                                     │
│                                                                         │
│  평가 지표:                                                              │
│  - 50개 태스크 × 10 rollouts × 5 seeds                                  │
│  - success rate per task                                                │
│  - 난이도별 분류: easy(28) / medium(11) / hard(6) / very_hard(5)        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 전체 파이프라인 상세 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE PIPELINE (TRAINING + EVAL)                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MetaWorld MT50 HDF5 Dataset                                        │   │
│  │  50 tasks × 50 demos × ~500 frames                                  │   │
│  │  keys: image(224×224×3), qpos(7→use[:4]), action(4), language       │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│              ┌────────────────────┼───────────────────┐                    │
│              │                    │                   │                    │
│              ▼                    ▼                   ▼                    │
│   ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐        │
│   │   STAGE 1       │  │                  │  │                  │        │
│   │  train_tinyvla  │  │  (Stage 2는 Stage│  │                  │        │
│   │      .py        │  │   1 이후 실행)   │  │                  │        │
│   │                 │  │                  │  │                  │        │
│   │ Base VLM        │  │                  │  │                  │        │
│   │ Llava-Pythia-   │  │                  │  │                  │        │
│   │    1.3B         │  │                  │  │                  │        │
│   │  + LoRA fine-   │  │                  │  │                  │        │
│   │    tuning       │  │                  │  │                  │        │
│   │  + ActionHead   │  │                  │  │                  │        │
│   │   (Unet1D)      │  │                  │  │                  │        │
│   └────────┬────────┘  └──────────────────┘  └──────────────────┘        │
│            │                                                               │
│            ▼                                                               │
│   checkpoint-10000                                                         │
│   (LoRA weights + ActionHead)                                              │
│            │                                                               │
│            ├──────────────────────────────────────────────────────────────│
│            │                                                               │
│            ▼                                                               │
│   ┌─────────────────┐                                                     │
│   │   STAGE 2       │                                                     │
│   │ extract_latents │                                                     │
│   │      .py        │                                                     │
│   │                 │                                                     │
│   │ VLM(frozen) →   │                                                     │
│   │ mean_pool →     │                                                     │
│   │ L2_norm →       │                                                     │
│   │ z_t (2048)      │                                                     │
│   └────────┬────────┘                                                     │
│            │                                                               │
│            ▼                                                               │
│   latents/*.pt  (T, 2048) per episode                                     │
│            │                                                               │
│            ▼                                                               │
│   ┌─────────────────┐                                                     │
│   │   STAGE 3       │                                                     │
│   │ train_subgoal_  │                                                     │
│   │  diffuser.py    │                                                     │
│   │                 │                                                     │
│   │ (z_t, z_{t+32})│                                                     │
│   │ pairs →         │                                                     │
│   │ DDIM diffusion  │                                                     │
│   │ MLP 학습        │                                                     │
│   │ hidden=4096     │                                                     │
│   └────────┬────────┘                                                     │
│            │                                                               │
│            ▼                                                               │
│   subgoal_diffuser_best_delta32.pt                                        │
│            │                                                               │
│            ▼                                                               │
│   ┌─────────────────────────────────────────────┐                        │
│   │              STAGE 4                        │                        │
│   │      train_action_head_subgoal.py            │                        │
│   │                                             │                        │
│   │  VLM(frozen) + SubgoalDiffuser(frozen)      │                        │
│   │                                             │                        │
│   │  hidden_states(2048)                        │                        │
│   │     + qpos(4)                               │                        │
│   │     + z_{t+32}(2048) ← teacher forcing GT  │                        │
│   │  ──▶ combine Linear(4100→2048)              │                        │
│   │  ──▶ UNet1D denoising                       │                        │
│   │  ──▶ action_chunk (16,4)                    │                        │
│   └────────┬────────────────────────────────────┘                        │
│            │                                                               │
│            ▼                                                               │
│   action_head_best_delta32.pt                                             │
│            │                                                               │
│            ▼                                                               │
│   ┌─────────────────────────────────────────────┐                        │
│   │              EVALUATION                      │                        │
│   │     eval_metaworld_subgoal.py                │                        │
│   │                                             │                        │
│   │  [VLM] → hidden_states                      │                        │
│   │       → z_t (GlobalLatentExtractor)          │                        │
│   │       → g_t (SubgoalDiffuser.sample)         │                        │
│   │       → actions (ActionHead + g_t)           │                        │
│   │  obs[:4] (pure proprioception, no obj pos)  │                        │
│   └─────────────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 핵심 설계 결정 사항

| 항목 | 결정 | 이유 |
|------|------|------|
| `qpos = obs[:4]` | EEF xyz + gripper만 사용 | `obs[4:7]`은 goal/obj xyz (privileged state) |
| `state_dim = 4` | 4차원 | 순수 proprioception |
| `delta = 32` | 32스텝 후 예측 | chunk_size(16)보다 2배 앞, 실질적 방향 제공 |
| `hidden_dim = 4096` | SubgoalDiffuserMLP 내부 | 입출력(2048)의 2배, 충분한 표현력 |
| `combine = Linear(2048+4+2048, 2048)` | ActionHead 조건 | hidden + state + subgoal 통합 |
| Teacher Forcing | Stage 4에서 GT z_future 사용 | 학습 안정성, but eval시 gap 존재 |

## 알려진 한계

1. **Teacher Forcing Gap**: Stage 4 학습 시 완벽한 z_future 사용 → 평가 시 예측값(g_t) 사용 → 분포 불일치
2. **Sequential Pipeline**: 각 Stage가 이전 Stage 완료 후 실행 → 학습 시간 긺
3. **Subgoal 타이밍 미지정**: SubgoalDiffuser가 delta 정보 없이 학습 → 평가 시 특정 delta 지정 불가

## 대안: E2E 학습 (train_tinyvla_e2e.py)

Teacher Forcing Gap을 해소하기 위한 완전 통합 학습:
- Stage 1~4를 동시에 학습
- z_future를 VLM으로 실시간 계산 (teacher forcing 없음)
- g_t = SubgoalDiffuser.sample(z_t) 를 학습 중에도 사용
- total_loss = action_loss + 0.1 × subgoal_loss
