"""
ConditionalUnet1DWithSubgoal

ConditionalUnet1D를 상속해 subgoal g_t(2048)를 conditioning에 추가.
combine 레이어 입력 차원만 변경하고 나머지 구조는 동일.
원본 droid_unet_diffusion.py는 수정하지 않음.
"""

import torch
import torch.nn as nn
from policy_heads.models.droid_unet_diffusion import ConditionalUnet1D


class ConditionalUnet1DWithSubgoal(ConditionalUnet1D):
    """
    ConditionalUnet1D + subgoal conditioning.

    combine: Linear(hidden_size + state_dim + subgoal_dim, hidden_size)
    forward에 subgoal 인자 추가.
    """

    def __init__(self, input_dim, global_cond_dim, subgoal_dim=2048,
                 diffusion_step_embed_dim=256, down_dims=[256, 512, 1024],
                 kernel_size=5, n_groups=8, state_dim=7):
        super().__init__(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            state_dim=state_dim,
        )
        # combine 레이어만 교체: subgoal_dim 추가
        self.combine = nn.Linear(global_cond_dim + state_dim + subgoal_dim, global_cond_dim)

    def forward(self, sample, timestep, global_cond=None, states=None, subgoal=None):
        """
        Args:
            sample:      (B, T, action_dim)
            timestep:    diffusion timestep
            global_cond: (B, seq_len, hidden_size) — VLM hidden states
            states:      (B, state_dim)             — qpos
            subgoal:     (B, subgoal_dim)           — g_t from SubgoalDiffuserMLP
        """
        sample = sample.moveaxis(-1, -2)

        # global conditioning: pool → norm → [cond, states, subgoal] → combine
        global_cond = self.global_1d_pool(global_cond.permute(0, 2, 1)).squeeze(-1)
        global_cond = self.norm_after_pool(global_cond)

        parts = [global_cond]
        if states is not None:
            parts.append(states)
        if subgoal is not None:
            parts.append(subgoal)
        global_cond = self.combine(torch.cat(parts, dim=-1))

        # 이하 원본과 동일 ─────────────────────────────────────
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-2, -1)
        return x


def build_subgoal_unet_from_checkpoint(base_embed_out, subgoal_dim=2048):
    """
    기존 ConditionalUnet1D 체크포인트에서 ConditionalUnet1DWithSubgoal를 초기화.
    combine 레이어를 제외한 모든 가중치를 복사.

    Args:
        base_embed_out: 로드된 기존 ConditionalUnet1D 모델
        subgoal_dim: subgoal 벡터 차원 (기본 2048)

    Returns:
        ConditionalUnet1DWithSubgoal (combine 레이어는 랜덤 초기화)
    """
    # 기존 모델에서 설정 추출
    old_combine = base_embed_out.combine        # Linear(hidden+state, hidden)
    global_cond_dim = old_combine.out_features
    state_dim = old_combine.in_features - global_cond_dim

    # 기존 모델에서 down_dims 추출
    down_dims = [m[0].blocks[0].block[0].out_channels
                 for m in base_embed_out.down_modules]

    new_unet = ConditionalUnet1DWithSubgoal(
        input_dim=base_embed_out.final_conv[1].out_channels,
        global_cond_dim=global_cond_dim,
        subgoal_dim=subgoal_dim,
        state_dim=state_dim,
        down_dims=down_dims,
    )

    # combine 제외 모든 가중치 복사
    old_state = base_embed_out.state_dict()
    new_state = new_unet.state_dict()
    for k in new_state:
        if k.startswith("combine."):
            continue   # 새 combine은 랜덤 초기화 유지
        if k in old_state and new_state[k].shape == old_state[k].shape:
            new_state[k] = old_state[k]
    new_unet.load_state_dict(new_state)

    return new_unet
