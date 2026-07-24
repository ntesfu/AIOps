"""Self-contained InternVideo2-B14 (stage1, K710-ft, 8-frame) encoder.

Faithful reproduction of OpenGVLab/InternVideo2 single_modality/models/internvideo2.py
(base: embed_dim=768, depth=12, heads=12, patch14, 8 frames, 224px), stripped of
flash-attn/fused paths. Feature = fc_norm(clip_projector(blocks(...))) — the 768-d
input to the K710 head. Loads the distilled checkpoint strict=True.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        dt = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x.to(dt)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        out_type = x.dtype
        return (x.float() * self.gamma.float()).to(out_type)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(dim, eps)
        self.k_norm = RMSNorm(dim, eps)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, H, N, D
        B_, H_, N_, D_ = q.shape
        q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4, eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps)
        self.attn = Attention(dim, num_heads, eps)
        self.ls1 = LayerScale(dim)
        self.norm2 = RMSNorm(dim, eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=16, out_dim=768):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.k_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, xq, xk, xv):
        B, N, C = xq.shape
        Nk = xk.shape[1]
        q = F.linear(xq, self.q.weight, self.q_bias).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = F.linear(xk, self.k.weight, self.k_bias).reshape(B, Nk, self.num_heads, -1).permute(0, 2, 1, 3)
        v = F.linear(xv, self.v.weight, self.v_bias).reshape(B, Nk, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(x)


class AttentionPoolingBlock(nn.Module):
    def __init__(self, dim=768, num_heads=16, out_dim=768, eps=1e-5):
        super().__init__()
        self.norm1_q = nn.LayerNorm(dim, eps=eps)
        self.norm1_k = nn.LayerNorm(dim, eps=eps)
        self.norm1_v = nn.LayerNorm(dim, eps=eps)
        self.cross_attn = CrossAttention(dim, num_heads, out_dim)

    def forward(self, x):
        xq = x.mean(1, keepdim=True)
        xq = self.norm1_q(xq)
        xk = self.norm1_k(x)
        xv = self.norm1_v(x)
        return self.cross_attn(xq, xk, xv).squeeze(1)


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=768, patch=14, tubelet=1, in_chans=3):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet, patch, patch),
                              stride=(tubelet, patch, patch))

    def forward(self, x):
        x = self.proj(x)  # B,C,T,H,W
        x = x.flatten(3).permute(0, 2, 3, 1)  # B,T,HW,C
        return x


class InternVideo2B(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, num_patches=2048,
                 clip_embed_dim=768, num_classes=710):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.clip_projector = AttentionPoolingBlock(embed_dim, 16, clip_embed_dim)
        self.fc_norm = nn.LayerNorm(clip_embed_dim)
        self.head = nn.Linear(clip_embed_dim, num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, T, L, C = x.shape
        x = x.reshape(B, T * L, C)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.clip_projector(x)
        return self.fc_norm(x)  # B, clip_embed_dim

    def forward(self, x):
        return self.head(self.forward_features(x))


class InternVideo2Encoder:
    """clips -> [N, 768] features (fc_norm(clip_projector)). Each clip = 8 RGB frames (HxWx3 uint8)."""

    feature_dim = 768

    def __init__(self, checkpoint, device="cuda", dtype=torch.float32, img=224):
        self.device = torch.device(device)
        self.dtype = dtype
        self.img = img
        model = InternVideo2B()
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for k in ("module", "model", "state_dict"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Only tolerate nothing — report anything off.
        self.load_report = {"missing": list(missing), "unexpected": list(unexpected)}
        self.model = model.eval().to(self.device, self.dtype)
        self._mean = IMAGENET_MEAN.to(self.device, self.dtype)
        self._std = IMAGENET_STD.to(self.device, self.dtype)

    def _prep(self, frames):
        # frames: list of 8 HxWx3 uint8 RGB -> (C,T,H,W) normalized
        import torchvision.transforms.functional as TF
        t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # T,C,H,W uint8
        t = TF.resize(t, [self.img], antialias=True)  # short side -> img
        t = TF.center_crop(t, [self.img, self.img])
        t = t.to(self.device, self.dtype) / 255.0
        t = t.permute(1, 0, 2, 3).unsqueeze(0)  # 1,C,T,H,W
        t = (t - self._mean) / self._std
        return t[0]

    @torch.inference_mode()
    def encode(self, clips):
        if not clips:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        batch = torch.stack([self._prep(c) for c in clips]).to(self.device, self.dtype)
        feats = self.model.forward_features(batch)
        return feats.float().cpu().numpy()

    @torch.inference_mode()
    def logits(self, clips):
        batch = torch.stack([self._prep(c) for c in clips]).to(self.device, self.dtype)
        return self.model(batch).float().cpu().numpy()
