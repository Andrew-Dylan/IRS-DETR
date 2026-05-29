import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import C2f
from ..modules.conv import Conv

class WaveletPool(nn.Module):
    def __init__(self):
        super(WaveletPool, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                            hl[None,::-1,::-1], hh[None,::-1,::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)
    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y

class WaveletUnPool(nn.Module):
    def __init__(self):
        super(WaveletUnPool, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                            hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = torch.floor_divide(x.shape[1], 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y

# ============================================================
# SHSA (Single-Head Self-Attention) blocks
# ============================================================

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse_self(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class SHSA_GroupNorm(torch.nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class SHSABlock_FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.SiLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class SHSA(torch.nn.Module):
    """Single-Head Self-Attention"""
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = SHSA_GroupNorm(pdim)

        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(torch.nn.SiLU(), Conv2d_BN(
            dim, dim, bn_weight_init = 0))
        

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim = 1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim = 1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim = 1))

        return x

class SHSABlock(torch.nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups = dim, bn_weight_init = 0))
        self.mixer = Residual(SHSA(dim, qk_dim, pdim))
        self.ffn = Residual(SHSABlock_FFN(dim, int(dim * 2)))
    
    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))

class EFAModule(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(SHSABlock(self.c) for _ in range(n))

# ============================================================
# MFPM (Multi-Frequency Pyramid Module) chain
# ============================================================

class ExternalAttention(nn.Module):
    """External attention: O(N*S) complexity via shared external memory."""
    def __init__(self, in_planes, S=8):
        super().__init__()
        self.mk = nn.Linear(in_planes, S, bias=False)
        self.mv = nn.Linear(S, in_planes, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        queries = x.view(b, c, n).permute(0, 2, 1)          # (b, n, c)
        attn = self.softmax(self.mk(queries))                # (b, n, S)
        attn = attn / (1e-9 + attn.sum(dim=2, keepdim=True))  # double-norm
        x_attn = self.mv(attn).permute(0, 2, 1).view(b, c, h, w)
        return F.relu(x + x_attn)


def get_freq_indices(method):
    """Parse frequency selection method, e.g. 'top16' -> 16 top-frequency DCT indices."""
    assert method in ('top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32')
    num_freq = int(method[3:])
    if 'top' in method:
        mapper_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        mapper_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
    elif 'low' in method:
        mapper_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        mapper_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
    else:  # 'bot'
        mapper_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        mapper_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
    return mapper_x[:num_freq], mapper_y[:num_freq]


class MultiFrequencyChannelAttention(nn.Module):
    """DCT-based multi-spectral channel attention with cached frequency buffers."""
    def __init__(self, in_channels, dct_h, dct_w,
                 frequency_branches=16, frequency_selection='top', reduction=16):
        super().__init__()
        assert frequency_branches in (1, 2, 4, 8, 16, 32)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection + str(frequency_branches))
        mapper_x = [mx * (dct_h // 7) for mx in mapper_x]
        mapper_y = [my * (dct_w // 7) for my in mapper_y]

        # Cache DCT weight buffers in a list — avoids state_dict() iteration in forward
        self._dct_buffers = []
        for freq_idx in range(frequency_branches):
            buf = self._make_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels)
            self.register_buffer(f'dct_weight_{freq_idx}', buf)
            self._dct_buffers.append(buf)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        B, C, H, W = x.shape

        xp = x if H == self.dct_h and W == self.dct_w else \
            torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        avg_feat, max_feat, min_feat = 0, 0, 0
        for buf in self._dct_buffers:
            spectral = xp * buf
            avg_feat = avg_feat + self.avg_pool(spectral)
            max_feat = max_feat + self.max_pool(spectral)
            min_feat = min_feat - self.max_pool(-spectral)

        avg_feat = avg_feat / self.num_freq
        max_feat = max_feat / self.num_freq
        min_feat = min_feat / self.num_freq

        attn = F.sigmoid(self.fc(avg_feat) + self.fc(max_feat) + self.fc(min_feat))
        return x * attn

    @staticmethod
    def _make_dct_filter(H, W, mx, my, C):
        """Build a 2D DCT basis filter for given frequency indices."""
        w = torch.zeros(C, H, W)
        cos_x = torch.cos(math.pi * mx * (torch.arange(H) + 0.5) / H) / math.sqrt(H)
        cos_y = torch.cos(math.pi * my * (torch.arange(W) + 0.5) / W) / math.sqrt(W)
        if mx != 0:
            cos_x = cos_x * math.sqrt(2)
        if my != 0:
            cos_y = cos_y * math.sqrt(2)
        w[:, :, :] = cos_x[:, None] * cos_y[None, :]
        return w


class EA_MF(nn.Module):
    """External Attention + Multi-Frequency channel attention fusion."""
    def __init__(self, out_channel, hw, frequency_branches=16, frequency_selection='top'):
        super().__init__()
        self.ea = nn.Sequential(
            Conv(out_channel, out_channel, k=3, act=nn.ReLU),
            ExternalAttention(out_channel),
        )
        self.mfca = MultiFrequencyChannelAttention(out_channel, hw[0], hw[1],
                                                   frequency_branches, frequency_selection)
        self.proj = Conv(out_channel, out_channel, k=3, act=nn.ReLU)

    def forward(self, x):
        return self.proj(self.mfca(self.ea(x)))


class MFPM(nn.Module):
    """Multi-Frequency Pyramid Module — gated cross-scale fusion."""
    def __init__(self, in_channel, out_channel, hw=(20, 20)):
        super().__init__()
        self.ea_mf = EA_MF(out_channel, hw)

        self.conv1x1 = nn.ModuleList([
            Conv(c, out_channel, 1) if c != out_channel else nn.Identity()
            for c in in_channel
        ])

    def forward(self, inputs):
        g, x = inputs
        g = self.conv1x1[0](g)
        x = self.conv1x1[1](x)
        scale = torch.sigmoid(self.ea_mf(g + x))
        return F.relu(x * scale)
    
