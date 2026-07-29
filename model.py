from typing import Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from diffusion_utils import EDMLoss

# =====================================================================
# MODUL ATTENTION TAMBAHAN (OPSIONAL UNTUK DIUJI)
# =====================================================================
class GatedAttention(nn.Module):
    def __init__(self, token_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        g = self.gate(x)
        return x + (attn_out * g) # Residual + Gated Filtering

class HybridSkipAttention(nn.Module):
    def __init__(self, token_dim, num_heads):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=token_dim, 
            nhead=num_heads, 
            dim_feedforward=token_dim * 2,
            batch_first=True,
            activation='gelu'
        )

    def forward(self, x):
        # Memungkinkan sinyal melewati Transformer jika tidak cocok
        return x + self.layer(x)

# =====================================================================
# MODEL UTAMA PENULIS (HANYA BAGIAN INI YANG DISESUAIKAN)
# =====================================================================
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512, num_tokens=16, num_heads=4, attn_type='base'):
        super().__init__()
        self.dim_t = dim_t
        self.num_tokens = num_tokens
        
        assert dim_t % num_tokens == 0, "dim_t must be perfectly divisible by num_tokens"
        self.token_dim = dim_t // num_tokens

        # 1. Project input to hidden dimension (ASLI KODE PENULIS)
        self.proj = nn.Linear(d_in, dim_t)

        # 2. Time Embedding layers (ASLI KODE PENULIS)
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        # 3. MODUL ATTENTION (Ganti tipe di sini tanpa merusak struktur lain)
        self.attn_type = attn_type
        if attn_type == 'gated':
            self.transformer_layer = GatedAttention(self.token_dim, num_heads)
        elif attn_type == 'hybrid_skip':
            self.transformer_layer = HybridSkipAttention(self.token_dim, num_heads)
        else: # 'base' -> TransformerEncoderLayer standar bawaan Anda
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=self.token_dim, 
                nhead=num_heads, 
                dim_feedforward=self.token_dim * 2,
                batch_first=True,
                activation='gelu'
            )

        # 4. Final MLP (ASLI KODE PENULIS)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

    def forward(self, x, noise_labels, class_labels=None):
        # A. Time Embedding (ASLI)
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        
        # B. Proyeksi fitur awal (ASLI)
        h = self.proj(x)
        orig_shape = h.shape
        
        # C. Gabungkan waktu (ASLI)
        h = h + emb 
        
        # D. Reshape menjadi Token (ASLI)
        h_tokens = h.view(-1, self.num_tokens, self.token_dim)
        
        # E. Attention Layer
        h_tokens = self.transformer_layer(h_tokens)
        
        # F. Return to original & MLP (ASLI)
        h = h_tokens.view(*orig_shape)
        out = self.mlp(h)
        return out


class Precond(nn.Module):
    def __init__(self, denoise_fn, hid_dim, sigma_min=0, sigma_max=float('inf'), sigma_data=0.5):
        super().__init__()
        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class Model(nn.Module):
    def __init__(self, denoise_fn, hid_dim, P_mean=-1.2, P_std=1.2, sigma_data=0.5, gamma=5, opts=None, pfgmpp=False):
        super().__init__()
        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x):
        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()
