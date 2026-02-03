import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import operator
from vae_base import *
from restormer_arch import *

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
np.random.seed(0)


def _make_2d_grid(S, batchsize, device):
    """Build [0,1]×[0,1] spatial grid, shape (batchsize, 2, S, S), on given device."""
    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridx = gridx.reshape(1, 1, S, 1).repeat([batchsize, 1, 1, S])
    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridy = gridy.reshape(1, 1, 1, S).repeat([batchsize, 1, S, 1])
    return torch.cat((gridx, gridy), dim=1).to(device)


def _concat_grid_channel(x, grid):
    """Stack grid as extra channels: input (B,C,H,W), grid (B,2,H,W) -> (B, C+2, H, W)."""
    x_with_grid = torch.cat((x, grid.permute(0, 2, 3, 1)), dim=-1)
    return x_with_grid.permute(0, 3, 1, 2)


class VAE_fwd_2d_trans(nn.Module):
    """
    2D VAE with forward operator: transformer encoder to shared feature, then separate
    heads for mu/logvar; linear forward operator and next-step mu/logvar in flat latent;
    decoder reconstructs from latent. Optional spatial grid as extra input channels.
    """

    def __init__(self, in_channel, out_channel, dim=4, num_blocks=[2, 2, 2, 2], heads=[1, 2, 4, 4], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', init_scale=1, grid_info=True):
        super(VAE_fwd_2d_trans, self).__init__()
        self.latent_dim = dim * 2 * 2 * 2
        self.grid_dim = 2 if grid_info else 0

        self.patch_embed = OverlapPatchEmbed(self.grid_dim + in_channel, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample_4x(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample_4x(int(dim * 2 ** 2))
        self.latent_mu = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent_logvar = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample_4x(int(dim * 2 ** 3))
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.up3_2 = Upsample_4x(int(dim * 2 ** 2))
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.output = nn.Conv2d(dim, out_channel, kernel_size=3, stride=1, padding=1, bias=bias)

        self.encoder = nn.Sequential(
            self.patch_embed,
            self.encoder_level1,
            self.down1_2,
            self.encoder_level2,
            self.down2_3,
            self.encoder_level3,
            self.down3_4,
        )
        self.decoder = nn.Sequential(
            self.up4_3,
            self.decoder_level3,
            self.up3_2,
            self.decoder_level2,
            self.up2_1,
            self.decoder_level1,
            self.output,
        )
        self.forward_operator = nn.Linear(self.latent_dim * 2 * 2, self.latent_dim * 2 * 2, bias=False)
        self.next_mu = nn.Linear(self.latent_dim * 2 * 2, self.latent_dim * 2 * 2, bias=False)
        self.next_logvar = nn.Linear(self.latent_dim * 2 * 2, self.latent_dim * 2 * 2, bias=False)

    def reparameterize(self, mu, logvar):
        """Sample z ~ N(mu, diag(exp(logvar))) via reparameterization."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if self.grid_dim > 0:
            grid = self.get_grid(x.shape[1], x.shape[0], x.device)
            enc_in = _concat_grid_channel(x, grid)
        else:
            enc_in = x
        z_t = self.encoder(enc_in)
        mu1 = self.latent_mu(z_t)
        logvar1 = self.latent_logvar(z_t)
        logvar1 = torch.clamp(logvar1, -10, 10)
        x_identity = self.decoder(z_t)

        mu1_flat = mu1.reshape(mu1.size(0), -1)
        logvar1_flat = logvar1.reshape(logvar1.size(0), -1)
        z_t_re = self.reparameterize(mu1_flat, logvar1_flat)
        z_t1_re = self.forward_operator(z_t_re)

        mu2_flat = self.next_mu(z_t1_re)
        logvar2_flat = self.next_logvar(z_t1_re)
        logvar2_flat = torch.clamp(logvar2_flat, -10, 10)
        z_t1 = self.reparameterize(mu2_flat, logvar2_flat)
        z_t1 = z_t1.view(z_t1.size(0), self.latent_dim, 2, 2)

        x_pred = self.decoder(z_t1)
        return x_pred, x_identity, mu1_flat, logvar1_flat, mu2_flat, logvar2_flat

    def get_grid(self, S, batchsize, device):
        return _make_2d_grid(S, batchsize, device)

    def count_params(self):
        n = 0
        for p in self.encoder.parameters():
            n += reduce(operator.mul, list(p.size()))
        for p in self.decoder.parameters():
            n += reduce(operator.mul, list(p.size()))
        for p in self.forward_operator.parameters():
            n += reduce(operator.mul, list(p.size()))
        return n
