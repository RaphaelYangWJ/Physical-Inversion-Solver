import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple



# === Positional Embedding
class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device # input: discrete timestamps [1,2,3,4,....,1000]
        half_dim = self.dim // 2 # compute half for sin and cos
        embeddings = math.log(10000) / (half_dim - 1) # compute scale factor
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings # time embeddings output


# === Cross Attention
class CrossAttentionBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int = None, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.cond_dim = cond_dim

        assert channels % num_heads == 0, "The number of channels must be divisible by num_heads."

        self.norm_x = nn.GroupNorm(32, channels)
        self.to_q = nn.Conv2d(channels, channels, 1)

        if cond_dim is not None:
            self.norm_c = nn.LayerNorm(cond_dim)
            self.to_kv_cond = nn.Linear(cond_dim, channels * 2)
        else:
            self.to_kv_self = nn.Conv2d(channels, channels * 2, 1)

        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, cond=None):
        b, c, h, w = x.shape
        residual = x

        # Q from image
        x_norm = self.norm_x(x)
        q = self.to_q(x_norm).reshape(b, self.num_heads, self.head_dim, h * w)

        if cond is not None and self.cond_dim is not None:
            # Cross-Attention
            if cond.dim() == 2:
                cond = cond.unsqueeze(1)  # (B, 1, cond_dim)
            cond_norm = self.norm_c(cond)
            kv = self.to_kv_cond(cond_norm)
            k, v = kv.chunk(2, dim=-1)

            k = k.permute(0, 2, 1).reshape(b, self.num_heads, self.head_dim, -1)
            v = v.permute(0, 2, 1).reshape(b, self.num_heads, self.head_dim, -1)
        else:
            # Self-Attention
            kv = self.to_kv_self(x_norm)
            k, v = kv.chunk(2, dim=1)
            k = k.reshape(b, self.num_heads, self.head_dim, h * w)
            v = v.reshape(b, self.num_heads, self.head_dim, h * w)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.reshape(b, c, h, w)

        out = self.to_out(out)
        return out + residual


# === Res Block
class ResBlock(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 res_dx: int = None, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels # input channels
        self.out_channels = out_channels # output channels

        self.norm1 = nn.GroupNorm(32, in_channels) # normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)


        # Condition inject
        if res_dx is not None:
            self.cond_linear = nn.Linear(res_dx, out_channels*2)

        # Time
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) # pass through MLP for time embeddings


        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels: # adjust channels for input and output
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb, res_dx=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # 时间嵌入
        h += self.time_mlp(time_emb)[:, :, None, None]

        # 条件嵌入
        if res_dx is not None:
            cond = res_dx.squeeze(1)
            cond = self.cond_linear(cond)
            scale, shift = cond.chunk(2,dim=1)
            h = h * scale[:,:,None,None]+shift[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x) # residual connection


# === Inversion Unet
class InversionUnet(nn.Module):

    def __init__(
            self,
            field_channels: int = None, # Number of channels in the inversion field
            field_size: int = None, # Dimensions of the inversion field
            num_res_blocks: int = 2, # Number of residual modules at each resolution
            attention_resolutions: Tuple[int, ...] = (16, 8), # Number of layers incorporating attention modules
            dropout: float = 0.1, # dropout rate
            channel_mult: Tuple[int, ...] = (1, 2, 4, 8), # Channel multiplier at each resolution
            num_heads: int = 8, # Number of heads in the main module
            spatial_feat_num: int = None, # Number of observation feature vector channels (= number of samples + 1)
            global_feat_len: int = None, # Length of the global feature vector
            out_channels: int = None, # Number of channels in the model output

    ):
        super().__init__()
        self.field_channels = field_channels
        self.in_channels = field_channels+spatial_feat_num
        self.field_size = field_size
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.spatial_feat_num = spatial_feat_num
        self.global_feat_len = global_feat_len
        self.out_channels = out_channels


        # Attention condition
        self.attention_cond = nn.Sequential(
            nn.Linear(self.global_feat_len, 64), # increase 4 times dims
            nn.SiLU(),
            nn.Linear(64, 32),)

        # Time Embeddings
        time_embed_dim = self.field_size * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(self.field_size), # generate time embeddings
            nn.Linear(self.field_size, time_embed_dim), # increase 4 times dims
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),)

        # Projection (in_channels -> model channels)
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(self.in_channels, self.field_size, 3, padding=1)])

        # Downsamples
        input_block_chans = [self.field_size] # store the channel size for each resolution for skip connections in upsamples
        ch = self.field_size
        ds = 1
        for level, mult in enumerate(channel_mult): # [1,2,4,8]
            for _ in range(num_res_blocks): # res block connections ***change point***
                layers = [ResBlock(
                    ch, mult * self.field_size, time_embed_dim,
                    self.global_feat_len, dropout
                )]
                ch = mult * self.field_size
                if ds in attention_resolutions:
                    layers.append(CrossAttentionBlock(ch, 32, num_heads)) # attention insert
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1: # last layer does not need donwsample
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_chans.append(ch)
                ds *= 2

        # Middle Blocks (Residual + Attention + Residual)
        self.middle_block = nn.Sequential(
            ResBlock(ch, ch, time_embed_dim,
                     self.global_feat_len, dropout),
            CrossAttentionBlock(ch, 32, num_heads),
            ResBlock(ch, ch, time_embed_dim,
                     self.global_feat_len, dropout),)

        # Upsamples
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]: # inverse
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(
                    ch + input_block_chans.pop(), # skip connection
                    mult * self.field_size,
                    time_embed_dim,
                    self.global_feat_len,
                    dropout
                )]
                ch = mult * self.field_size
                if ds in attention_resolutions:
                    layers.append(CrossAttentionBlock(ch, 32, num_heads)) # attention insert
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))

        # Project to field channels
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, self.out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, spatial_feat, global_feat):

        # Time Embeddings
        t_emb = self.time_embed(timesteps)
        # spatial_feat concatenation
        x = torch.concat([x,spatial_feat], dim=1)
        # global_feat concatenation
        res_dx = global_feat
        attention_dx = self.attention_cond(global_feat)


        # Downsample
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb, res_dx)
                    elif isinstance(layer, CrossAttentionBlock):
                        h = layer(h, attention_dx)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)

        # Middle Block
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb, res_dx)
            else:
                h = layer(h, attention_dx)

        # Upsample
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb, res_dx)
                elif isinstance(layer, CrossAttentionBlock):
                    h = layer(h, attention_dx)
                else:
                    h = layer(h)

        return self.out(h)