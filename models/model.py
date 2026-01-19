from typing import Tuple
import torch
import torch.nn as nn
from models.encoder_net import ObservationEncoder
from models.denoise_unet import InversionUnet
from functions.helper import obs_field_reconstruction



# Global Diffuser
class GlobalDiffuser(nn.Module):
    def __init__(self,
                # === field information
                field_size: int = 64, # Target field size
                field_channels: int = 1, # Number of channels in the inversion field
                # === observation set shape
                obs_vector_len: int = 21, # Observed vector length (coordinate (2) + observed vector length)
                # === Encoders params
                global_feat_len: int = 256, # Length of the global feature vector
                obs_element_num: int = None, # Number of elements
                # === Task params
                num_res_blocks: int = 2, #  Number of residual modules at each resolution
                attention_resolutions: Tuple[int, ...] = (16, 8), # Number of layers incorporating attention modules
                dropout: float = 0.1, # dropout rate
                channel_mult: Tuple[int, ...] = (1, 2, 4, 8), # Channel multiplier at each resolution
                num_heads: int = 8, # Number of heads in the main module
                spatial_recon: bool = True, # Should the Obs space field be rebuilt?
                uncertain_quant: bool = False, #  Should the side quest be activated: Quantifying Uncertainty
                 ):
        super().__init__()
        self.field_size = field_size
        self.obs_vector_len = obs_vector_len
        self.global_feat_len = global_feat_len
        self.field_channels = field_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.obs_element_num = obs_element_num
        self.spatial_recon = spatial_recon
        self.uncertain_quant = uncertain_quant

        # 1.0 Observation Encoder Create (Set Transformer) & Decoder Create (Light U-Net)
        self.obs_encoder = ObservationEncoder(field_size=self.field_size,obs_len=self.obs_vector_len,global_feat_len=self.global_feat_len)

        # 3.0 Main Task: DDPM Inversion Net Create (U-Net)
        self.inversion_net = InversionUnet(
                        field_size = self.field_size,
                        global_feat_len = self.global_feat_len,
                        field_channels = self.field_channels,
                        num_res_blocks = self.num_res_blocks,
                        attention_resolutions = self.attention_resolutions,
                        dropout = self.dropout,
                        channel_mult = self.channel_mult,
                        num_heads = self.num_heads,
                        spatial_feat_num = obs_vector_len-1 if self.spatial_recon==True else 1,
                        out_channels = self.field_channels+1 if self.uncertain_quant==True else self.field_channels,)

    def forward(self, x, t, observation):

        # Set Transformer Feature Encoding
        spatial_feat, global_feat = self.obs_encoder(observation)
        spatial_feat = spatial_feat.unsqueeze(1)
        global_feat = global_feat.squeeze(1)

        # if Spatial Recon -> concatenate Obs field
        if self.spatial_recon:
            obs_recon = obs_field_reconstruction(observation, self.field_size)
            spatial_feat = torch.cat([obs_recon, spatial_feat], dim=1)

        # Noise prediction
        output = self.inversion_net(x, t, spatial_feat, global_feat)

        return output