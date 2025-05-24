import torch
import torch.nn as nn
import torch.nn.functional as F

from conditional_unet_1d import *

# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim, To, Ta, Tp, noise_scheduler, use_ema=True, device=None):
        super().__init__()
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            diffusion_step_embed_dim=256,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8
        ).to(device)

        self.noise_scheduler = noise_scheduler

        if use_ema:
            self.ema = EMAModel(self.noise_pred_net, power=0.75)
        else:
            self.ema = None
        
        self.To = To
        self.Ta = Ta
        self.Tp = Tp
        self.device = device
    
    
        
