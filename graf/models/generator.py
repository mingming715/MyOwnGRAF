import numpy as np
import torch
from ..utils import sample_on_sphere, to_sphere, look_at
from ..transforms import FullRaySampler
from submodules.nerf_pytorch.run_nerf_mod import render, run_network
from functools import partial

class Generator(object):
    def __init__(self, H, W, focal, radius, ray_sampler, render_kwargs_train, render_kwargs_test, parameters, named_parameters,
                    range_u=(0,1), range_v=(0.01, 0.49), chunk=None, device='cuda', orthographic=False):
            self.device=device
            self.H=int(H)
            self.W=int(W)
            self.focal=focal
            self.radius=radius
            self.range_u=range_u
            self.range_v=range_v
            self.chunk=chunk
            