import torch
from math import sqrt, exp

from submodules.nerf_pytorch.run_nerf_helpers_mod import get_rays, get_rays_ortho

class RaySampler(object):
    def __init__(self,N_samples,orthographic=False):
        super(RaySampler,self).__init__()
        self.N_samples = N_samples
        self.scale = torch.ones(1,).float()
        self.return_indices = True
        self.orthographic = orthographic

    def __call__(self,H,W,focal,pose):
        if self.orthographic:
            size_h, size_w = focal
            rays_o, rays_d = get_rays_ortho(H,W,pose,size_h,size_w)
        else:
            rays_o, rays_d = get_rays(H,W,focal,pose)

        
class FullRaySampler(RaySampler):