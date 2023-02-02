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

        select_inds = self.sample_rays(H, W)

        if self.select_inds:
            rays_o = rays_o.view(-1, 3)[select_inds]
            rays_d = rays_d.view(-1, 3)[select_inds]

            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds % W) / float(W) - 0.5
            hw = torch.stack([h, w]).t()
        else:
            rays_o = torch.nn.functional.grid_sample(rays_o.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_d = torch.nn.functional.grid_sample(rays_d.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_o = rays_o.permute(1,2,0).view(-1, 3)
            rays_d = rays_d.permute(1,2,0).view(-1, 3)

            hw = select_inds
            select_inds = None

        return torch.stack([rays_o, rays_d]), select_inds, hw
    
    def sample_rays(self, H, W):
        raise NotImplementedError
        
class FullRaySampler(RaySampler):
    def __init__(self, **kwargs):
        super(FullRaySampler, self).__init__(N_samples=None, **kwargs)

    def sample_rays(self, H, W):
        return torch.arange(0, H*W)