import numpy as np
import torch
import imageio
import os

def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = np.arccos(1-2*v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi)*np.sin(theta)
    cz = np.cos(phi)
    s = np.stack([cx, cy, cz])
    return s

def sample_on_sphere(range_u=(0, 1), range_v=(0, 1)):
    u = np.random.uniform(*range_u)
    v = np.random.uniform(*range_v)
    return to_sphere(u, v)