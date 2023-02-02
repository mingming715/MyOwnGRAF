import numpy as np
import torch
import imageio
import os

def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = np.arccos(1-2*v)
    cx = np.sin(phi)*np.cos(theta)
    cy = np.sin(phi)*np.sin(theta)
    cz = np.cos(phi)
    s = np.stack([cx, cy, cz])
    return s

def sample_on_sphere(range_u=(0, 1), range_v=(0, 1)):
    u = np.random.uniform(*range_u)
    v = np.random.uniform(*range_v)
    return to_sphere(u, v)

def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshpae(1, 3)

    eye = eye.reshape(-1, 3) # 행의 위치에 -1이 들어있으면, 3개의 column을 가지도록 만듬
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1,1).repeat(up.shape[0], axis=0)

    #z축 설정
    z_axis = eye-at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))

    #x축 설정
    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

    #y축 설정
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

    r_mat = np.concatenate((x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(-1, 3, 1)), axis = 2)

    return r_mat