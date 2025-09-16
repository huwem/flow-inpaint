import torch
import random

def random_rectangle_mask(H=64, W=64, max_size=0.4):
    mask = torch.ones(1, H, W)
    h, w = int(H * max_size), int(W * max_size)
    rh = random.randint(0, H - h)
    rw = random.randint(0, W - w)
    mask[0, rh:rh+h, rw:rw+w] = 0
    return mask
