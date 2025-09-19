import torch

def get_linear_noise_schedule(t):
    return 1 - t, t

def compute_flow_vector(x0, x1, t):
    s0, s1 = get_linear_noise_schedule(t)
    xt = s0 * x0 + s1 * x1
    vt = x1 - x0
    return xt, vt

def flow_matching_loss(model, x0, x_cond, t=None):
    B = x0.size(0)
    t = torch.rand(B, device=x0.device) if t is None else t
    x1 = torch.randn_like(x0)
    xt, vt = compute_flow_vector(x0, x1, t)
    pred_vt = model(xt, t, x_cond)
    return torch.mean((pred_vt - vt) ** 2)
