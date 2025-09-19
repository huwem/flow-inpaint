# utils/flow_utils.py
import torch
import torch.nn.functional as F

def get_linear_noise_schedule(t):
    return 1 - t, t

def compute_flow_vector(x0, x1, t):
    """
    计算从x0到x1的线性流场向量
    Args:
        x0: 起始点 (例如噪声图像) [B, C, H, W]
        x1: 终止点 (例如真实图像) [B, C, H, W]
        t: 时间步 [B, 1] 或标量
    Returns:
        vt: 在时间t处的速度向量 [B, C, H, W]
    """
    # 确保t的形状正确
    if isinstance(t, torch.Tensor) and t.dim() == 0:
        t = t.unsqueeze(0)  # 添加batch维度
    
    # 线性插值的流向量就是从x0到x1的方向向量
    # 对于线性路径: path(t) = (1-t) * x0 + t * x1
    # 速度向量: v(t) = d/dt path(t) = x1 - x0
    vt = x1 - x0
    return vt

def flow_matching_loss(model, clean_img, masked_img, mask=None, num_time_samples=1):
    """
    计算流匹配损失
    Args:
        model: 条件UNet模型
        clean_img: 真实图像 [B, C, H, W]
        masked_img: 带掩码的图像 [B, C, H, W]
        mask: 掩码 (可选)
        num_time_samples: 每个样本的时间点采样数
    Returns:
        loss: 流匹配损失
    """
    B, C, H, W = clean_img.shape
    device = clean_img.device
    
    losses = []
    
    for _ in range(num_time_samples):
        # 随机采样时间点 t ~ Uniform(0, 1)
        t = torch.rand(B, device=device)
        
        # 生成噪声样本 x0 ~ N(0, I)
        x0 = torch.randn_like(clean_img)
        
        # 使用线性插值计算 xt
        # xt = (1 - t) * x0 + t * x1
        t_expand = t.view(B, 1, 1, 1)
        xt = (1 - t_expand) * x0 + t_expand * clean_img
        
        # 计算真实流场向量 vt = x1 - x0
        vt_true = compute_flow_vector(x0, clean_img, t)
        
        # 模型预测流场向量
        vt_pred = model(xt, t, masked_img)
        
        # 计算均方误差损失
        # 只在掩码区域计算损失（如果提供了掩码）
        if mask is not None:
            # 扩展mask以匹配图像维度
            mask_expand = mask.view(B, 1, H, W).expand_as(vt_true)
            loss = F.mse_loss(vt_pred[mask_expand], vt_true[mask_expand])
        else:
            loss = F.mse_loss(vt_pred, vt_true)
            
        losses.append(loss)
    
    return torch.stack(losses).mean()