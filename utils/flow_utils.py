# utils/flow_utils.py
import torch

def get_linear_noise_schedule(t):
    return 1 - t, t

def compute_flow_vector(x0, x1, t):
    # 确保 t 的形状与 x0, x1 匹配
    if t.dim() == 1:
        # 将 t 从 [B] 扩展为 [B, 1, 1, 1] 以匹配图像张量的维度
        t = t.view(-1, 1, 1, 1)
    
    # 扩展 t 到与 x0, x1 相同的形状
    t = t.expand_as(x0)
    
    s0, s1 = get_linear_noise_schedule(t)
    xt = s0 * x0 + s1 * x1
    vt = x1 - x0
    return xt, vt

def flow_matching_loss(model, x0, x_cond, t=None):
    """
    计算流匹配损失函数
    
    Args:
        model: 条件UNet模型
        x0: 目标图像 (干净图像)
        x_cond: 条件图像 (掩码图像)
        t: 时间步，如果为None则随机生成
    
    Returns:
        loss: 流匹配损失值
    """
    try:
        print(f"\n=== Flow Matching Loss Debug Info ===")
        # 确保所有输入张量在相同设备上
        device = x0.device
        x_cond = x_cond.to(device)
        
        # 获取批次大小
        batch_size = x0.size(0)
        
        # 如果没有提供时间步，则随机生成[0,1)范围内的时间步
        if t is None:
            t = torch.rand(batch_size, device=device)
        else:
            t = t.to(device)
        
        # 生成随机噪声图像x1
        x1 = torch.randn_like(x0, device=device)
        
        # 计算在时间t时的图像状态xt和对应的流场向量vt
        xt, vt = compute_flow_vector(x0, x1, t)
        
        # 打印调试信息
        print(f"Input shapes:")
        print(f"  x0 shape: {x0.shape}, dtype: {x0.dtype}")
        print(f"  x_cond shape: {x_cond.shape}, dtype: {x_cond.dtype}")
        print(f"  x1 shape: {x1.shape}, dtype: {x1.dtype}")
        print(f"  xt shape: {xt.shape}, dtype: {xt.dtype}")
        print(f"  t shape: {t.shape}, dtype: {t.dtype}")
        print(f"  vt shape: {vt.shape}, dtype: {vt.dtype}")
        
        # 模型预测流场向量
        # 参数顺序：噪声图像xt，时间步t，条件图像x_cond
        print(f"Calling model forward pass...")
        pred_vt = model(xt, t, x_cond)
        print(f"Model forward pass completed, pred_vt shape: {pred_vt.shape}")
        
        # 计算均方误差损失
        print(f"Computing loss...")
        loss = torch.mean((pred_vt - vt) ** 2)
        print(f"Loss computed: {loss.item()}")
        
        return loss
    except Exception as e:
        print(f"Error in flow_matching_loss: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise e