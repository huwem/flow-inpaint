# mini_demo.py
import torch
from models.conditional_unet import ConditionalUNet
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def mini_demo(image_path="test.jpg"):
    """
    简化版demo，用于测试训练后的模型
    """
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = ConditionalUNet().to(device)
    
    # 检查是否存在简化训练的模型
    model_path = "checkpoints/mini_flow_inpaint.pth"
    if not os.path.exists(model_path):
        # 如果简化模型不存在，尝试使用完整训练的模型
        model_path = "checkpoints/flow_inpaint.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载并预处理图像
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到测试图像: {image_path}，请提供一张测试图像")
        
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # 保存原始尺寸
    
    # 调整图像大小以匹配模型输入
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 创建反向遮罩：遮住大部分图片，只留下单块区域可见
    def create_inverse_mask_tensor(H, W, min_visible_ratio=0.1, max_visible_ratio=0.3):
        mask = torch.zeros(1, H, W, device=device)
        
        # 计算可见区域大小
        visible_ratio = random.uniform(min_visible_ratio, max_visible_ratio)
        visible_h = int(H * visible_ratio)
        visible_w = int(W * visible_ratio)
        
        # 随机确定可见区域位置
        rh = random.randint(0, H - visible_h)
        rw = random.randint(0, W - visible_w)
        
        # 设置可见区域
        mask[0, rh:rh+visible_h, rw:rw+visible_w] = 1
        return mask

    mask = create_inverse_mask_tensor(64, 64)
    mask = mask.unsqueeze(0).repeat(1, 3, 1, 1)  # 扩展到3个通道
    masked_img = img_tensor * mask

    # 使用模型进行推理
    x = torch.randn_like(img_tensor)
    with torch.no_grad():
        for i in range(50, 0, -1):  # 减少步数以加快推理速度
            t = torch.full((1,), i / 50.0, device=device)
            dt = 1.0 / 50
            vt = model(x, t, masked_img)
            x = x - vt * dt

    # 后处理函数
    def tensor_to_pil(t):
        t = (t[0].cpu() + 1) / 2
        img_array = t.permute(1, 2, 0).numpy()
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    # 保存结果
    original_resized = tensor_to_pil(img_tensor)
    masked_result = tensor_to_pil(masked_img)
    inpainted_result = tensor_to_pil(x)
    
    original_resized.save("results/original_resized.png")
    masked_result.save("results/masked.png")
    inpainted_result.save("results/mini_inpainted.png")
    
    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_resized)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(masked_result)
    axes[1].set_title("Masked Image")
    axes[1].axis("off")
    
    axes[2].imshow(inpainted_result)
    axes[2].set_title("Inpainted Result")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("results/mini_demo_result.png")
    plt.show()
    
    print("✅ 简化测试完成，结果已保存到 results/ 目录")

if __name__ == "__main__":
    mini_demo()