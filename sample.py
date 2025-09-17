# sample.py
import torch
from models.conditional_unet import ConditionalUNet
from PIL import Image
from torchvision import transforms
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalUNet().to(device)
model.load_state_dict(torch.load("checkpoints/flow_inpaint.pth"))
model.eval()

img = Image.open("test.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img = transform(img).unsqueeze(0).to(device)

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
masked_img = img * mask

x = torch.randn_like(img)
with torch.no_grad():
    for i in range(100, 0, -1):
        t = torch.full((1,), i / 100.0, device=device)
        dt = 1.0 / 100
        vt = model(x, t, masked_img)
        x = x - vt * dt

def tensor_to_pil(t):
    t = (t[0].cpu() + 1) / 2
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

from PIL import Image
Image.fromarray(tensor_to_pil(masked_img)).save("results/masked.png")
Image.fromarray(tensor_to_pil(x)).save("results/inpainted.png")
print("✅ 推理完成，结果已保存。")