# sample.py
import torch
from models.conditional_unet import ConditionalUNet
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalUNet(in_channels=3, width=64).to(device)

# 使用weights_only参数加载模型，避免安全警告
try:
    model.load_state_dict(torch.load("checkpoints/flow_inpaint.pth", weights_only=True))
except FileNotFoundError:
    # 尝试加载mini_train生成的模型
    try:
        model.load_state_dict(torch.load("checkpoints/mini_batch_model.pth", weights_only=True))
        print("Loaded mini_batch_model.pth")
    except FileNotFoundError:
        # 尝试加载完整训练的模型
        try:
            model.load_state_dict(torch.load("checkpoints/final_model.pth", weights_only=True))
            print("Loaded final_model.pth")
        except FileNotFoundError:
            print("No checkpoint found. Using randomly initialized model.")
        except Exception as e:
            print(f"Error loading final_model.pth: {e}")
    except Exception as e:
        print(f"Error loading mini_batch_model.pth: {e}")
except Exception as e:
    print(f"Error loading flow_inpaint.pth: {e}")

model.eval()

# 检查test.jpg是否存在
if not os.path.exists("test.jpg"):
    print("test.jpg not found. Please provide a test image.")
    exit()

img = Image.open("test.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 与训练时保持一致
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img = transform(img).unsqueeze(0).to(device)

# 创建反向遮罩：遮住大部分图片，只留下单块区域可见
def create_inverse_mask_tensor(H, W, min_visible_ratio=0.5, max_visible_ratio=0.8):
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

mask = create_inverse_mask_tensor(256, 256)
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

# 创建结果目录
os.makedirs("results/test", exist_ok=True)

# 保存单个图像
original_img = tensor_to_pil(img)
masked_img_pil = tensor_to_pil(masked_img)
inpainted_img = tensor_to_pil(x)

from PIL import Image as PILImage
PILImage.fromarray(original_img).save("results/test/original.png")
PILImage.fromarray(masked_img_pil).save("results/test/masked.png")
PILImage.fromarray(inpainted_img).save("results/test/inpainted.png")

# 创建对比图
# 创建一个大的图像来容纳三个子图像
combined_width = 256 * 3 + 20 * 2  # 三个图像宽度 + 两个间隔
combined_height = 256 + 50  # 图像高度 + 标题空间
combined_img = PILImage.new('RGB', (combined_width, combined_height), (255, 255, 255))

# 粘贴图像
combined_img.paste(PILImage.fromarray(original_img), (0, 50))
combined_img.paste(PILImage.fromarray(masked_img_pil), (256 + 20, 50))
combined_img.paste(PILImage.fromarray(inpainted_img), (2 * (256 + 20), 50))

# 添加标题
try:
    draw = ImageDraw.Draw(combined_img)
    draw.text((50, 10), "Original", fill=(0, 0, 0))
    draw.text((256 + 20 + 50, 10), "Masked", fill=(0, 0, 0))
    draw.text((2 * (256 + 20) + 50, 10), "Inpainted", fill=(0, 0, 0))
except:
    pass  # 如果没有合适的字体则跳过添加文字

combined_img.save("results/test/comparison.png")

print("✅ 推理完成，结果已保存。")