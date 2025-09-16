import gradio as gr
import torch
from PIL import Image
import numpy as np
import os

# 导入模型（确保路径正确）
from models.conditional_unet import ConditionalUNet

# 检查是否有预训练权重
MODEL_PATH = "checkpoints/flow_inpaint.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"请先训练模型并保存到 {MODEL_PATH}")

# 加载设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = ConditionalUNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def create_mask(image, bbox=[50, 50, 114, 114]):
    """在图像上创建矩形遮挡"""
    mask = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 0
    return mask

def preprocess_image(image):
    """转换为张量"""
    transform = lambda x: (x.astype(np.float32) / 255.0) * 2 - 1  # [0,255] -> [-1,1]
    return torch.tensor(transform(image)).permute(2, 0, 1).unsqueeze(0)

def postprocess_tensor(tensor):
    """转回图像"""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img + 1) / 2  # [-1,1] -> [0,1]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

@torch.no_grad()
def inpaint_image(input_img, steps=100):
    # 调整大小
    input_img = Image.fromarray(input_img).resize((64, 64))
    input_np = np.array(input_img)
    
    # 创建遮挡图
    mask = create_mask(input_np)
    masked_img = input_np * mask[..., None]

    # 预处理
    x_cond = preprocess_image(masked_img).to(device)
    x = torch.randn(1, 3, 64, 64).to(device)

    # 流匹配反向采样
    for i in range(steps, 0, -1):
        t = torch.full((1,), i / steps, device=device)
        dt = 1.0 / steps
        vt = model(x, t, x_cond)
        x = x - vt * dt

    # 后处理
    result = postprocess_tensor(x)
    original = np.array(input_img)
    masked = (masked_img).astype(np.uint8)

    return original, masked, result

# 构建 Gradio 界面
demo = gr.Interface(
    fn=inpaint_image,
    inputs=gr.Image(type="numpy", label="上传图像"),
    outputs=[
        gr.Image(type="numpy", label="原始图像"),
        gr.Image(type="numpy", label="遮挡图像"),
        gr.Image(type="numpy", label="补全结果")
    ],
    title="🎨 FlowInpaint - 基于流匹配的图像修复",
    description="上传一张人脸图像，系统将自动遮挡中间区域并进行修复。",
    examples=["test.jpg"],  # 准备一张测试图
    cache_examples=False
)

# 启动
if __name__ == "__main__":
    demo.launch()
