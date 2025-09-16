import torch
from models.conditional_unet import ConditionalUNet
from PIL import Image
from torchvision import transforms
import numpy as np

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

mask = torch.ones_like(img)
mask[:, :, 20:50, 15:45] = 0
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
