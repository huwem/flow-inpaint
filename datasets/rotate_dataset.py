# datasets/celeba_dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from torchvision import transforms

class CelebADataset(Dataset):
    def __init__(self, data_root, img_size=256, train_size=1000, augment_size=200000):
        self.data_root = data_root
        self.img_size = img_size
        self.train_size = train_size
        self.augment_size = augment_size
        
        # 获取所有图像文件路径
        self.image_files = [f for f in os.listdir(data_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files = sorted(self.image_files)[:train_size]  # 只取前train_size张图片
        
        # 基础转换：调整大小和转换为张量
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 数据增强转换
        self.augment_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=30),  # 随机旋转
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        print(f"Loaded {len(self.image_files)} original images")
        
        # 计算需要生成的增强样本数
        self.total_size = min(augment_size, len(self.image_files))  # 限制不超过原始图像数量
        
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # 如果索引超出原始图像数量，则使用数据增强
        if idx < len(self.image_files):
            # 使用原始图像
            img_path = os.path.join(self.data_root, self.image_files[idx])
            image = Image.open(img_path).convert('RGB')
            image = self.base_transform(image)
        else:
            # 使用数据增强生成新样本
            # 随机选择一张原始图像进行增强
            original_idx = random.randint(0, len(self.image_files) - 1)
            img_path = os.path.join(self.data_root, self.image_files[original_idx])
            image = Image.open(img_path).convert('RGB')
            image = self.augment_transform(image)
        
        # 创建遮罩图像（用于条件输入）
        mask = self.create_mask(image.shape[1], image.shape[2])
        masked_image = image * mask
        
        # 返回顺序：masked_image, mask, original_image
        return masked_image, mask, image
    
    def create_mask(self, H, W, min_visible_ratio=0.1, max_visible_ratio=0.3):
        """
        创建遮罩：遮住大部分图片，只留下单块区域可见
        """
        mask = torch.ones(3, H, W)
        
        # 计算可见区域大小
        visible_ratio = random.uniform(min_visible_ratio, max_visible_ratio)
        visible_h = int(H * visible_ratio)
        visible_w = int(W * visible_ratio)
        
        # 随机确定可见区域位置
        rh = random.randint(0, H - visible_h)
        rw = random.randint(0, W - visible_w)
        
        # 设置可见区域
        mask[:, rh:rh+visible_h, rw:rw+visible_w] = 0  # 遮罩区域为0，可见区域为1
        return mask