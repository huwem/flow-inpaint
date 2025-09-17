# datasets/celeba_dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from utils.mask_utils import inverse_rectangle_mask  # 更改导入的函数

class CelebADataset(Dataset):
    def __init__(self, root, img_size=64):
        self.root = root
        self.img_size = img_size
        self.filenames = [f for f in os.listdir(root) if f.endswith('.jpg')]
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.filenames[idx])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        # 使用新的遮罩函数，遮住大部分图片，只留下单块区域可见
        mask = inverse_rectangle_mask(self.img_size, self.img_size)
        masked_img = img * mask

        return masked_img, mask, img