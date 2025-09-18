import torch
from torch.utils.data import DataLoader
from models.conditional_unet import ConditionalUNet
from datasets.celeba_dataset import CelebADataset
from utils.flow_utils import flow_matching_loss
from utils.visualize import save_inpainting_result
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def load_model(checkpoint_path, config, device):
    """
    从检查点加载模型
    """
    # 根据配置创建模型实例，确保与训练时参数一致
    model = ConditionalUNet(
        in_channels=3,
        cond_channels=3, 
        width=config.get('model_width', 64),
        num_blocks=config.get('model_num_blocks', 2)
    )
    
    # 加载检查点
    if os.path.exists(checkpoint_path):
        try:
            # 加载检查点，指定map_location确保设备兼容性
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 根据训练代码的保存方式，检查点包含model_state_dict键
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 兼容直接保存state_dict的情况
                model.load_state_dict(checkpoint)
                
            print(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading model from {checkpoint_path}: {e}")
            print("Using randomly initialized model")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model")
    
    return model.to(device)

def main():
    # 加载配置
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建必要的目录
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)

    # 创建数据集和数据加载器
    print("Loading dataset...")
    try:
        dataset = CelebADataset(config['data_root'], img_size=config['img_size'])
        print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True  # 加速数据传输到GPU
    )

    # 从/root/autodl-tmp/flow-inpaint/checkpoints中加载模型
    model = load_model("checkpoints/model_epoch_50.pth", config, device).to(device)

    
    # 从数据文件夹data/celeba中选择四张图片作为验证样本
    val_sample = None

    # 测试模型
    model.eval()  # 设置模型为评估模式
    print("Testing model...")

    # 只处理一个batch用于测试
    with torch.no_grad():
        # 获取第一个batch的数据
        first_batch = next(iter(dataloader))
        masked, _, clean = first_batch
        
        # 确保所有张量都在相同设备上
        masked, clean = masked.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        
        # 保存验证样本
        val_sample = (masked[:4].clone(), clean[:4].clone())
        
        # 生成可视化结果
        try:                    
            # 保存固定样本的可视化结果
            if val_sample is not None:
                val_masked, val_clean = val_sample
                val_masked, val_clean = val_masked.to(device), val_clean.to(device)
                save_inpainting_result(
                    model,
                    (val_masked, val_clean),
                    device,
                    f"{config['results_dir']}/test.png"
                )
                print(f"Test result saved to {config['results_dir']}/test.png")
                    
        except Exception as e:
            print(f"Failed to save visualization: {e}")

    print("Testing completed.")

if __name__ == "__main__":
    main()