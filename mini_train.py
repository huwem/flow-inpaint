# mini_train.py
import torch
from torch.utils.data import DataLoader
from models.conditional_unet import ConditionalUNet
from datasets.celeba_dataset import CelebADataset
from utils.flow_utils import flow_matching_loss
import os
import yaml

def mini_train():
    # 从配置文件加载设置
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建必要的目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # 创建小型数据集（仅使用少量数据进行测试训练）
    dataset = CelebADataset(config['data_root'], img_size=config['img_size'])
    
    # 使用较小的batch size和较少的数据
    batch_size = min(16, config['batch_size'])  # 最多使用8个样本
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = ConditionalUNet().to(device)
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'] * 0.1)  # 使用更小的学习率
    
    print("🚀 开始简化训练...")
    
    # 只训练很少的epoch用于测试
    num_epochs = 50
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 只处理前几个batch以加快训练
        for i, (masked, _, clean) in enumerate(dataloader):
            if i >= 10:  # 只处理前3个batch
                break
                
            masked, clean = masked.to(device), clean.to(device)
            
            # 计算损失
            loss = flow_matching_loss(model, clean, masked)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "checkpoints/mini_flow_inpaint.pth")
    print("✅ 简化训练完成，模型已保存到 checkpoints/mini_flow_inpaint.pth")

if __name__ == "__main__":
    mini_train()