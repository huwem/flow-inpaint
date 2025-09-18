# mini_train.py
import torch
from torch.utils.data import DataLoader
from models.conditional_unet import ConditionalUNet
from datasets.celeba_dataset import CelebADataset
from utils.flow_utils import flow_matching_loss
from utils.visualize import save_inpainting_result
import yaml
import os

def mini_batch_train():
    """
    小批量训练代码 - 用于快速验证训练流程
    """
    # 加载配置
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改配置以进行小批量训练
    config['num_epochs'] = 3  # 只训练3个epoch
    config['batch_size'] = 4  # 小批量
    config['img_size'] = 256  # 设置图像尺寸为 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建必要的目录
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 创建小数据集
    print("Loading dataset...")
    try:
        dataset = CelebADataset(config['data_root'], img_size=config['img_size'])
        print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # 创建小批量数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2  # 减少worker数量
    )
    
    # 初始化模型并确保在正确设备上
    print("Initializing model...")
    model = ConditionalUNet(in_channels=3, width=64)  # 明确指定参数
    model = model.to(device)  # 确保模型在正确设备上
    print(f"Model device: {next(model.parameters()).device}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'] * 0.1)  # 降低学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    print(f"🚀 开始小批量训练测试 ({config['num_epochs']} epochs)...")
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 只处理前几个batch以加快测试
        for i, (masked, _, clean) in enumerate(dataloader):
            if i >= 5:  # 只处理前5个batch
                break
                
            # 确保所有张量都在相同设备上
            masked, clean = masked.to(device), clean.to(device)
            
            # 检查输入是否有效
            if torch.isnan(masked).any() or torch.isnan(clean).any():
                print("NaN values detected in input, skipping batch...")
                continue
                
            # 前向传播和损失计算
            try:
                # 参数顺序: model, 目标图像(clean), 条件图像(masked)
                # 在修复任务中，我们希望从masked图像生成clean图像
                loss = flow_matching_loss(model, clean, masked)
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Invalid loss value, skipping batch...")
                    continue
                    
            except RuntimeError as e:
                print(f"Error in flow_matching_loss: {e}")
                print("Skipping this batch...")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            print(f"  Batch {i+1}, Loss: {loss.item():.4f}")
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            scheduler.step()
            
            print(f"Epoch [{epoch+1}/{config['num_epochs']}], Average Loss: {avg_loss:.4f}")
            
            # 每个epoch后保存一次结果
            with torch.no_grad():
                try:
                    save_inpainting_result(
                        model,
                        (masked[:4].to(device), clean[:4].to(device)),  # 确保在正确设备上
                        device,
                        f"{config['results_dir']}/mini_batch_epoch_{epoch+1}.png"
                    )
                except Exception as e:
                    print(f"Failed to save visualization: {e}")
        else:
            print(f"Epoch [{epoch+1}/{config['num_epochs']}], No valid batches processed")
    
    # 保存模型检查点
    try:
        torch.save(model.state_dict(), "checkpoints/mini_batch_model.pth")
        print("✅ 小批量训练测试完成，模型已保存至 checkpoints/mini_batch_model.pth")
    except Exception as e:
        print(f"Failed to save model: {e}")

if __name__ == "__main__":
    mini_batch_train()