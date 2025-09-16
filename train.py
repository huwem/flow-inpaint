import torch
from torch.utils.data import DataLoader
from models.conditional_unet import ConditionalUNet
from datasets.celeba_dataset import CelebADataset
from utils.flow_utils import flow_matching_loss
from utils.visualize import save_inpainting_result
import yaml
import os

def main():
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)

    dataset = CelebADataset(config['data_root'], img_size=config['img_size'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    model = ConditionalUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    print("🚀 开始训练...")

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0.0
        for masked, _, clean in dataloader:
            masked, clean = masked.to(device), clean.to(device)
            loss = flow_matching_loss(model, clean, masked)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                save_inpainting_result(
                    model,
                    (masked[:4], clean[:4]),
                    device,
                    f"{config['results_dir']}/epoch_{epoch+1}.png"
                )

    torch.save(model.state_dict(), config['model_save_path'])
    print("✅ 训练完成，模型已保存。")

if __name__ == "__main__":
    main()
