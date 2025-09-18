# train.py (部分修改)
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
    
    # 创建TensorBoard日志目录
    writer = SummaryWriter(log_dir='runs/flow_inpaint_training')
    
    # 记录配置信息
    writer.add_text('Config', str(config))

    # 创建数据集和数据加载器
    print("Loading dataset...")
    try:
        dataset = CelebADataset(
            config['data_root'], 
            img_size=config['img_size'],
            train_size=config.get('train_size', 1000),
            augment_size=config.get('augment_size', 200000)
        )
        print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # ... 其余代码保持不变 ...