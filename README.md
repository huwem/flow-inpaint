# FlowInpaint: 基于流匹配（Flow Matching）的图像补全模型

> 🎯 使用 **流匹配（Flow Matching）** 实现图像修复（Inpainting），支持从遮挡图像恢复完整内容。

## 🔧 特性

- ✅ 基于 **条件流匹配（Conditional Flow Matching）**
- ✅ 支持任意遮挡补全（矩形、随机）
- ✅ 使用 U-Net 架构 + 时间嵌入
- ✅ 连续归一化流生成，无需 GAN
- ✅ 简洁、可扩展、注释清晰

## 📦 依赖

```
pip install -r requirements.txt
```

## 📥 数据准备

1. 下载 [CelebA 数据集](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. 解压后放入 `data/celeba/img_align_celeba/`
3. 示例路径：`data/celeba/img_align_celeba/000001.jpg`

## 🚀 训练

```
python train.py
```

## 🖼️ 推理

准备一张测试图 `test.jpg`，运行：

```
python sample.py
```

结果保存在 `results/` 目录。

## 🌟 后续扩展

- 支持文本引导补全
- 升级为 Rectified Flow 加速生成
- 添加 Wandb 日志

---

🚀 Enjoy!
