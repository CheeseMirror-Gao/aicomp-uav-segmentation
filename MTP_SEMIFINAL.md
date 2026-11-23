# 无人机低空航拍图像语义分割

AIC 2026 半决赛团队项目。基于 MTP ViT-L / RVSA 与 UPerNet，对低空航拍图像逐像素预测背景、建筑、道路、水体、荒地、植被、农田和车辆。

[技术报告（PDF）](MTP_SEMIFINAL_REPORT.pdf) · [下载源码与复现说明](mtp-semifinal-source.zip)

```mermaid
flowchart LR
    A[航拍 RGB] --> B[三尺度与水平翻转]
    B --> C[512 像素滑窗]
    C --> D[MTP RVSA + UPerNet]
    D --> E[高斯加权拼接]
    E --> F[尺度与方向概率融合]
    F --> G[八类分割掩码]
```

## 方法

- **扩展到八类**。以公开的 MTP LoveDA 权重初始化，沿用七类分类参数，并添加独立的车辆分类分支。
- **兼顾全局和细节**。训练采样包含上下文与细节视图，推理对同一模型采用三尺度和水平翻转，在窗口重叠处用高斯权重融合 logits。
- **验证预测交付**。最终预测覆盖 3,594 张图。技术报告记录了文件格式检查和固定权重下的推理一致性核验。

## 代码

| 路径 | 内容 |
| --- | --- |
| `code/train.py` | 分阶段训练及恢复 |
| `code/inference.py` | 滑窗推理、掩码导出及 ZIP 打包 |
| `code/models/` | RVSA 骨干、UPerNet 及八类分割模型 |
| `code/configs/`、`code/utils/` | 训练与推理配置、采样、损失和运行实现 |
| `code/environment.yml` | Conda 环境 |
| `REPRODUCE.md` | 环境、输入格式、运行命令及复现范围 |

运行需要 Linux、兼容的 NVIDIA CUDA GPU 以及复现说明所列的外部数据和权重。

本仓库展示团队实现。第三方代码来源、改动说明和许可见 源码包中的 `code/THIRD_PARTY_NOTICES.md`。
