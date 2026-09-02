# 团队交接指南

更新时间：2026-09-02

本文档是三名队员及各自 Codex 的共同交接入口。开始工作前先读本文、`README.md`、
`AGENTS.md` 和 `docs/EXPERIMENTS.md`。

## 当前状态

- 私有 GitHub 仓库：`CheeseMirror-Gao/aicomp-uav-segmentation`
- 当前基线：单个 `SegFormer-B0`，公开初始权重 `nvidia/mit-b0`
- 基线代码提交：`e6c59a5 Add SegFormer B0 training baseline`
- 日志警告修复：`4bc0869 Avoid tensor conversion warning in training log`
- 数据体检：11 对样例图像/掩码均健康，尺寸为 `1024x1024`，标签 ID 为 `0-8`
- 云端环境：PyTorch 2.8.0、Python 3.12、CUDA 12.8、单卡 RTX 4090D 24GB 已验证
- 完整闭环已跑通：公开权重下载 → 微调 → 验证 mIoU → 保存检查点 → 生成预测掩码

## 已完成的冒烟实验

这不是正式成绩，只用于确认程序可以端到端运行。

```text
样例总数：11
训练/验证：9/2
模型：SegFormer-B0
训练轮数：2
batch size：2
crop size：512
学习率：0.00006
随机种子：3407
第1轮 mIoU：0.043504218693065026
第2轮 mIoU：0.07028324067868295
最佳 mIoU：0.07028324067868295
```

正式数据到手后应从公开 `nvidia/mit-b0` 权重重新训练，不要从这次 11 张样例的
`last.pt` 继续训练。

## 文件在哪里

GitHub 只保存代码和文档，不保存数据与权重。当前云端约定路径：

```text
/root/autodl-tmp/aicomp/repo/aicomp-uav-segmentation  # Git 仓库
/root/autodl-tmp/aicomp/datasets                       # 数据集
/root/autodl-tmp/aicomp/models                         # 公开预训练权重缓存
/root/autodl-tmp/aicomp/outputs/b0_smoke/best.pt       # 冒烟实验最佳权重
/root/autodl-tmp/aicomp/outputs/b0_smoke/last.pt       # 冒烟实验最后状态
/root/autodl-tmp/aicomp/outputs/b0_smoke/config.json   # 实验参数
/root/autodl-tmp/aicomp/outputs/b0_smoke/metrics.jsonl # 每轮指标
/root/autodl-tmp/aicomp/logs/b0_smoke.log              # 训练日志
```

这些云端文件是否仍存在，接手者必须通过 `ls` 验证，不能只依赖本文档。

## 新队员或新 Codex 如何接手

1. 接受 GitHub Collaborator 邀请并克隆仓库。
2. 打开仓库后，让 Codex 先读四个文件：`README.md`、`AGENTS.md`、本文和实验表。
3. 运行：

   ```bash
   git status
   git log --oneline -5
   python -m unittest discover -s tests -v
   ```

4. 不要立即重写训练代码；先说明准备验证的一个假设。
5. 需要云端操作时，通过私下渠道获取当前实例的 SSH 信息。密码和 Token 不得进入仓库或 Codex 提示词。
6. 登录云端后先检查：

   ```bash
   nvidia-smi
   df -h /root/autodl-tmp
   ls -lah /root/autodl-tmp/aicomp
   screen -ls
   ```

7. 群内声明已接管云端，确认无人正在训练后再操作。

## 日常协作流程

开始代码工作前：

```bash
git pull
git switch -c <姓名或缩写>-<任务名称>
```

修改后运行测试，提交并推送分支。合并前说明唯一改动、预期影响和验证结果。云端实验必须
使用一个全新的实验编号和输出目录，禁止覆盖已有结果。

开始训练前在 `docs/EXPERIMENTS.md` 预登记实验；结束后补充 mIoU、各类别 IoU、耗时和
结论。交接时更新本文档的“当前状态”和“下一步”。

## 训练会话交接

所有长训练都应在命名的 `screen` 会话中运行，例如：

```bash
screen -S a01
```

离开但不中断训练：按 `Ctrl+A`，松开后按 `D`。接手者查看和恢复：

```bash
screen -ls
screen -r a01
tail -n 50 /root/autodl-tmp/aicomp/logs/a01.log
```

继续训练使用对应实验的 `last.pt`；推理和提交候选使用验证 mIoU 最好的 `best.pt`。

## 关机与切换实例

按量实例关机后停止 GPU 计费，但不会预留 GPU。关机前必须：

1. 确认没有仍需运行的进程，或确认已保存 `last.pt`。
2. 保存 `best.pt`、`last.pt`、`config.json`、`metrics.jsonl` 和日志。
3. 将代码提交并推送 GitHub。
4. 更新本文档和实验表。
5. 将关键权重备份到本地或可靠网盘。

原实例无卡时，不要先删除原实例。使用 AutoDL 的克隆/迁移功能，优先同地区，并勾选复制
数据盘。新实例使用新的 SSH 地址、端口和密码。登录后执行：

```bash
nvidia-smi
df -h /root/autodl-tmp
ls -lah /root/autodl-tmp/aicomp
```

只有确认代码、数据、模型缓存、权重和日志完整后，才能释放旧实例。若未复制数据盘，必须
通过跨实例拷贝、AutoDL 网盘或本地备份恢复，不能假设 GitHub 包含这些文件。

## 当前下一步

1. 等待并下载正式训练数据和官方提交说明。
2. 运行完整数据体检，确认类别、目录、文件格式和数据规模。
3. 识别数据来源/场景信息，设计避免相邻 patch 泄漏的验证划分。
4. 从公开 `nvidia/mit-b0` 重新训练正式 B0 基线。
5. 在固定划分上依次比较 B1/B2、类别平衡损失、稀有类采样和跨域增强；每次只改变一项。
6. 最终提交只能使用一个模型，禁止模型集成。

