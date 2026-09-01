# AIC 2026 UAV Semantic Segmentation

第八届全球校园人工智能算法精英大赛“无人机低空航拍图像语义分割”团队代码仓库。

## 目录

- `configs/`：训练与推理配置
- `src/`：数据、模型、损失函数和训练核心代码
- `scripts/`：训练、验证、推理及提交脚本
- `tools/`：数据检查与辅助工具
- `docs/`：实验记录和技术文档
- `data/`：比赛数据，仅保存在本地，不提交
- `outputs/`：日志、权重和预测结果，不提交

## 协作约定

1. 禁止提交官方数据集、测试集、模型权重和账号密钥。
2. 每项实验记录配置、代码版本、随机种子和验证指标。
3. 训练与推理脚本应保持可复现。

## 数据体检

正式训练前，先检查图片与标签的配对、尺寸、标签ID、损坏文件和类别分布：

```powershell
python tools/check_dataset.py `
  --images data/train/images `
  --masks data/train/labels `
  --output outputs/dataset_check
```

工具默认检查官方规则中的 `1024x1024` 图像、标签ID `0-8`，并将ID `4`
视为忽略类。若正式数据目录或标签定义与此不同，应以赛方随数据发布的说明为准，
通过 `--expected-size`、`--class-ids` 和 `--ignore-ids` 调整。

输出文件：

- `dataset_summary.json`：总体完整性和各类别像素占比。
- `dataset_files.csv`：逐文件状态，便于定位缺失、损坏或异常样本。

运行自动化测试：

```powershell
python -m unittest discover -s tests -v
```
