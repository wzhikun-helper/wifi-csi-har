# ESP-Fi HAR：Seed 42 正式实验分析

## 实验协议与结果来源

固定跨人员划分：P1–P5 训练（1400 样本），P6 验证（280），P7–P8 测试（560）。四个环境均包含在各集合中；这不是留一环境评估。仅使用训练集拟合的 52 通道均值和总体标准差，所有集合共用 `models/csi_normalizer.npz`。

唯一模型为 35,655 参数 LightweightCNN。Seed=42、Batch Size=32、Adam、学习率 0.001、weight decay=0、CrossEntropyLoss、上限 40 Epoch、patience=10；GPU 为 RTX 3060 Laptop GPU。以 Val Accuracy 优先、相同时 Val Loss 更低选择 checkpoint；完全相同时不算改善。正式配置保留于 `results/formal_seed42/config_snapshot.yaml`。

实际完成 26 Epoch，第 16 轮后连续 10 轮无改善而早停。冻结模型为 `models/formal_seed42/epoch_016.pt`。测试只执行一次，未根据测试结果重选 Epoch 或调整预处理。本分析和图表只读取已有 CSV/元数据，没有运行模型或读取 MAT。

## 观察事实

最佳 Epoch=16：Train Loss=1.230499、Train Accuracy=52.6429%；Val Loss=2.353830、Val Accuracy=27.1429%。

正式 Test Accuracy=26.4286%（148/560），Macro-F1=0.257181，固定包含全部七类，零分母按 0 处理。

| 测试人员 | 样本数 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| P7 | 280 | 23.9286% | 0.226313 |
| P8 | 280 | 28.9286% | 0.247832 |

P8 Accuracy 比 P7 高 5.0000 个百分点。两人的类别级错误结构可能不同；合并 Macro-F1 不等于两位人员 Macro-F1 的简单平均。这些差异是描述性观察，不代表具有统计显著性。

### 动作差异

| 动作 | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| running | 48.08% | 31.25% | 0.378788 | 80 |
| falling | 55.95% | 58.75% | 0.573171 | 80 |
| walking | 100.00% | 3.75% | 0.072289 | 80 |
| turning | 16.67% | 23.75% | 0.195876 | 80 |
| jumping | 9.52% | 15.00% | 0.116505 | 80 |
| squatting | 26.83% | 13.75% | 0.181818 | 80 |
| waving | 22.14% | 38.75% | 0.281818 | 80 |

按 F1，falling 最容易识别（0.573171），walking 最难（0.072289）。walking 的 Precision=100% 仅来自 3 个预测、3 个正确，真实 80 个 walking 中仅识别 3 个，Recall=3.75%，不能据此声称行走识别优秀。

### 主要错误分类

混淆矩阵行为真实类别、列为预测类别；以下比例以相应真实类别的 80 个样本为分母。

| 真实 → 预测 | 数量 | 真实类别内比例 |
| --- | ---: | ---: |
| walking → jumping | 33 | 41.25% |
| squatting → turning | 32 | 40.00% |
| jumping → turning | 29 | 36.25% |
| waving → jumping | 28 | 35.00% |
| running → falling | 28 | 35.00% |
| turning → jumping | 25 | 31.25% |
| turning → waving | 24 | 30.00% |
| walking → waving | 23 | 28.75% |

### 训练与验证走势

Train Loss 从 1.936258 降至 0.927492，Train Accuracy 从 16.2857% 升至 64.7857%。Val Loss 从 1.958855 升至 3.186177，末轮 Val Accuracy=21.4286%。训练优化正常，但验证损失总体上升、训练/验证差距扩大，存在过拟合或跨人员泛化不足的迹象。当前结果不支持良好的跨人员识别能力。

## 可能原因（尚未验证）

人员动作执行差异和传播条件差异可能引起输入分布偏移；有限训练人员与轻量模型的表示能力也可能影响泛化。平均时间池化可能削弱动作时序信息。这些都是待验证假设，不能仅凭当前混淆矩阵确定因果，也不能直接把特定混淆归因于动作相似性或采集质量。本阶段不为验证这些假设重跑测试或调整实验。

## 局限与解释边界

只有一个 seed、一次固定划分和两名测试参与者，没有重复实验的方差或置信区间。样本共享参与者和环境，不能将 560 个样本当作 560 名独立人员来估计泛化不确定性。P6 参与模型选择，其最优验证表现不属于独立测试证据。

各类别支持数均为 80，类别样本数平衡不代表各类同样容易识别。此实验未做跨环境留出评估，也未验证实时部署、在线流数据、硬件迁移或真实场景可靠性。不能将结果解释为跨环境泛化或真实部署能力。

## 图表及图注

图中准确率与 Precision/Recall/F1 均使用百分比；混淆矩阵使用原始计数。曲线完整保留 26 个 Epoch 的记录点，仅连接实际点，不平滑、不重采样、不补点。没有跨 seed/划分的重复实验，故不添加误差条或置信区间。

1. `loss_curve`：训练损失下降而验证损失总体升高，提示泛化差距。竖虚线标记验证规则选择的 Epoch 16；训练 n=1400、验证 n=280，Seed 42 单次运行。
2. `accuracy_curve`：训练准确率提升幅度大于验证集，最佳验证点为 Epoch 16；准确率单位为百分比。训练指标为该 Epoch 各训练 Batch 更新过程中的累计指标。
3. `confusion_matrix`：未见人员上的错误分布不均；行=真实、列=预测，每格为样本数。测试 n=560、每类 n=80，Epoch 16 单次测试。
4. `per_class_metrics`：falling 的 F1 最高；walking 的高 Precision 与低 Recall 并存。每类 n=80，全部来自同一冻结测试预测。

每张图保存 PNG（300 dpi）、PDF、SVG，目录 `figures/formal_seed42/`；PNG 用于预览，PDF/SVG 保留可编辑文字。图形使用英文动作标签，避免中文字体缺失；颜色与线型/标记或填充纹理共同编码。

## 来源、数值核验与重新生成

绘图来源：`results/formal_seed42/metrics.csv`、`test/predictions.csv`、`test/confusion_matrix.csv`、`test/classification_report.csv`（后三者相对于正式结果目录）。从预测 CSV 独立复算总体、按人员和按类指标，并与冻结结果、报告和混淆矩阵核对；图形 artist 的曲线点、矩阵和柱高均与 CSV 数值核对。

生成脚本：`scripts/make_figures.py`。来源文件 SHA256、配置路径、图表清单、输出 SHA256 与自动版面检查记录于图表目录的 `figure_provenance.json` 和 `figure_manifest.csv`。自动几何检查不能替代实际图像检查。

本次命令（在项目根目录运行）：

```powershell
& 'D:\miniconda\envs\wifi-csi\python.exe' -B scripts/make_figures.py
```

保留现有产物并重新生成到新位置：

```powershell
python -B scripts/make_figures.py --output-dir figures/formal_seed42_reproduced --analysis-path results/formal_seed42/analysis_reproduced.md
```

脚本不覆盖已存在的输出，不修改冻结 CSV、checkpoint、标准化参数或实验配置；重新绘图不等于重新测试。当前图表结论仅适用于本次 ESP-Fi HAR 固定跨人员基线。
