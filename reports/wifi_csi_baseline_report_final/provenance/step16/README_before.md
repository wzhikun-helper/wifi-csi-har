# ESP32 Wi-Fi CSI Cross-Person HAR Baseline

基于 ESP-Fi HAR 幅度序列的固定跨人员七分类基线。核心问题：**模型未见过某位参与者时，能否识别其动作？**

A lightweight PyTorch 1D CNN baseline for seven-class, cross-person Wi-Fi CSI activity recognition. The dataset audit, formal training, one frozen test evaluation, figures, notes and final report are complete. Cross-machine reproduction has not been validated.

## 1. 数据来源与实际结构 / Dataset

作者仓库：[AutoSmartGroup/ESP-Fi-HAR](https://github.com/AutoSmartGroup/ESP-Fi-HAR)。请通过作者提供的获取方式取得数据，并核对其引用与许可要求；本仓库不分发原始数据，也不提供猜测的直链。

本地已核实 **2240 对 MAT/CSV**，不是4480个独立样本。四个环境各560对，8名参与者、7类动作、10次重复。解压后放置为：

```text
data/raw/ESP-Fi_dataset/
├── EnvironmentNo.1(corridor)/
├── EnvironmentNo.2（Office）/
├── EnvironmentNo.3（boardrooms）/
└── EnvironmentNo.4（laboratory）/
```

保留原有目录名（含全角括号和大小写）。文件名 `X-Y-Z-M.mat` / 同主干CSV分别表示环境、参与者、动作、重复编号。Manifest路径相对于 `data/raw/ESP-Fi_dataset/`，不含本机绝对路径。

MAT变量 `CSIamp` 的形状为 `(950,52)`，即本项目使用的时间位置×幅度通道。不能由此推断采样率或持续秒数。本实验只读取幅度，不读取或处理相位。

| activity_id | 动作 / Activity | PyTorch标签 |
|---:|---|---:|
| 1 | 跑步 / running | 0 |
| 2 | 跌倒 / falling | 1 |
| 3 | 行走 / walking | 2 |
| 4 | 转身 / turning | 3 |
| 5 | 跳跃 / jumping | 4 |
| 6 | 下蹲 / squatting | 5 |
| 7 | 挥手 / waving | 6 |

## 2. 固定协议与方法 / Protocol and method

| 集合 | 参与者 | 样本数 | 用途 |
|---|---|---:|---|
| train | P1–P5 | 1400 | 拟合标准化、更新权重 |
| val | P6 | 280 | checkpoint选择与早停 |
| test | P7–P8 | 560 | 模型冻结后的正式评价 |

人员完全不重叠；各集合均含四环境和七动作。训练shuffle只改变读取顺序，不改变划分。

仅用1400个训练样本、每通道合并950个时间位置，以float64累积52通道均值及总体标准差（ddof=0），保存 `models/csi_normalizer.npz`。Train/Val/Test共用这一组固定参数。近零标准差阈值为1e-8，本次没有近零通道。Dataset执行 `(x-mean)/std` 后返回float32 `(52,950)`；Batch为 `(B,52,950)`，标签为int64。

唯一 `LightweightCNN`：

```text
Conv1d(52,64,k=3,padding=1) → ReLU → MaxPool1d(2)
→ Conv1d(64,128,k=3,padding=1) → ReLU
→ AdaptiveAvgPool1d(1) → Flatten → Linear(128,7)
```

参数量 **35,655 = 10,048 + 24,704 + 903**。输出原始logits，交叉熵损失前不加Softmax。

## 3. 正式冻结结果 / Frozen results

| 项目 | 结果 |
|---|---|
| Seed / Batch Size | 42 / 32 |
| 优化器 / 学习率 / weight decay | Adam / 0.001 / 0 |
| 损失 | CrossEntropyLoss |
| 最大轮数 / patience | 40 / 10 |
| 实际训练轮数 | 26，触发早停 |
| checkpoint规则 | Val Accuracy优先；相同时Val Loss更低 |
| 最佳Epoch | **16** |
| 最佳Val Accuracy / Val Loss | **27.1429% / 2.353830** |
| Test Accuracy | **26.4286%（148/560）** |
| Test Macro-F1 | **0.257181** |
| 最佳权重 | `models/formal_seed42/epoch_016.pt` |
| 原运行GPU | NVIDIA GeForce RTX 3060 Laptop GPU |

测试结果已冻结，不因分数低而重选模型、重拟合标准化或重测。训练集表现改善、验证表现不足，存在明显泛化差距。

- 正式训练记录：`results/formal_seed42/metrics.csv`（26轮）。
- 正式测试：`results/formal_seed42/test/` 下的 `test_metrics.json`、`predictions.csv`、`confusion_matrix.csv`、`classification_report.csv`、`evaluation_metadata.json`。
- 四张正式图：`figures/formal_seed42/`（每张PNG/PDF/SVG）。
- 分析：`results/formal_seed42/analysis.md`。
- 唯一保留的报告：[最终PDF](reports/wifi_csi_baseline_report_final.pdf)；可编辑源码及发布说明：[报告目录](reports/wifi_csi_baseline_report_final/RELEASE.md)。

`configs/baseline.yaml` 是 **2 Epoch Smoke Test**，输出在 `results/smoke_test/` 和 `models/smoke_test/`；`configs/formal_seed42.yaml` 才是正式配置。两者指标不能混用。正式模型只在验证改善时保存，因此没有Epoch26权重并非文件缺失。

## 4. 环境准备 / Environment

`environment.yml` 记录实测直接依赖，不包含本机prefix，不是已求解的跨平台锁文件。Python为conda-forge构建的3.11.16；核心科学计算包由pip安装。无须为本项目额外引入torchvision或torchaudio，代码未导入它们。

PyTorch实测 `2.14.0+cu126`、CUDA构建12.6，安装器记录为pip；原始安装索引和原轮子哈希没有保留。YAML选择当前可核实的[官方cu126索引](https://download.pytorch.org/whl/cu126/torch/)，其中有该版本的cp311 Windows x64轮子。这是可用来源核对，**不是对原安装来源的追溯证明，也不是安装成功验证**。

准备顺序：

1. 使用可用的Conda工具读取YAML，在独立环境中准备Python及pip依赖；不要覆盖正在使用的原环境。
2. 若YAML无法解析CUDA轮子，停止并从已核实的官方来源取得精确的 `torch==2.14.0+cu126`、Python 3.11、Windows x64轮子，再单独安装；不得静默替换CPU版或其他版本。
3. 核对驱动、PyTorch版本和CUDA可用性。本项目原训练入口要求CUDA，正式模式还校验RTX3060 Laptop GPU名称；不能承诺任意GPU直接运行。
4. 本次未安装依赖、未求解环境、未异机验证，因此不提供冒充已验证的环境创建命令。

YAML中的 `pypdf==6.10.0`、`pdfplumber==0.11.9` 来自独立报告解释器的实测信息，原 `wifi-csi` 环境未安装这两个包。Notebook/JupyterLab用于学习，九个实验脚本均可脱离Notebook运行。

PDF系统工具另行准备：XeLaTeX、ctex、相关LaTeX宏包、现有源码指定的宋体/黑体/Times New Roman/Arial/Consolas，以及Poppler页面渲染工具；不包含在Conda YAML中。源码仍有Windows字体路径适配，跨系统编译尚待验证。

## 5. 脚本职责与执行顺序 / Workflow

以下从项目根目录运行。表中的历史执行形式对应本项目已完成阶段，本次发布维护没有重复训练或评价。**已有结果会被保护：有的脚本仅接受完全相同的旧输出，其余会拒绝覆盖；不是可在当前归档目录连续重跑的一键命令。**

| 顺序 | 已有脚本 / 历史执行形式 | 职责与边界 |
|---:|---|---|
| 1 | `python -B scripts/build_manifest.py` | 按真实MAT文件名建立2240行清单并配对CSV；不读数组；已有输出相同才接受 |
| 2 | `python -B scripts/create_split.py` | 按人员增加split列，不随机重划分；已有输出不同则拒绝 |
| 3 | `python -B scripts/check_dataloader.py` | 读取真实单样本及Batch，检查shape/标签；会读取test样本，属于早期功能检查，不是正式准确率评价 |
| 4 | `python -B scripts/check_model.py` | 真实train Batch前向/反向功能检查，无参数更新 |
| 5 | `python -B scripts/check_numerics.py` | train Batch原尺度诊断；可用已有 `--normalizer` 参数检查标准化后尺度 |
| 6 | `python -B scripts/fit_normalizer.py` | 只用train拟合固定参数；已存在normalizer时拒绝覆盖 |
| 7 | `python -B scripts/run_experiment.py --config configs/baseline.yaml` | 2轮GPU Smoke Test；已有输出时停止 |
| 8 | `python -B scripts/run_experiment.py --config configs/formal_seed42.yaml` | 固定正式训练；已有输出时停止 |
| 9 | `python -B scripts/evaluate.py` | 固定Epoch16正式测试；已有test目录时停止，不能为发布重复运行 |
| 10 | `python -B scripts/make_figures.py` | 从冻结CSV/JSON绘图，无MAT/网络推理；默认已有图表时停止 |

正式重训尚缺独立输出入口：训练配置锁定原输出路径，评价也锁定结果位置。本次不删除归档结果来绕过保护、不修改这些入口。绘图已有新输出路径参数，但本文不将未实测的参数组合列为已验证命令。

## 6. 无原始数据的复现边界 / Without raw data

**可以做：** 阅读CSV、分析、笔记及PDF；核验已归档文件SHA256；使用固定数据片段、排版图副本和系统工具重编PDF。绘图逻辑不读MAT，可利用冻结预测和训练记录重建图表，但还要求Manifest/Split、normalizer及相关源码通过来源校验，并须选择不存在的新输出位置。

**不能做：** 重新建立真实数据清单、核验MAT内容、拟合normalizer、运行真实DataLoader、重训或对原测试集进行新推理。重新绘图不等于重新测试。

本次实际执行通过的只读发布核验入口（仅Python标准库，无MAT读取、无模型推理、无文件写入）：

```powershell
python -B reports/wifi_csi_baseline_report_final/verify_release.py
```

它检查当前保留的报告、图表、来源记录与冻结实验文件，不要求已删除的旧报告仍存在。旧 `verify_layout.py`、`COMPILE.md` 及旧 provenance 保留为历史记录，**不作为清理后的当前运行入口**；以 `RELEASE.md` 为准。

## 7. 项目结构、Git与证据 / Structure and evidence

```text
wifi-csi-har/
├── README.md, environment.yml, .gitattributes, .gitignore
├── data/raw/ESP-Fi_dataset/       # 不提交原始数据
├── data/processed/               # 当前空目录
├── src/                          # Dataset和唯一CNN
├── scripts/                      # 九个实验/检查脚本
├── configs/                      # Smoke与正式配置
├── models/                       # 固定normalizer；只放行最佳Epoch16权重
├── results/                      # Manifest、Split、Smoke、正式冻结结果
├── figures/formal_seed42/         # 正式图表和来源记录
├── notes/                        # CSI与深度学习笔记
├── notebooks/                    # 当前空目录
└── reports/
    ├── wifi_csi_baseline_report_final.pdf
    └── wifi_csi_baseline_report_final/  # 源码、排版副本和provenance
```

仓库地址：[wzhikun-helper/wifi-csi-har](https://github.com/wzhikun-helper/wifi-csi-har)。本次没有commit/push，不能把本机新增文件视为已发布到GitHub。

`.gitattributes` 对冻结证据禁用文本换行转换，保留现有字节；只为已有README和.gitignore保留原先LF Blob/CRLF工作副本映射。不运行批量归一化或 `git add --renormalize`。`.gitignore` 保留原始数据及其他权重保护，仅允许 `models/formal_seed42/epoch_016.pt`；报告build/qa中间产物不提交，正式PDF、源码、图表和来源记录保留。

按用户要求仅保留最终版本报告；旧版PDF和源码目录删除前，必要JSON/Markdown来源记录逐字节复制到最终目录的 `provenance/history/`。这些记录中的旧路径和旧哈希描述历史状态，不代表旧报告仍应存在。最终PDF、TeX、固定数据片段、正式图表和冻结结果未变。

## 8. 限制 / Limitations

只有一个seed、一次固定人员划分和两名测试人员，不能推断所有未见人员上的效果。四个环境均进入各集合，本实验是**跨人员**而非留一环境评价；未验证跨环境泛化、硬件迁移或真实部署可靠性。

未完成跨电脑安装和端到端复现；已有防覆盖路径和GPU型号限制仍需后续单独处理。原始数据许可及再分发条件应由使用者查阅作者说明；本仓库不重新分发数据。当前低测试分数如实保留，不作为重新选择模型的依据。
