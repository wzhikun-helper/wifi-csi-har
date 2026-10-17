# ESP32 Wi-Fi CSI Cross-Person HAR Baseline

## 1. 项目简介 / Overview

基于公开 ESP-Fi HAR 数据集，计划使用 Wi-Fi 信道状态信息（CSI）幅度序列和轻量级 PyTorch 1D CNN，建立七分类人体动作识别（HAR）基线，重点评估跨人员泛化能力。

This project plans to build a seven-class human activity recognition (HAR) baseline using Wi-Fi channel state information (CSI) amplitude sequences from the public ESP-Fi HAR dataset and a lightweight PyTorch 1D CNN, with an emphasis on cross-person generalization.

目前处于项目初始化阶段；尚未完成数据核查、正式训练或实验评估。本文描述的模型和流程均为计划，不代表已取得实验结果。

The project is currently being initialized. Data auditing, formal training, and evaluation are pending. The model and workflow below are plans, not completed experimental results.

## 2. 研究问题与目标 / Research question and objectives

**研究问题：当模型没有见过某位参与者的数据时，能否识别该参与者的动作？**

**Research question: Can the model recognize activities performed by a participant whose data it has never seen during training?**

目标是建立可复现的 CSI 幅度处理和 1D CNN 基线，严格按参与者划分数据，并用 Accuracy、Macro-F1 和混淆矩阵评估未见人员上的表现。

The objectives are to establish a reproducible CSI amplitude processing and 1D CNN baseline, separate data by participant, and evaluate unseen participants using Accuracy, Macro-F1, and a confusion matrix.

## 3. 数据集 / Dataset

拟使用公开 ESP-Fi HAR 数据集。下列数值来自任务规划，**均待实际核验**，不能视为数据审计结论：

The public ESP-Fi HAR dataset is the intended data source. The following values come from project planning and **must be verified against the actual dataset**:

| 规划信息 / Planned property | 预期值 / Expected value | 状态 / Status |
| --- | --- | --- |
| 样本数 / Number of samples | 2240 | 待核验 / Unverified |
| 环境数 / Number of environments | 4 | 待核验 / Unverified |
| 参与者数 / Number of participants | 8 | 待核验 / Unverified |
| 单样本维度 / Per-sample dimensions | `1 × 950 × 52` | 待核验 / Unverified |

维度轴的含义、采样方式、文件格式、人员与动作标签映射、环境分布、缺失与重复样本，以及许可和引用要求均待检查。上述维度不能直接当作 Conv1d 输入布局。数据来源和下载地址尚未核验，因此暂不提供链接。

Axis meanings, sampling, file formats, participant and activity label mappings, environment distribution, missing or duplicate samples, and license and citation requirements remain to be checked. The expected dimensions must not be treated directly as a Conv1d input layout. No download link is provided until the source is verified.

## 4. 七类动作 / Seven activity classes

以下为项目目标类别；正式类别编号、原始标签拼写和具体动作定义以数据核查结果为准。

These are the target classes. Numeric IDs, original label spellings, and precise activity definitions will be confirmed during the data audit.

| 中文 | English | 说明 / Description |
| --- | --- | --- |
| 跑步 | Running | 跑步动作 / Running activity |
| 跌倒 | Falling | 跌倒动作 / Falling activity |
| 行走 | Walking | 行走动作 / Walking activity |
| 转身 | Turning | 身体转向 / Body turning |
| 跳跃 | Jumping | 跳跃动作 / Jumping activity |
| 下蹲 | Squatting | 下蹲动作 / Squatting activity |
| 挥手 | Hand waving | 挥手动作 / Hand-waving activity |

## 5. 固定跨人员协议 / Fixed cross-person protocol

| 数据子集 / Split | 参与者 / Participants | 用途 / Purpose |
| --- | --- | --- |
| 训练集 / Training | P1–P5 | 拟合模型和预处理统计量 / Fit the model and preprocessing statistics |
| 验证集 / Validation | P6 | 选择超参数与检查点 / Select hyperparameters and checkpoints |
| 测试集 / Test | P7–P8 | 最终评估未见人员 / Final evaluation on unseen participants |

人员标签须先与原始数据对应。所有来自同一参与者的样本和派生窗口必须保留在同一子集中。归一化等需要拟合的统计量仅使用训练集估计；不得用测试集选择模型、阈值或超参数。该协议是跨人员评估，不能自动解释为跨环境评估。

Participant identifiers must first be mapped to the source data. All samples and derived windows from a participant must stay in the same split. Fit normalization and other learned preprocessing statistics on training data only. Do not use the test set to select models, thresholds, or hyperparameters. This is a cross-person protocol and does not by itself establish cross-environment generalization.

## 6. 基线方法 / Baseline method

计划从 CSI 中提取幅度，核实时间轴和子载波轴后，将输入转换为 PyTorch Conv1d 所需的 `(batch, channels, length)`。以子载波作为通道、沿时间轴卷积是待验证的候选布局。

The plan is to extract CSI amplitude, verify the time and subcarrier axes, and convert inputs to the PyTorch Conv1d layout `(batch, channels, length)`. Treating subcarriers as channels and convolving over time is a candidate layout to be validated.

候选模型为少量 Conv1d 特征提取层、非线性激活、时间池化和七分类输出层。具体层数、通道数、卷积核、正则化、损失函数、优化器和训练参数尚未确定；后续以固定配置和代码记录。当前尚无模型实现或训练完成的权重。

The candidate model uses a small number of Conv1d feature extraction layers, nonlinear activations, temporal pooling, and a seven-class output layer. Layer counts, channel widths, kernels, regularization, loss function, optimizer, and training settings are pending and will be recorded in code and fixed configurations. No model implementation or trained weights currently exist.

## 7. 评价指标 / Evaluation metrics

- **Accuracy**：正确分类样本占全部样本的比例。 / Fraction of samples classified correctly.
- **Macro-F1**：分别计算七类的 F1，再等权平均；固定包含全部七类，并明确零分母处理。 / Unweighted mean of per-class F1 over all seven classes, with explicit handling of zero denominators.
- **混淆矩阵 / Confusion matrix**：固定类别顺序，记录真实类别与预测类别的计数；若提供归一化版本，标明归一化方向。 / Counts of true versus predicted classes in a fixed class order; any normalized version must state its normalization axis.

正式指标、类别顺序、评估样本数和实验配置将在后续记录。当前没有任何实验指标或模型优劣结论。

Metrics, class order, evaluation sample counts, and experimental settings will be recorded later. No experimental scores or comparative model claims are available yet.

## 8. 目录结构 / Project structure

```text
wifi-csi-har/
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── models/
├── notebooks/
├── notes/
├── results/
├── scripts/
├── src/
├── configs/
└── reports/
```

| 目录 / Directory | 用途 / Purpose |
| --- | --- |
| `data/raw/` | 原始数据，原则上不直接修改，不提交 Git / Original data; generally kept unchanged and excluded from Git |
| `data/processed/` | 预处理数据 / Processed data |
| `figures/` | 实验图表 / Experimental figures |
| `models/` | 模型与检查点 / Models and checkpoints |
| `notebooks/` | Jupyter 学习与探索记录 / Jupyter learning and exploration notebooks |
| `notes/` | CSI 与深度学习笔记 / CSI and deep learning notes |
| `results/` | 正式 CSV 结果、数据清单与评估指标 / Formal CSV results, data manifests, and evaluation metrics |
| `scripts/` | 可执行实验脚本 / Executable experiment scripts |
| `src/` | 可复用 Python 模块 / Reusable Python modules |
| `configs/` | 固定实验配置 / Fixed experimental configurations |
| `reports/` | 最终 PDF 和报告材料 / Final PDFs and report materials |

空目录在普通 Git 提交中不会被保留；后续从 GitHub 获取项目时，按需补齐缺失目录。当前不添加占位文件。

Empty directories are not preserved by ordinary Git commits. Recreate missing directories as needed after obtaining the project from GitHub. No placeholder files are currently added.

## 9. 环境说明 / Environment

以下是当前电脑的审计结果，**不代表正式实验已经完成，也不是已锁定的跨电脑依赖清单**。

The following describes the audited local environment. **It is neither evidence of completed experiments nor a locked dependency specification for other machines.**

| 项目 / Item | 已核实结果 / Audited result |
| --- | --- |
| Conda 环境 / Environment | `wifi-csi` |
| Python | 3.11.16, 64-bit |
| PyTorch | 2.14.0+cu126 |
| PyTorch CUDA 构建 / CUDA build | 12.6 |
| `torch.cuda.is_available()` | `True` |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Jupyter Kernel | `wifi-csi` / Python (wifi-csi) |
| Notebook / JupyterLab | 7.6.2 / 4.6.3 |

本机 `conda.exe` 启动曾报“无效的系统 DLL 重定位”；通过 base Python 的 `-m conda` 查询成功。当前 PATH 未发现独立的 `jupyter` 命令，但使用 `wifi-csi` 的解释器调用 `python -m jupyter` 已通过检查。该异常尚未修复。

The local `conda.exe` launcher reported an invalid system DLL relocation error, while invoking `-m conda` through base Python succeeded. A standalone `jupyter` command was not found on the current PATH, but `python -m jupyter` succeeded using the `wifi-csi` interpreter. The launcher issue remains unresolved.

使用 Notebook 时选择 **Python (wifi-csi)**，并核对 `sys.executable` 指向该环境。不得将 base 或未明确指向环境的 `python3` Kernel 当成 `wifi-csi`。跨电脑安装说明和可复现依赖清单待后续验证后提供，本阶段不安装或升级依赖。

Select **Python (wifi-csi)** in notebooks and verify that `sys.executable` points to that environment. Do not substitute base or an ambiguous `python3` kernel. Installation instructions for other machines and a reproducible dependency specification will be provided after validation. No dependencies are installed or upgraded at this stage.

## 10. 运行步骤 / Workflow

以下为后续执行顺序，**尚无可直接运行的端到端训练命令**；脚本名称和参数将在实现并验证后补充。

This is the planned execution order. **No runnable end-to-end training command exists yet.** Script names and arguments will be documented after implementation and verification.

1. 核对解释器和 Notebook Kernel，处理影响使用的环境问题。 / Verify the interpreter and notebook kernel, and address environment issues that affect use.
2. 核实数据来源、许可、标签、维度与完整性，将原始数据放入 `data/raw/`。 / Verify the data source, license, labels, dimensions, and integrity, then place source data in `data/raw/`.
3. 按 P1–P5 / P6 / P7–P8 固定划分并审计泄漏。 / Apply the fixed participant split and audit for leakage.
4. 实现幅度预处理，仅用训练集拟合统计量。 / Implement amplitude preprocessing with statistics fitted only on training data.
5. 实现并训练轻量级 1D CNN，使用 P6 选择配置和检查点。 / Implement and train the lightweight 1D CNN, selecting configurations and checkpoints using P6.
6. 配置固定后，在 P7–P8 上进行最终评估，保存正式 CSV 和图表。 / Freeze the configuration, evaluate on P7–P8, and save formal CSV results and figures.
7. 整理复现说明和报告，生成并验证最终 PDF。 / Complete reproduction documentation and reports, then generate and verify the final PDF.

## 11. 当前进度 / Current progress

| 工作 / Work item | 状态 / Status |
| --- | --- |
| 项目目录创建与检查 / Directory creation and inspection | 已完成 / Complete |
| 本机开发环境审计 / Local environment audit | 已完成；Conda 启动入口异常待处理 / Complete; Conda launcher issue remains |
| README 与忽略规则 / README and ignore rules | 已建立 / Created |
| 数据来源、许可与实际数据审计 / Source, license, and actual data audit | 待完成 / Pending |
| 数据划分与预处理 / Splitting and preprocessing | 待完成 / Pending |
| 模型实现与正式训练 / Model implementation and formal training | 待完成 / Pending |
| 正式指标、CSV 与实验图表 / Formal metrics, CSV files, and figures | 待完成 / Pending |
| 报告与最终 PDF / Report and final PDF | 待完成 / Pending |
| Git 初始化、远程仓库与发布 / Git initialization, remote repository, and publication | 待完成；本阶段不执行 / Pending; outside this stage |
| 跨电脑复现验证 / Reproduction on another machine | 待完成 / Pending |

## 12. 限制与注意事项 / Limitations and cautions

数据规模和形状仍待核验。固定人员划分的可执行性需确认原始人员标签；不同人员的环境分布也可能影响解释。单次固定划分不能代表所有未见人员或真实部署条件，后续结论必须限定在实际评估范围内。

Dataset size and shape remain unverified. Applying the fixed split depends on source participant labels, and environment distribution across participants may affect interpretation. One fixed split cannot represent every unseen person or deployment condition; future claims must stay within the evaluated scope.

当前未锁定依赖、未固定训练随机种子，也未验证跨电脑复现。`.gitignore` 不会自动取消已有文件的 Git 跟踪；提交前仍需检查数据、权重、密钥和隐私信息。正式结果 CSV、图表、论文源文件和最终 PDF 不应被整体忽略。

Dependencies and training seeds are not yet fixed, and reproduction on another machine is untested. `.gitignore` does not untrack files already in Git; future submissions must be checked for data, weights, secrets, and private information. Formal CSV results, figures, paper source files, and final PDFs must not be excluded wholesale.

## 13. GitHub 后续复现说明 / Future GitHub reproduction

当前未初始化或发布本项目的 Git 仓库，因此没有可核实的仓库链接。后续发布前，应补齐经验证的数据获取与引用说明、依赖版本、类别映射、固定划分、预处理参数、随机种子、训练和评估命令、硬件信息以及正式结果，并在干净环境中验证完整流程。

This project has not been initialized or published as a Git repository, so no verified repository URL is available. Before publication, provide validated data access and citation instructions, dependency versions, class mappings, the fixed split, preprocessing parameters, random seeds, training and evaluation commands, hardware details, and formal results. Verify the full workflow in a clean environment.

获得公开仓库后，复现者应按已验证说明获取数据、补齐空目录、配置环境并运行固定协议。原始数据和大型权重的分发方式须符合许可，并在后续单独说明；本 README 暂不承诺尚未实现的复现命令。

Once a public repository exists, users should follow its validated instructions to obtain data, recreate empty directories, configure an environment, and run the fixed protocol. Distribution of raw data and large weights must comply with the applicable license and will be documented separately. This README does not promise commands that have not been implemented.
