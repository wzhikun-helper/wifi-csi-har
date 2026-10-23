# 一维时序深度学习笔记：对照 ESP-Fi HAR 基线理解训练过程

本文面向初学者，对照项目现有代码和冻结结果解释概念。**标注“教学示例”的数字仅用于解释；标注“正式结果”的数字来自已经完成的实验。** Shape、划分数量和模型结构属于本项目实际实现，不是另一次实验。

## 1. Train、Validation、Test 各负责什么

可以把训练集理解为练习材料，验证集理解为帮助选择学习方案的检查材料，测试集理解为方案冻结后的独立考核。这只是角色类比，三者都是真实带标签的数据。

| 集合 | 固定参与者 | 样本数 | 实际职责 |
| --- | --- | ---: | --- |
| Train | P1–P5 | 1400 | 更新网络权重；拟合标准化参数 |
| Validation | P6 | 280 | 选择最佳 checkpoint，判断是否早停；不更新权重 |
| Test | P7–P8 | 560 | 模型和预处理冻结后的一次正式评价 |

划分写在 `results/dataset_split.csv`。`CSIDataset` 根据 `split` 字段选行，不重新随机划分。训练时的 `shuffle=True` 只打乱训练样本的读取顺序，不会把验证或测试样本移入训练集。

**数据泄漏**就是把本应未知的信息提前用于训练或方案选择。例如：同一人的重复动作随机散落在训练与测试中；合并所有人员拟合均值和标准差；根据测试集表现重选 Epoch、学习率或处理方法。对于本项目的跨人员目标，这些做法都会破坏评价边界。

P6 虽然不参与反向传播，但其指标参与 checkpoint 选择，因此“最佳验证指标”不等于完全独立的最终测试证据。正式测试完成后，也不能因为结果较低就不断调整方案并重复测试，再只报告最好的一次。

## 2. Tensor、Dataset、DataLoader

**Tensor** 是有形状、数值类型和设备位置的多维数组。Shape 说明每一维有多少个元素，不直接说明物理含义；含义来自数据协议和代码。

本项目的 `X` 使用 `torch.float32`；标签 `y` 使用 `torch.long`，即 int64。Tensor 可以在 CPU 或 GPU 上，但不会因为 dtype 是 float32 就自动位于 GPU；训练入口明确调用 `.to(device)`。

**Dataset** 定义“如何取得一个样本”。`src/dataset.py` 中：

- `__init__`：读取划分元数据，检查样本 ID、固定人员分组、路径和数量；可选加载固定标准化参数。
- `__len__`：返回当前集合的样本数。
- `__getitem__(index)`：找到 MAT，使用 `loadmat` 读取 `CSIamp`，检查形状和有限值，按选项标准化，再转置、转换 dtype，返回 `(X, y)`。出错时明确报告样本及路径，不丢弃样本。

**DataLoader** 把 Dataset 取出的多个样本堆成 Batch，并管理读取顺序。PyTorch 的自定义 Dataset 和 DataLoader 的职责可参阅[官方入门教程](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)。

实际 Shape 流程：

```text
MAT 中 CSIamp：       (950, 52)
一个 Dataset 样本 X： (52, 950)          float32
一个样本的 y：        ()                 long 标量，值为 0–6
32 个样本组成 X：     (32, 52, 950)      Batch × 通道 × 时间
32 个样本组成 y：     (32,)              每个样本一个类别编号
```

原始动作编号 1–7 减一后对应标签 0–6。当前实现使用整数标签，不把每个标签转成七维 one-hot。转置是交换时间轴和通道轴，不能用简单 reshape 代替，否则可能改变数值排列的含义。

## 3. 训练集拟合标准化与数据隔离

每个通道独立计算 `z = (x - mean) / std`。均值描述中心位置，标准差描述波动尺度。标准化不修改原始 MAT，只改变模型收到的输入数值。

**教学示例，非实际拟合参数：** 若某通道训练均值为 50、标准差为 10，一个幅度值 70 就变为 `(70-50)/10=2`。这不表示每个样本都变成均值 0、标准差 1。

本项目实际做法是：对 P1–P5 的 **1400 个训练样本**，每通道合并所有 **950 个时间位置**，即每通道 **1,330,000 个值**。`scripts/fit_normalizer.py` 使用 float64 分块累积，计算总体标准差（ddof=0）。结果是两个形状均为 `(52,)` 的参数数组。

`models/csi_normalizer.npz` 保存均值、原始标准差、安全除数、近零掩码、训练人员、样本数、Split 哈希等。近零阈值为 `1e-8`，对应通道的除数设为 1，避免除零；本次实际没有近零标准差通道。

Train、Val、Test 必须加载同一份固定参数。禁止用 P6 或 P7–P8 重新拟合，也不在每个 Batch 内重新求均值来替代此协议。单个 Batch 只包含训练集的一部分，因此其标准化后的均值不必恰好为 0，标准差也不必恰好为 1。

`CSIDataset` **默认不标准化**，以保持早期读取检查的兼容性；正式训练和测试通过 `normalizer_path` 显式传入固定文件。以下只是接口阅读示例，本笔记不执行它：

```python
dataset = CSIDataset("train", normalizer_path="models/csi_normalizer.npz")
```

先前诊断中，原始 96 个训练样本的输入均值约 60.87、标准差约 24.64，随机初始化平均 Loss 约 9.01；加载训练集参数后，同种子、同一批样本的平均 Loss 约 1.9686。**这是训练前功能诊断，不是正式测试结果，也不能据此声称模型准确率提高。**

## 4. 轻量 1D CNN 的每一层在做什么

本项目唯一模型为 `src/model.py` 的 `LightweightCNN`，共 **35,655 个可训练参数**。类的 `__init__` 定义层，`forward` 规定数据依次经过哪些层。

| 层 | 输入 Shape | 输出 Shape | 初学者解释 |
| --- | --- | --- | --- |
| Conv1d(52,64,3,padding=1) | `(32,52,950)` | `(32,64,950)` | 在时间轴上滑动长度为 3 的窗口，学习 64 组特征 |
| ReLU | `(32,64,950)` | `(32,64,950)` | 将负数变为 0，正数保留，加入非线性 |
| MaxPool1d(2) | `(32,64,950)` | `(32,64,475)` | 每两个时间位置取最大值，缩短序列 |
| Conv1d(64,128,3,padding=1) | `(32,64,475)` | `(32,128,475)` | 从前一层特征中再学习 128 组特征 |
| ReLU | `(32,128,475)` | `(32,128,475)` | 非线性变换，不改变 Shape |
| AdaptiveAvgPool1d(1) | `(32,128,475)` | `(32,128,1)` | 每个特征通道沿时间取平均，保留一个汇总值 |
| Flatten | `(32,128,1)` | `(32,128)` | 合并通道和长度为 1 的轴，不合并 Batch 轴 |
| Linear(128,7) | `(32,128)` | `(32,7)` | 为每个样本输出七类原始分数 logits |

Conv1d 的“一维”指卷积核沿一条序列轴移动，并不是说输入只能有一个通道。第一层默认会综合全部 52 个输入通道；输出的 64 个通道是学到的特征，不是新增了 64 个物理子载波。`padding=1` 在两端补零，配合当前 kernel 和 stride 保持时间长度。

**教学示例，非真实特征值：** ReLU 把 `[-2, 0, 3]` 变成 `[0, 0, 3]`；最大池化在两个值 `[1, 4]` 中取 4。池化缩短序列、汇总局部信息，但也可能丢失细节。全局平均池化丢弃部分时间位置结构是否限制本实验性能，是待验证假设，不是已经证明的原因。

Conv 和 Linear 含权重与偏置；ReLU、这两种池化和 Flatten 本身不含可训练参数。参数总量可核对为 `10,048 + 24,704 + 903 = 35,655`。

逐层公式已与实际模型对象核对：Conv1 为 `(52×3+1)×64=10,048`，Conv2 为 `(64×3+1)×128=24,704`，Linear 为 `(128+1)×7=903`；括号里的 `+1` 对应每个输出通道/单元的偏置。原笔记的加和正确，本次仅补充核算依据。

## 5. Forward、Loss、Backward、Optimizer 的区别

一次训练 Batch 的主要过程与 `scripts/run_experiment.py` 的 `train_epoch` 一致：

```python
optimizer.zero_grad(set_to_none=True)  # 清掉上一个 Batch 的梯度
logits = model(X)                     # Forward：算出预测分数
loss = criterion(logits, y)           # Loss：比较预测与真实标签
loss.backward()                      # Backward：计算各参数的梯度
optimizer.step()                     # 使用梯度更新参数
```

这是已有训练流程的教学摘录，**不在笔记整理阶段执行**。

Forward 本身只计算输出。Loss 是可用于优化的误差度量；本项目使用 CrossEntropyLoss，把 `(B,7)` logits 与 `(B,)` 整数标签比较，得到一个标量。logits 可以为负，也不要求加起来等于 1；它们不是概率。

本项目不在模型输出后先加 Softmax 再送入 CrossEntropyLoss，因为该损失已经包含对应的稳定对数概率运算。测试脚本为了保存可解释分数，才对 logits 做 Softmax。[来源：PyTorch CrossEntropyLoss 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)

Backward 计算“参数往哪个方向改变会影响 Loss”，并把梯度放到参数的 `.grad` 中。**只执行 `backward()` 不等于已经更新权重**；真正执行更新的是本项目 Adam 的 `optimizer.step()`。梯度默认会累积，所以每个训练 Batch 前要清梯度。

若七类概率恰好均为 `1/7`，单样本交叉熵为 `-ln(1/7)=ln(7)≈1.94591`。这是理论参考值，不是要求随机初始化模型必须达到的 Loss；随机 logits 可能有明显偏置。

验证和测试使用 `model.eval()` 配合 `torch.no_grad()`，且不调用 backward/optimizer.step。`eval()` 设置层的运行模式，`no_grad()` 关闭梯度记录，二者职责不同；**只调用 eval 并不会自动关闭梯度**。本 CNN 没有 Dropout 或 BatchNorm，但仍保持标准的训练/评价模式切换。

NaN/Inf 检查覆盖输入、logits、Loss、梯度和更新后权重。有限值说明没有这类数值故障，不等于模型一定学得好。

## 6. Epoch、Batch 与 Early Stopping

一个 Batch 是一次处理的一小组样本；一个 Epoch 是训练集全部样本被遍历一轮。本项目 Batch Size=32，保留末尾不足 32 的 Batch（`drop_last=False`）：

| 集合 | 样本数 | 每轮 Batch 数 | 最后一个 Batch 的 X Shape |
| --- | ---: | ---: | --- |
| Train | 1400 | 44 | `(24,52,950)` |
| Validation | 280 | 9 | `(24,52,950)` |
| Test | 560 | 18 | `(16,52,950)` |

“训练 40 Epoch”在此是上限，不是必须跑满。配置中的 patience=10 表示连续 10 个 Epoch 未按规定改善时停止，不是整个训练只允许改善 10 次。

改善规则预先固定为：**Val Accuracy 更高优先；Accuracy 完全相同则 Val Loss 更低优先；两者完全相同不算改善。** 一旦改善就重置未改善计数。Val Loss 变小但 Accuracy 更低，也不算本协议中的改善。

实际最佳 Epoch 是 16，之后第 17–26 轮连续无改善，因此第 26 轮早停。正式训练共完成 `26×44=1144` 次参数更新；最终第 26 轮权重不等于被选中的最佳权重。正式测试固定使用 `models/formal_seed42/epoch_016.pt`，没有根据测试表现重新挑选。

**过拟合迹象**是模型在训练集越来越好，但对未参与更新的数据没有相应改善。本实验 Train Loss 从 1.936258 降至 0.927492，Train Accuracy 从 16.29% 升至 64.79%；Val Loss 从 1.958855 升至 3.186177，末轮 Val Accuracy 为 21.43%。这体现明显的泛化差距，不能称为训练和验证都良好收敛。

## 7. Accuracy、Precision、Recall、F1 与 Macro-F1

**Accuracy** 是正确样本数除以总样本数。

对某一类单独看，先把它当作“这一类”和“其他所有类”的二分类：TP 是真实属于该类且预测正确的数量；FP 是其他类被错报为该类；FN 是该类被漏报为其他类。

```text
Precision = TP / (TP + FP)       预测成这一类的样本中，有多少是真的？
Recall    = TP / (TP + FN)       真实属于这一类的样本中，找回了多少？
F1        = 2TP / (2TP + FP + FN)
Macro-F1  = 七个类别 F1 的等权平均
```

本项目固定计算全部七类，零分母按 0 处理。Macro-F1 不是先平均 Precision/Recall 后再求 F1，也不是按 Support 加权；它关注每个动作类别，防止只看总正确率而忽略某些类很差。

混淆矩阵**行=真实类别，列=预测类别**。对角线是正确分类；行和是该真实类别的 Support，列和是被预测成该类的数量。

**正式测试中的真实例子：** falling 正确 47 个，真实 falling 共 80 个，被预测为 falling 共 84 个。因此 Precision=`47/84≈55.95%`，Recall=`47/80=58.75%`，F1=`94/164≈0.573171`。这里的数字来自冻结预测，不是教学构造的数据。

walking 的真实样本有 80 个，只正确识别 3 个，模型也只预测了 3 个 walking。所以 Precision=100%，Recall=3.75%，F1≈0.072289。**单看高 Precision 会误解识别能力**，必须结合 Recall 和 Support。

## 8. 正式实验结果与证据边界

本节均为**正式实验数字**；前期 2 Epoch Smoke Test 只是功能检查，不混入本结果。

| 项目 | 冻结结果 |
| --- | --- |
| Seed / 模型 | 42 / LightweightCNN，35,655 参数 |
| 最佳 checkpoint | `models/formal_seed42/epoch_016.pt` |
| 最佳 Epoch | **16** |
| 最佳 Val Accuracy | **27.14%**（27.142857…%，76/280） |
| 对应 Val Loss | 2.353830 |
| 实际训练轮数 | 26，触发 patience=10 早停 |
| Test Accuracy | **26.43%**（26.428571…%，148/560） |
| Test Macro-F1 | **0.257181**（未舍入 0.257180753398113） |
| P7 | Accuracy=23.9286%，Macro-F1=0.226313，280 样本 |
| P8 | Accuracy=28.9286%，Macro-F1=0.247832，280 样本 |

P8 的 Accuracy 比 P7 高 5 个百分点，不代表 P8 的所有动作都更容易识别。合并测试集的 Macro-F1 也不等于两个参与者 Macro-F1 的简单平均，因为先合并计数再求各类 F1 与先分别求 F1 是不同运算。

按 F1，falling 最好（0.573171），walking 最差（0.072289）。主要错误包括 walking→jumping 33 例、squatting→turning 32 例。它们是可核对的误分类事实，不能直接证明误差由动作相似性、人员体型或环境干扰中的某一个因素造成。

**观察事实：** 训练/验证差距扩大，测试表现有限；checkpoint 可重载，保存的 CSV 可独立复算指标，训练过程没有已检测出的 NaN/Inf。

**尚未验证的可能原因：** 跨人员输入分布变化、有限的训练人员、模型表达能力及时间汇总方式等可能影响泛化。没有针对性对照实验时，不能把这些假设写成定论，也不能通过反复使用当前测试集来验证它们。

**实验局限：** 只有单一 seed、单一固定划分、两位测试人员，未估计跨 seed 或划分的不确定性；同一参与者和环境内样本可能相关。实验不是留一环境测试，不能声称已验证跨环境泛化、在线实时识别或真实部署能力。工程流程执行成功与识别效果理想是两件事。

## 9. 沿着代码与结果回看

| 想理解的问题 | 对照文件/函数 |
| --- | --- |
| 一个样本如何读入 | `src/dataset.py`：`CSIDataset.__getitem__` |
| 标准化参数如何拟合 | `scripts/fit_normalizer.py`：`ChannelMoments`、`fit` |
| 一维网络如何连起来 | `src/model.py`：`LightweightCNN.__init__`、`forward` |
| 权重如何更新、验证如何隔离 | `scripts/run_experiment.py`：`train_epoch`、`validate_epoch` |
| 最佳模型如何选择 | `scripts/run_experiment.py`：`is_better`、`execute_run` |
| 测试标签和指标如何保存 | `scripts/evaluate.py` |
| 正式协议与训练指标 | `configs/formal_seed42.yaml`、`results/formal_seed42/metrics.csv` |
| 冻结验证与测试结果 | `results/formal_seed42/summary.json`、`test/test_metrics.json` |
| 各类错误与图表 | `results/formal_seed42/analysis.md`、`figures/formal_seed42/` |

最后两行中 `test/test_metrics.json` 相对于 `results/formal_seed42/`。训练数值诊断与标准化诊断是历史检查，不应在阅读笔记时自动执行；正式测试结果已经冻结，也不要为了阅读笔记重新评价。

复习时可以自问：为什么不能对 Val 单独拟合均值？为什么 `backward()` 后权重可能仍未变？为什么 `(950,52)` 要转成 `(52,950)`？为什么 walking Precision=100% 仍表现很差？为什么最佳 Epoch 不是最后一轮？答案分别来自数据隔离、优化器更新、Conv1d 轴约定、Recall/Support 和预先冻结的选择规则。
