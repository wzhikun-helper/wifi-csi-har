# 复现与实验协议指南

[项目首页](../README.md) · [文档导航](README.md)

## 环境准备

[environment.yml](../environment.yml)记录本机直接依赖，不含本机prefix，也不是已求解的跨平台锁文件。历史环境是Python 3.11.16和PyTorch 2.14.0+cu126，原GPU为RTX 3060 Laptop；不要求其他机器使用同型号GPU。

请在独立环境中准备依赖。原始PyTorch轮子索引及文件哈希没有保留，声明的官方cu126来源不代表原始安装来源已追溯。若精确版本无法解析，应停止并核对来源，不静默替换版本。跨机器安装与完整训练没有验证，本次不安装依赖。

报告工具pypdf/pdfplumber来自独立报告解释器，不是原训练环境依赖均已安装的证明。LaTeX、字体和Poppler属于额外系统工具，见[报告编译说明](../reports/wifi_csi_baseline_report_final/COMPILE.md)。

## 数据和预处理

依据[数据作者](https://github.com/AutoSmartGroup/ESP-Fi-HAR)的获取与许可说明准备数据，放在data/raw/ESP-Fi_dataset/，保留原有全角括号和大小写。原始MAT/CSV不公开。

本项目用MAT中的CSIamp，形状950×52，只使用幅度，不处理相位。文件主干X-Y-Z-M分别编码环境、参与者、动作、重复编号。一对MAT/CSV是一个样本。人员划分为P1–P5训练、P6验证、P7–P8测试；每组含全部四环境。

normalizer只用1400个训练样本，以float64累计52通道均值和总体标准差（ddof=0），近零阈值1e-8。固定参数应用于三组，Dataset返回float32的52×950张量和int64标签。实现与来源见[Dataset](../src/dataset.py)、[标准化程序](../scripts/fit_normalizer.py)、[人员划分](../results/dataset_split.csv)。

## 运行入口

首页的dry-run及check-only不会构造真实Dataset、训练、推理或创建输出。正式结果和模型目录始终受保护；新运行使用不存在的输出目录，包括空目录也拒绝覆盖。

| 入口 | 职责 |
|---|---|
| scripts/build_manifest.py | 文件名解析和MAT/CSV配对，不读取数组 |
| scripts/create_split.py | 固定人员划分 |
| scripts/check_dataloader.py | 真实样本和Batch功能检查，会读取数据 |
| scripts/check_model.py、check_numerics.py | 模型和训练Batch功能/数值检查 |
| scripts/fit_normalizer.py | 仅训练集拟合标准化 |
| scripts/run_experiment.py | 配置驱动训练；dry-run仅预检 |
| scripts/evaluate.py | 验证选定checkpoint评价；check-only不推理 |
| scripts/make_figures.py | 从保存结果绘图，不作为本次整理操作 |

完整执行顺序及安全命令见[流水线协议](../reports/wifi_csi_baseline_report_final/PIPELINE_REPRODUCTION.md)。不要在归档结果上按历史顺序直接重跑。

configs/baseline.yaml是2轮Smoke Test，不是正式配置。正式配置为[formal_seed42.yaml](../configs/formal_seed42.yaml)：seed42、batch32、Adam、学习率0.001、weight decay0、CrossEntropyLoss、最多40轮、patience10。历史训练26轮，最佳epoch16按Validation Accuracy选择，相同时比较Validation Loss。

设备需显式指定。CUDA不可用默认报错，仅--device cpu或--allow-cpu-fallback允许CPU；记录实际设备和回退，不把CPU结果当GPU结果。评价自有复现模型还需--training-summary与验证选模记录对应，不按test分数挑选checkpoint。

## 离线检查和公开冻结范围

```bash
python -B -m unittest discover -s tests -v
python -B reports/wifi_csi_baseline_report_final/verify_release.py
```

测试使用临时合成数据，不重新评价真实测试集。发布核验读取已保存结果和元数据；原始数据存在时只核对文件路径，不读MAT数组。模型权重不需要在来源审计中反序列化。

当前公开发布索引88项位于release_integrity.json；历史47项位于provenance/step16/verification.json。它们不是同一范围：历史47项中的10个中间checkpoint仅本地保留，公开克隆只能核验其中37项；当前88项公开文件可全部核验。README变更只更新当前索引对应项，不修改历史冻结清单或指标。

详细来源见[发布说明](../reports/wifi_csi_baseline_report_final/RELEASE.md)和[整理核验记录](COURSE_DELIVERY_CLEANUP.md)。provenance中的旧README是历史快照，不是当前运行入口。

## 评价边界

Accuracy是正确数/总数；Macro-F1为七类F1的等权平均。正式测试N=560，Accuracy为26.4286%，Macro-F1为0.257181。没有根据低测试分数重新训练或选择划分；本轮只读这些已保存指标。

单seed、固定划分和两名测试人员不足以证明广泛泛化；四个环境出现在每一组，不是跨环境评价。采样率、持续时间、硬件迁移和部署性能没有在本实验中建立证据。
