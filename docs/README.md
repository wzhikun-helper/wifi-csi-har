# 文档导航

## 从哪里开始

- **了解实验**：[项目首页](../README.md)包含协议、模型和结果图表。
- **运行与复现**：[复现指南](REPRODUCIBILITY.md)解释环境、预处理、命令及公开范围。
- **阅读完整报告**：[正式PDF](../reports/wifi_csi_baseline_report_final.pdf)包含实验分析和来源。
- **补充基础知识**：下表中的CSI与Conv1d笔记可独立阅读。

| 材料 | 路径 |
|---|---|
| 1D CNN正式报告 | [PDF](../reports/wifi_csi_baseline_report_final.pdf)、[LaTeX源码](../reports/wifi_csi_baseline_report_final/main.tex)、[编译说明](../reports/wifi_csi_baseline_report_final/COMPILE.md) |
| Wi-Fi CSI基础 | [学习笔记](../notes/wifi_csi_notes.md) |
| Conv1d与一维时序深度学习 | [学习笔记](../notes/deep_learning_1d_timeseries_notes.md) |
| 数据和人员划分 | [样本元数据](../results/dataset_manifest.csv)、[人员划分](../results/dataset_split.csv)、[协议与命令](../reports/wifi_csi_baseline_report_final/PIPELINE_REPRODUCTION.md) |
| 冻结基线结果 | [测试指标](../results/formal_seed42/test/test_metrics.json)、[分析](../results/formal_seed42/analysis.md) |
| 训练和评价程序 | [训练](../scripts/run_experiment.py)、[评价](../scripts/evaluate.py)、[离线测试](../tests/test_pipeline.py) |
| 实验验收 | [当前发布说明](../reports/wifi_csi_baseline_report_final/RELEASE.md)、[只读核验脚本](../reports/wifi_csi_baseline_report_final/verify_release.py) |
| 本次整理 | [清理及验收记录](COURSE_DELIVERY_CLEANUP.md)、[候选表](CLEANUP_CANDIDATES.csv) |

47项历史冻结文件、模型权重、scaler、划分、正式测试结果和报告保持原字节。`provenance/`保留历史来源及脱敏对应记录，旧路径只是历史事实，不要求恢复已删除的旧报告。原始数据不公开；不能把MAT/CSV成对文件当成两个独立样本。本次只做离线合成测试和预检，不重训、不推理、不重新评价测试集。
