# 第16步：安全复现入口与离线验收

日期：2026-10-23。状态：PASS（入口、合成测试和静态构建审查）；未执行训练、测试集推理、统计量拟合或PDF重编。

## 运行位置与实验配置分离

`scripts/run_experiment.py` 继续读取原配置，不改动 `configs/formal_seed42.yaml`。新增 `--output-dir results/<新名称>`，训练日志、配置快照及summary保存到该目录，权重保存到其 `checkpoints/`。输出目录已存在（含空目录）即拒绝，不自动删除或复用；原 `results/formal_seed42/`、`models/formal_seed42/` 不可写。未指定新目录时仍保留原Smoke Test流程和防覆盖行为。

原seed42、Batch32、Adam/0.001、weight decay0、CrossEntropyLoss、40轮上限、patience10、P1–P5/P6和同一train normalizer保持不变。Val Accuracy优先、相同时更低Val Loss的选择规则不变。新增独立运行标识，不把复现运行冒充原冻结正式实验。

`--device cuda:N`选择GPU，无型号限制；`--device cpu`显式CPU；CUDA不可用默认失败，只有 `--allow-cpu-fallback`才允许回退。实际设备、GPU名（CPU时为null）、CPU回退标记、PyTorch及CUDA构建信息保存在runtime/provenance/summary与初始status中。配置中的原device字段保持原值，实际设备必须看runtime，不得仅凭配置声称GPU运行。

## 已实际通过的安全命令

从项目根目录使用现有wifi-csi解释器（命令中的python必须指向该解释器）：

```powershell
python -B scripts/run_experiment.py --help
python -B scripts/evaluate.py --help
python -B scripts/run_experiment.py --config configs/formal_seed42.yaml --output-dir results/reproduction_check --device cpu --dry-run
python -B scripts/run_experiment.py --config configs/formal_seed42.yaml --output-dir results/reproduction_check --device cuda:0 --dry-run
python -B scripts/evaluate.py --output-dir results/reproduction_eval --checkpoint models/formal_seed42/epoch_016.pt --device cpu --check-only
python -B -m unittest discover -s tests -v
```

训练dry-run只检查配置、来源文件哈希、固定normalizer和设备，不构造真实Dataset、网络或优化器，不创建结果目录。评价check-only只读取checkpoint元数据/权重及配置来源，不构造Dataset，不运行网络，不计算测试指标，不创建结果目录。上述两个结果目录运行后仍不存在。

真实训练/评价需去掉相应预检开关，本阶段没有执行。其他GPU的训练数值一致性尚未验证；GPU型号分支由mock单测覆盖，不等于在其他GPU上做过训练。

## 评价checkpoint关联与保护

默认仍选择原Epoch16；仅允许显式新输出目录。`--checkpoint`必须与关联summary选出的最佳checkpoint字节一致，不能指定任意轮次。原权重还校验其冻结SHA256、Epoch16、参数量35,655、配置、验证指标、selection标记、Split及train normalizer来源。

若评价未来独立复现训练的权重，必须同时提供 `--training-summary`（该次运行的summary路径）；校验其完成状态、未使用test、验证历史选出的最佳轮次、checkpoint、配置和来源一致性。不会根据测试表现重新选模。预检不会读取真实MAT或调用指标函数。

## 合成测试

31个unittest全部通过。测试只用临时合成MAT、极小normalizer数组和内存元数据；固定划分函数要求完整网格，因此合成2240条短字符串元数据，不生成2240个MAT，不读取真实数据。

覆盖固定人员规则/顺序/重复异常，错误形状/复数/NaN/Inf，标准化的训练来源/人员/Split哈希/标准差，35,655参数与(2,52,950)→(2,7)，已有目录保护/路径逃逸，未选中checkpoint拒绝，无数据dry-run/check-only，以及显式CPU和设备回退策略。

首轮Windows沙盒系统Temp子目录ACL拒绝访问；测试改为在tests目录下创建专属TemporaryDirectory，每项由unittest cleanup正常清理。没有触碰正式文件，没有残留 `.pipeline_test_*` 目录。实际网络只对合成零Tensor做CPU前向，不调用backward或optimizer.step。

## 报告可移植性

仅按归档report_data.tex、architecture.tex、assets和正式图表即可重编报告，不需要原始数据、模型推理或重新拟合normalizer。main.tex仅替换字体初始化，正文保持原字节内容；Windows字体文件为可选分支，其他系统可用Noto/Fandol与TeX Gyre字体。当前构建不再必须引用已删除报告目录。

COMPILE.md改为PATH工具说明，无作者电脑的解释器或MiKTeX安装位置依赖。系统工具和字体仍须存在；本阶段不安装软件、不重编PDF，不能保证字体回退后的页数与原PDF完全相同。旧脚本/历史快照中的旧目录引用只作来源证据，不执行它们构建当前报告。

## 来源留痕

原训练、评价入口及原报告源码/编译说明/发布清单等原字节保留在 `provenance/step16/`，对应原summary、evaluation_metadata和final_provenance中的历史哈希。原正式实验provenance不改写。当前release_integrity.json只更新工程发布清单，用来核验新入口及归档材料；修订与冻结检查见 `provenance/step16/verification.json`。

未更改冻结配置、Manifest、Split、normalizer、checkpoint、CSV/JSON结果、图表或最终PDF；未Git add/commit/push，也未再次清理历史文件。
