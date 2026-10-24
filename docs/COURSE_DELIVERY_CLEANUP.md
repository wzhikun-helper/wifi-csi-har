# 课程交付目录整理与安全验收

日期：2026-10-24。本轮只整理文档导航及核验；未训练、推理、重划分或重算测试性能。

## 盘点与候选处理

整理前：`main`，HEAD `9e4447a98435d755c2eb4b880be1afe1b105aa19`，89个跟踪文件，工作区干净。远程为`https://github.com/wzhikun-helper/wifi-csi-har.git`。完整HEAD、status、log、remote、ls-files、目录树、大小及逐文件基线在仓库外本地`course-delivery-backup-20261024/wifi-csi-har/`，不公开私有原始数据。

候选见[CLEANUP_CANDIDATES.csv](CLEANUP_CANDIDATES.csv)。没有发现可以证明安全且值得删除的缓存，因此本仓库**删除0文件、0字节**。

| 候选 | 决定 | 原因 |
|---|---|---|
| models/smoke_test、results/smoke_test | 保留 | 历史流水线来源，不凭smoke名称判定无价值 |
| 正式报告provenance | 保留 | 唯一历史来源、脱敏及失败/修订证据 |
| 正式报告assets | 保留 | LaTeX排版副本仍被源码引用，不是可随意删的重复图 |

未遍历或读取`data/raw/`中的私有MAT/CSV内容。没有删除项目外Fresh Clone或private-backups。

## 整理前后目录树

```text
整理前                           整理后
configs/ src/ scripts/ tests/    同左；业务代码和测试原字节保留
notes/                          同左；CSI、Conv1d学习笔记保留
models/ results/ figures/       同左；权重、scaler、划分、正式指标冻结
reports/                        同左；正式6页PDF、源码、provenance保留
data/raw/                       同左；私有、不公开
notebooks/                      原空目录不清除
README.md                       新增课程作业交付导航
                                docs/README.md
                                docs/COURSE_DELIVERY_CLEANUP.md
                                docs/CLEANUP_CANDIDATES.csv
```

盘点大小（不含`.git`及私有原始数据）：models约4.81 MB，reports约4.99 MB，figures约0.98 MB，results约0.69 MB，notes约0.026 MB。本轮保留现有全部成果，仅增加导航与审计文档。

## 发布索引与冻结边界

47项历史冻结表位于`reports/wifi_csi_baseline_report_final/provenance/step16/verification.json`，未修改，其中不含根README。

README属于当前88项发布索引。为使新导航可被当前核验器正确验证，只更新`release_integrity.json`中README的SHA256，不改其余87项、历史47项或真实实验元数据：

- 更新前：`a635ae0934ae16a05b66601e15d6b707635b778bbf8883f8921c696ed8af2333`
- 更新后：`c62fade7a227f0652e547c45b23b5590c19345d839862e8a24baf9d4cca3f0e7`

旧索引和README仍可通过整理前Git HEAD及仓库外备份查阅；当前发布索引的文档变化不被冒称为历史冻结哈希一致。新增docs由Git归档，不反向加入历史47项。没有修改用户级Skill安装。

## 验收结果

| 检查 | 结果 |
|---|---|
| 全部离线测试 | 31通过、0失败、0跳过；临时合成数据，不读取真实数据数组 |
| 训练CLI dry-run | PASS，不构造真实Dataset、不训练、不创建目标目录 |
| 评价CLI check-only | PASS，不反序列化checkpoint、不推理、不计算测试性能 |
| 历史冻结哈希 | 47/47一致 |
| 当前发布索引 | 88/88一致；两项公开脱敏记录通过；读取的仅是已存元数据和结果 |
| 正式报告 | 6页，PDF及LaTeX、assets、模型、scaler、两份笔记均未修改 |
| README与当前文档导航 | 相对链接检查；不含本机盘符绝对路径 |
| 非授权变化 | 对照整理前逐文件基线，除README及当前发布索引外无原文件改变 |

核验环境为既有`wifi-csi`环境，PyTorch 2.14.0+cu126，显式CPU预检；不安装依赖，不声明跨机器复现已完成。发布核验器仅检查原始样本路径是否存在，没有读取MAT数组。没有重新计算分类测试指标。

历史`provenance/step16/README_before.md`中两条相对链接仍描述当时根README的上下文，不是当前导航；它们在归档子目录直接点击会失效。为保护原始来源记录，不修改该快照；当前根README与[文档索引](README.md)提供正确入口。

原始数据、其他非发布checkpoint及本地运行材料继续保留但不公开。公开保留唯一最佳epoch016权重、正式normalizer及必要来源文件，不额外上传私有数据或临时日志。

## GitHub与全新克隆的独立复核

正常推送后从GitHub取得全新浅克隆。当前发布索引88/88、两项脱敏对应关系及正式6页报告通过；README和PDF的公开URL均返回HTTP 200。未从本地复制缺失权重到克隆中冒充远程交付。

完整历史47项只能在原工作副本核验：全新克隆能核验37项，另10项是原来就被Git忽略、从未进入整理前提交的中间checkpoint。具体为formal_seed42的epoch001、005、008、011、012、013、014、015以及smoke_test的epoch001、002。原项目本地10个文件仍存在且哈希正确，没有删除、改写或新增公开。整理前Git树也仅公开normalizer及epoch016，证明这不是本轮造成的缺失。

因此：**本轮整理与本地冻结保护PASS；公开发布范围的Fresh Clone核验PASS；从GitHub独立核验完整历史47项属于PARTIAL PASS（10项仅本地）**。不扩大公开权重范围，也不把37/47误写成47/47远程验收。跨机器复现、单次人员划分和低测试表现等既有研究限制不因目录整理而消失。剩余历史材料有明确来源价值，不再为减小体积强行清理。
