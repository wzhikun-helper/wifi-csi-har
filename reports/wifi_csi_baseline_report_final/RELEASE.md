# 当前唯一报告：发布与来源说明

保留的交付为 `../wifi_csi_baseline_report_final.pdf`，6页，SHA256：
`b61846d000772ddf4a8dc1c3318ab3c01b99bc4e06b4f6dd6d88db87ef5e7839`。

2026-10-23按用户“只保留最终版本，其他版本删除”的补充指令，删除两份旧交付PDF及两个旧报告源码目录。删除前将必要JSON/Markdown来源与排版记录逐字节复制到 `provenance/history/initial/` 和 `provenance/history/academic/`。只保留记录，不暗中保留旧版PDF副本。删除清单与复制哈希见 `publication_changes.json`。

最终PDF、`main.tex`、`architecture.tex`、`report_data.tex`、排版副本、正式模型/CSV/图表均没有改动。`final_provenance.json`、`before_sha256.json`、`layout_validation.json`、`REVISION_LOG.md`和归档的`provenance/step16/COMPILE_before.md`描述原修订现场，作为历史证据保留；其中旧路径不再表示当前必需文件。

## 当前只读核验

从项目根目录执行本次已经实际通过的入口：

```powershell
python -B reports/wifi_csi_baseline_report_final/verify_release.py
```

仅用Python标准库，按 `release_integrity.json` 核验当前保留的实验/报告/来源文件，不要求旧PDF、旧源码目录、编译缓存或全部QA图片存在。读取已冻结CSV/JSON，不重新评价模型、不读取MAT数组、不写入验证结果。原始数据存在时仅检查文件路径；没有原始数据时仍可检查归档结果完整性。

旧 `verify_layout.py` 是针对一次排版操作的历史脚本，依赖已删除的旧报告和两处特定编译变化，**不再作为当前执行入口**。其字节与旧 provenance 中记录的一致，没有改写历史记录。

## 重编译边界

最终LaTeX仍用 `assets/` 两张曲线排版副本及 `../../figures/formal_seed42/` 正式图表；`report_data.tex` 是已冻结数值片段。重编PDF不需要原始数据、模型推理或重新生成标准化。

需要XeLaTeX/ctex、可用的Windows或Noto/Fandol字体和可选Poppler；工具位置由使用者环境决定。当前COMPILE使用PATH工具说明；原本机绝对路径记录另行归档。本次没有重编PDF或验证新机器工具链，因此不新增未经执行验证的编译命令。

历史已核实的编译选项为禁用自动安装、遇错停止、两遍解析引用、输出到build目录；未来重编应另存结果，不覆盖当前冻结PDF。`build/`和`qa/`不作为发布依赖，发布时被.gitignore忽略。

## 环境来源

实验直接依赖在根目录 `environment.yml`；pypdf/pdfplumber版本来自独立报告解释器。PyTorch安装器确认为pip，但原安装索引未保留；官方cu126索引的目标轮子存在，只证明可用来源，不证明本机原安装来源。未安装软件或执行跨电脑验证。

## 第16步工程修订

训练/评价入口支持独立结果目录及显式设备；`--dry-run`/`--check-only`不运行真实数据。新增合成数据单元测试，未训练或评价正式测试集。main.tex仅将原字体设置提取为font_config.tex，保留Windows文件路径可选绕过并加入非Windows字体回退。报告正文、数据片段和最终PDF未变。原入口、源文件和上一发布清单逐字节归档在provenance/step16/；当前修改清单与验收见PIPELINE_REPRODUCTION.md。
