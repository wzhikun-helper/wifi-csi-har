# 当前报告构建与复现边界

当前交付为 `../wifi_csi_baseline_report_final.pdf`，6页，SHA256保持原冻结值。本阶段只调整字体加载的可移植性，**没有重编或覆盖PDF**。原完整源码和编译记录分别逐字节保留于 `provenance/step16/main_before.tex` 与 `provenance/step16/COMPILE_before.md`，对应原报告来源哈希。

## 可仅靠归档材料重建的部分

- main.tex加载font_config.tex、report_data.tex、architecture.tex。
- 两张训练曲线读取本目录assets/排版副本；混淆矩阵和分类指标图读取项目figures/formal_seed42/。
- 重编PDF及渲染页面不需要MAT、CUDA、checkpoint推理或重新拟合标准化。
- prepare_curve_layout.py重建排版副本只读冻结图表PDF。
- scripts/make_figures.py从CSV重画图还要求相关来源文件通过校验，须使用新输出位置，不覆盖正式图表。
- 训练、真实DataLoader检查和重新拟合normalizer才需要MAT，不属于报告构建。

## 系统工具与字体

使用系统PATH中的XeLaTeX/ctex、所需宏包及可选Poppler，不依赖作者电脑的Conda、MiKTeX或Codex缓存路径。Python报告依赖另见根目录environment.yml。

font_config.tex先检测现有Windows字体文件，为原电脑Fontconfig问题保留可选绕过；不存在时使用TeX Gyre西文字体，以及Noto CJK或TeX发行版Fandol中文字体。绝对路径仅为可选分支，不是其他系统的必需条件。所选字体若缺失仍需自行准备；本阶段未安装字体、未验证异机编译。

## 后续编译示例（本阶段未执行）

在本目录执行，输出只留在build，不自动覆盖交付PDF：

```powershell
New-Item -ItemType Directory -Path build -Force | Out-Null
xelatex -disable-installer -halt-on-error -interaction=nonstopmode -output-directory=build main.tex
xelatex -disable-installer -halt-on-error -interaction=nonstopmode -output-directory=build main.tex
pdftoppm -r 130 -png build/main.pdf build/page
```

每次编译须检查退出码并逐页看图。不同字体可能改变分页，未实测前不能保证新编译与原冻结PDF逐字节或逐页一致。

## 当前核验

从项目根目录执行：`python -B reports/wifi_csi_baseline_report_final/verify_release.py`。新的发布清单记录当前工程版本；前一清单归档为provenance/step16/release_integrity_before.json，原正式实验结果与provenance不改写。

当前main/font配置、图片及数据片段没有对已删除旧报告目录的必需引用。旧verify_layout.py、before快照和历史provenance可能仍含旧路径，仅作历史证据，不用于当前构建。
