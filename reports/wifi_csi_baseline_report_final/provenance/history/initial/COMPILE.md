# 编译与复核

源文件：`main.tex`；冻结数据生成器：`build_report.py`；自动数据片段：`report_data.tex`。
正式交付：项目根目录下 `reports/wifi_csi_baseline_report.pdf`。

先使用已有 `wifi-csi` 环境（生成器使用现有 NumPy 读取固定参数，不拟合统计量），在本目录执行：

```powershell
python -B build_report.py
New-Item -ItemType Directory -Path build -Force | Out-Null
xelatex -disable-installer -halt-on-error -interaction=nonstopmode -output-directory=build main.tex
xelatex -disable-installer -halt-on-error -interaction=nonstopmode -output-directory=build main.tex
Copy-Item -LiteralPath build/main.pdf -Destination ../wifi_csi_baseline_report.pdf
```

复制前应确认目标为本报告自己的旧版本，不覆盖不相关文件。源文件使用现有 XeLaTeX、ctex 和 Windows 字体；通过字体文件路径直接加载宋体、黑体及西文字体，绕过本机失效的 Fontconfig 配置，不修改系统配置。`-disable-installer` 禁止编译时自动安装缺失宏包。本项目未安装或升级依赖。

两次编译用于解析图表与参考文献编号。图片来自 `../../figures/formal_seed42/*.pdf`，不改变图表；不调用训练、测试或绘图脚本。若冻结来源哈希不一致，生成器应停止，不能用新数值冒充本次实验。

复核：用 `pdftoppm -r 110 -png ../wifi_csi_baseline_report.pdf <临时目录>/page` 逐页渲染；检查全部页的字体、边距、图注、表格、留白和参考文献位置。渲染 PNG 属于 QA 中间文件，不是新的实验结果。

来源与核验记录见 `report_provenance.json`。文献核验仅确认 arXiv:2601.12980v2 的题录、摘要与引言，不声称全文精读；未核验的期刊题录不用于正式参考文献。
