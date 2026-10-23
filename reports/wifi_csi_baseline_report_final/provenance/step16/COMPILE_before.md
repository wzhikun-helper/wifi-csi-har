# 最终排版修订版：编译与验收

> 公开脱敏版本（第17.1步）：原始文档已在仓库外私下留存，原件与本公开版的 SHA256 含义及对应关系见本报告目录的 `provenance/public_redaction.json`。本版本不与历史原件字节一致。
> 以下保留历史阶段说明与相对引用，不是清理后项目的当前运行入口；工具名通过 PATH 查找，示例未因脱敏而重新执行。当前构建请遵循报告目录的 `COMPILE.md`。涉及 `Set-Location` 的相对路径从项目根目录解释。

基于 `../wifi_csi_baseline_report_academic/main.tex` 小修，不重写实验内容。上一版交付 PDF `../wifi_csi_baseline_report_academic.pdf` 和源码保留，新版为 `../wifi_csi_baseline_report_final.pdf`。

PowerShell 必须先进入本目录：

```powershell
Set-Location -LiteralPath './reports/wifi_csi_baseline_report_final'
$reportPython = 'python'
$reportXeLaTeX = 'xelatex'
& $reportPython -B prepare_curve_layout.py
& $reportXeLaTeX -disable-installer -halt-on-error -interaction=nonstopmode -output-directory=build main.tex
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' }
& $reportXeLaTeX -disable-installer -halt-on-error -interaction=nonstopmode -output-directory=build main.tex
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' }
& 'pdftoppm' -r 130 -png build/main.pdf qa/final
& $reportPython -B verify_layout.py
```

所有页面均需实际查看；自动验证不能替代人工视觉检查。不要自动覆盖任何旧 PDF，使用独立文件名保存已经验收的新 PDF。

`prepare_curve_layout.py` 读取两张冻结曲线 PDF，只在 `assets/` 创建排版副本：各移除1条长标题和2条底部说明的文字绘制指令。全部非文字绘制指令保持一致，26个Epoch、图例、坐标轴和Epoch16标记保留；不重算结果、不调用训练或评价脚本。移除文字及源图/副本SHA256在 `curve_layout_provenance.json`。PDF仍以矢量形式嵌入。其余两张正式图表继续直接引用冻结源PDF。

`report_data.tex` 和 `architecture.tex` 与上一版逐字节一致；原始冻结来源及完整SHA256继续保留在 `../wifi_csi_baseline_report/report_provenance.json`，上一版工程记录在 `../wifi_csi_baseline_report_academic/academic_provenance.json`，本次保护快照为 `before_sha256.json`。本次修改记录见 `REVISION_LOG.md`。

本次编译未安装软件，保持字号、页边距、页眉页脚、表格样式。旧Fontconfig路径/外部编译日志权限提示及小幅数学字体尺寸替代仍存在，不影响已核验的嵌入字体和页面显示。曲线副本工具存在 pypdf 当前可用接口的弃用提示，输出已重读核验；不为此升级依赖。

第一次编译曾误指向上一版目录，仅刷新上一版 `build/main.log` 和 `build/main.pdf`；上一版交付PDF、源码、provenance和全部冻结实验文件均未变化。`verify_layout.py` 对这两个已披露中间产物单独记录，对其他既有文件严格校验，不把它们冒充为全部未变。
