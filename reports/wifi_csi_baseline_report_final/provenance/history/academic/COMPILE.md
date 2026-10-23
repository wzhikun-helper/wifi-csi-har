# 学术排版版：编译、对照与验收

> 公开脱敏版本（第17.1步）：原始文档已在仓库外私下留存，原件与本公开版的 SHA256 含义及对应关系见本报告目录的 `provenance/public_redaction.json`。本版本不与历史原件字节一致。
> 以下保留历史阶段说明与相对引用，不是清理后项目的当前运行入口；工具名通过 PATH 查找，示例未因脱敏而重新执行。当前构建请遵循报告目录的 `COMPILE.md`。涉及 `Set-Location` 的相对路径从项目根目录解释。

本目录在原有 LaTeX 上修改排版；原目录 `../wifi_csi_baseline_report/` 与原 PDF `../wifi_csi_baseline_report.pdf` 原样保留。新版交付文件为 `../wifi_csi_baseline_report_academic.pdf`。

## 编译

使用原项目已有 XeLaTeX，无需安装软件。PowerShell 在本目录执行：

```powershell
$texCompiler = 'xelatex'
New-Item -ItemType Directory -Path build -Force | Out-Null
& $texCompiler -disable-installer -halt-on-error -interaction=nonstopmode -output-directory=build main.tex
if ($LASTEXITCODE -ne 0) { throw 'First compilation failed' }
& $texCompiler -disable-installer -halt-on-error -interaction=nonstopmode -output-directory=build main.tex
if ($LASTEXITCODE -ne 0) { throw 'Second compilation failed' }
```

`report_data.tex` 是原版数据片段的逐字节副本，不运行原版生成器，以免更新原版 provenance。图片引用冻结目录 `../../figures/formal_seed42/` 的四张 PDF。图内重复标题、样本量脚注通过 `trim` 与 `clip` 裁去，完整坐标轴、26个数据点、图例、矩阵单元格、分类柱形均保留；实验条件统一写入图注及正文，不更改源图。

`architecture.tex` 为按 `../../src/model.py` 绘制的原生 TikZ 张量示意，不调用模型或读取 MAT。独立导出：

```powershell
& $texCompiler -disable-installer -halt-on-error -interaction=nonstopmode -output-directory=build architecture_preview.tex
Copy-Item -LiteralPath build/architecture_preview.pdf -Destination architecture.pdf
& 'pdftoppm' -r 300 -singlefile -png architecture.pdf architecture
& 'pdftocairo' -svg architecture.pdf architecture.svg
```

## 渲染与核验

```powershell
& 'pdftoppm' -r 130 -png build/main.pdf qa/accepted
& 'python' -B verify_layout.py
```

必须人工查看所有 `qa/accepted-*.png`，自动检查不能替代视觉检查。记录见 `LAYOUT_CHANGES.md` 与 `layout_validation.json`。源文件没有 `newpage`、`clearpage`、固定图片高度；`FloatBarrier` 约束图表不越过所属章节，`Needspace` 防止标题、公式与关联内容孤立，表格以不可拆的 minipage 保持完整。

原版及新版正式 PDF 均不应被编译命令自动覆盖。人工验收后如需另存 build/main.pdf，应先确认目标不存在，使用独立文件名。本次交付使用独占创建写入目标。

## 字体与本机非排版提示

A4、单栏、正文小四（保持原字号），边距左右23mm、上下22mm，宋体正文、黑体标题、正文西文 Times New Roman、代码 Consolas。正文行距因子1.18，段间2pt；表格、图注及图表间距集中定义。

MiKTeX 本机仍有旧 Fontconfig 路径提示、外部日志目录写权限提示及小于0.411pt的数学扩展字体尺寸替代；这些不影响编译出的嵌入字体及已检查页面。本次未修改系统配置或安装宏包，编译显式禁用自动安装。
