# 阴阳师 · 对弈竞猜数据台 v3.3

面向固定“阵容详情”截图的数据采集工具：红/蓝双方截图 → ROI OCR → 式神名称约束 → 八维面板匹配 `duel-panels.json` 反推御魂 → 人工核验 → 密码保护保存 → 赛后补录真实胜负 → CSV/JSONL 导出。

## 主要变化

- 式神与御魂均为动态下拉列表，启动时从 `data/duel-panels.json` 自动读取；替换 JSON 后重启应用即可更新。
- 御魂不是写死：仅在当前式神的历史记录中对八维面板进行加权匹配，显示 Top 候选、score、margin 和证据等级。
- OCR 使用固定 ROI、多种图像增强；历史面板只提供“建议修正”，不会静默覆盖截图识别值。
- 保存前要求 10 个式神逐个“已人工核验”。数据库同时保留自动识别/推断值和最终确认值，可用于后续计算识别准确率。
- 保存、补录结果、历史修改、删除均需要 `SAVE_PASSWORD`；密码不会写入 SQLite/CSV/JSONL/日志。
- SQLite 每次写操作前保留最近 20 个数据库备份；截图按 match_id 保存。

## Windows 安装

```powershell
cd D:\Pyproject\onmyoji_panel_recorder_v3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果 `.venv` 已经激活（PowerShell 前面显示 `(.venv)`），不要再次执行 `python -m venv .venv`。

## 配置密码

复制：

```text
.streamlit/secrets.example.toml
```

为：

```text
.streamlit/secrets.toml
```

修改：

```toml
SAVE_PASSWORD = "换成你的强密码"
```

`.streamlit/secrets.toml` 已在 `.gitignore`，不要提交到 GitHub。

## 启动

```powershell
python -m streamlit run app.py
```

## 更新 duel-panels.json

直接替换：

```text
data/duel-panels.json
```

然后重启 Streamlit。程序会重新生成式神/御魂下拉选项，并记录参考库 SHA-256 短版本号到每场比赛元数据中。已经人工确认的历史比赛不会因为参考库更新而自动改变。

## 数据

运行后自动创建：

```text
storage/
  matches.sqlite3
  matches.csv
  units.csv
  matches.jsonl
  screenshots/
  backups/
```

`matches.sqlite3` 是本地权威数据源。CSV/JSONL 是自动导出。

## Streamlit Community Cloud

将代码上传 GitHub，但不要上传真实 `secrets.toml`。部署后在 App Settings → Secrets 中配置：

```toml
SAVE_PASSWORD = "你的强密码"
```

注意：Community Cloud 的应用本地文件系统不应被当作长期数据库。如果要在线长期积累真实竞猜数据，下一步应把 `Store` 后端换成持久化云数据库（例如 PostgreSQL），UI/识别/匹配层不需要重写。

## V3.1 performance update

V3.1 separates OCR from Streamlit reruns. OCR runs only when **智能识别双方** is pressed for a new image pair. The result is stored in `st.session_state`, and the same image pair/reference version is also cached with `st.cache_data`.

The fixed-layout OCR pipeline now packs the five name ROIs and forty numeric ROIs on each side into contact sheets. In the normal path this reduces RapidOCR engine invocations from roughly 350 per match in V3 to 4 per match (2 per side). Only cells missed by the sheet pass use a one-call fallback.

After OCR, changing a shikigami or any of its eight stats only recomputes soul evidence for that single unit. Changing a soul manually, checking confirmation, entering a password, changing notes/date/winner, or navigating the page does not run OCR again.

Deployment runtime is pinned to Python 3.12 via `.python-version` for better OpenCV/ONNX compatibility on Streamlit Community Cloud.

## V3.2 performance architecture

V3.2 removes automatic per-cell OCR fallback. In the normal recognition path it runs RapidOCR exactly twice per match: once for the full red screenshot and once for the full blue screenshot. OCR boxes are mapped back to the fixed 5-column x 9-row layout by coordinates. Missing cells remain editable for human review instead of triggering dozens of extra OCR calls.

The recognition page now reports real stage progress (0-100%) and stores timing diagnostics for OCR model initialization, red-side OCR, blue-side OCR, coordinate parsing, soul matching, and total elapsed time. Normal Streamlit widget reruns do not invoke OCR again. Editing one unit only recalculates that unit's soul evidence.

## V3.3：自适应主面板定位

V3.3 不再假设整张截图具有固定分辨率或固定长宽比。识别前先利用「阵容详情」表格的纵横网格几何结构定位主面板，裁剪后统一映射到 1000×491 的内部坐标系，再执行 5 列 × 8 属性解析。

- 支持不同截图分辨率和长宽比；背景 UI 可以有不同宽度。
- 主面板定位失败时停止 OCR，不会在错误区域强行识别。
- 识别进度新增红/蓝双方主面板定位阶段。
- 性能诊断新增面板定位耗时。
- 「查看主面板定位结果」会显示检测框和定位置信度，便于人工核验。
- OCR 正常路径仍为每张截图一次，共两次，不恢复逐格 fallback。

已用项目内原始 1026×542 左右截图，以及 1536×706 新截图验证主面板几何定位。
