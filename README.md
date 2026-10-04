# 阴阳师 · 对弈竞猜数据台 v3

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
