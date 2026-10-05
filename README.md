# 阴阳师 · 对弈竞猜数据台 v3.4

面向《阴阳师》对弈竞猜“阵容详情”截图的数据采集与辅助识别工具：

**红/蓝双方截图 → 自适应定位阵容详情主面板 → 整图 OCR → 式神名称匹配 → 八维面板与历史御魂图片联合推断 → 人工核验 → PostgreSQL 持久化 → 赛后补录真实胜负 → CSV/JSONL 导出。**

## 当前主要功能

- 式神与御魂下拉列表均从 `data/duel-panels.json` 动态生成，不在代码中写死。
- 每场正常识别路径固定执行 **2 次 RapidOCR**：红方整图一次、蓝方整图一次。
- 自动定位不同分辨率、不同长宽比截图中的“阵容详情”主面板，并归一化到 `1000 × 491` 坐标系。
- 解析双方共 10 个式神及每个式神的 8 项面板属性：
  `ATK / HP / DEF / SPD / CRI / CRID / EFH / EFR`。
- 式神名称采用 **精确匹配优先**。合法完整名称不会因为存在包含关系而被更长名称覆盖。
- 对非精确 OCR 名称，可结合名称相似度、名称包含关系及八维属性进行辅助消歧。
- 御魂推断同时保留：
  - `duel-panels.json` 八维属性证据；
  - 历史人工确认的御魂图标图片证据；
  - 最终人工确认结果。
- 每个单位均需人工核验后才能保存比赛。
- 自动识别值和人工确认值分别保存，便于后续评估识别准确率和继续改进模型。
- 相同红/蓝截图使用稳定的 `img-<hash>` match ID，避免重复点击保存生成重复比赛。
- 保存、补录结果、历史修改、删除和导出均受 `SAVE_PASSWORD` 保护。

## 式神名称识别

名称识别遵循以下原则：

1. **Exact match 优先**  
   如果 OCR 结果本身就是 `duel-panels.json` 中的合法式神名称，则直接保留。

   例如：

   ```text
   酒吞童子 → 酒吞童子
   鬼王酒吞童子 → 鬼王酒吞童子
   一目连 → 一目连
   苍风一目连 → 苍风一目连
   ```

2. **只有非精确名称才进行模糊消歧**  
   OCR 出现缺字、错字时，再结合 fuzzy matching、名称包含关系和八维面板属性选择候选。

3. **合法短名称不会被长名称静默覆盖**  
   例如 OCR 明确得到 `酒吞童子` 时，不会仅因为数据库中还有 `鬼王酒吞童子` 就自动改名。

4. 如果名称文本与面板属性出现明显冲突，系统可以提示人工复核，而最终确认仍由用户决定。

## 御魂推断

御魂不写死在代码中。

### 属性证据

系统首先在当前式神对应的 `duel-panels.json` 历史记录中比较八维面板，计算候选御魂的：

- score
- margin
- evidence level
- Top 候选
- 可能的 OCR 属性异常建议

修改式神或任意属性时，只重新计算当前单位，不重新运行 OCR。

### 图片证据

每场比赛在 10 个单位全部人工核验后，会从归一化主面板中裁剪御魂图标区域。

每个样本保存：

- match ID
- RED / BLUE
- slot
- 最终确认式神
- 最终确认御魂
- 自动属性推断御魂
- 御魂图标图片
- crop 坐标
- 主面板定位置信度

御魂图片存入 Supabase private Storage bucket `soul-icons`，标签和路径记录在 PostgreSQL `soul_samples` 表中。

后续新比赛可以利用历史人工确认图片进行 Top-K 图像相似度匹配，并与属性证据共同辅助御魂判断。

### 人工确认是最终结果

图片预测和属性预测都只是辅助证据。人工选择的 `soul` 才是最终保存结果，历史人工确认数据不会因为之后算法或参考库更新而被静默修改。

## OCR 与主面板定位

程序不假设整张截图具有固定分辨率或固定长宽比。

识别流程：

```text
原始截图
  ↓
检测“阵容详情”表格网格
  ↓
定位主面板
  ↓
归一化为 1000 × 491
  ↓
红方一次整图 OCR + 蓝方一次整图 OCR
  ↓
按固定内部坐标映射 5 个式神 ×（名称 + 8 属性）
```

如果无法可靠定位主面板，识别会直接停止，而不是在错误区域继续 OCR。

界面可以查看：

- 主面板 bbox
- 定位置信度
- 定位预览
- OCR / 定位 / 解析耗时
- 缺失属性数量

缺失字段由人工补充，不执行逐格 OCR fallback。

## 数据架构

### PostgreSQL

Streamlit Cloud 生产环境使用 PostgreSQL 作为结构化数据的权威数据源。

主要表包括：

```text
matches
units
soul_samples
```

其中：

- `matches`：比赛级元数据；
- `units`：每场 10 个式神的 OCR、属性、御魂推断和人工确认结果；
- `soul_samples`：御魂图片样本索引及人工确认标签。

### Supabase Storage

private bucket：

```text
soul-icons/
  <match_id>/
    RED_1.webp
    RED_2.webp
    ...
    BLUE_5.webp
```

每场成功采集最多 10 个御魂图标样本。

### 本地 SQLite fallback

如果没有配置 `DATABASE_URL`，程序仍可使用本地 SQLite fallback：

```text
storage/
  matches.sqlite3
  matches.csv
  units.csv
  matches.jsonl
  screenshots/
  backups/
```

SQLite 模式下写操作前会保留最近的数据库备份。

**注意：Streamlit Community Cloud 的本地文件系统不是长期持久化存储，因此线上生产环境应使用 PostgreSQL/Supabase。**

## Windows 安装

```powershell
cd D:\Pyproject\onmyoji_panel_recorder_v3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果 `.venv` 已经激活（PowerShell 前面显示 `(.venv)`），不要再次运行 `python -m venv .venv`。

## Secrets 配置

本地开发可以复制：

```text
.streamlit/secrets.example.toml
```

为：

```text
.streamlit/secrets.toml
```

生产环境在 Streamlit Community Cloud 的 **App Settings → Secrets** 中配置。

示例：

```toml
SAVE_PASSWORD = "replace-with-a-strong-password"

DATABASE_URL = "postgresql://USER:PASSWORD@HOST:PORT/postgres"

SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"
SUPABASE_SECRET_KEY = "YOUR_SERVER_SIDE_SECRET_KEY"
```

`SUPABASE_URL` 应为项目根 URL，不要附加 `/rest/v1` 等路径。

不要把真实密码、数据库连接串或 Supabase secret 提交到 GitHub。

`.streamlit/secrets.toml` 应包含在 `.gitignore` 中。

## 启动

```powershell
python -m streamlit run app.py
```

## 更新 duel-panels.json

直接替换：

```text
data/duel-panels.json
```

然后重启 Streamlit。

程序会：

- 重新生成式神下拉列表；
- 重新生成御魂下拉列表；
- 使用新的参考数据进行后续属性匹配；
- 计算并记录参考库 SHA-256 短版本号。

已经人工确认并保存的历史比赛不会因为参考库更新而自动改变。

## 保存流程

一场新比赛的大致流程：

```text
上传 RED / BLUE 截图
        ↓
智能识别双方
        ↓
定位两个主面板
        ↓
2 次整图 OCR
        ↓
式神名称 + 八维属性
        ↓
属性御魂证据 + 历史图片证据
        ↓
人工修改 / 核验 10 个单位
        ↓
保存比赛
        ↓
PostgreSQL: matches + units
        ↓
Supabase Storage: 10 个 soul icon crops
        ↓
PostgreSQL: soul_samples 标签索引
```

御魂图片上传失败不会回滚已经成功保存的比赛结构化数据。

## 防止重复保存

红方和蓝方原始截图字节会生成稳定的 SHA-256 图像对标识：

```text
img-<image_pair_hash>
```

因此，同一对截图重复点击保存不会不断创建新的 match ID。

## 历史管理

支持：

- 补录比赛结果；
- 修改日期、时间、备注和结果；
- 修改式神、属性及御魂；
- 保持人工核验状态；
- 删除整场比赛；
- 自动同步已有 `soul_samples` 的人工确认标签。

## 导出

应用自动生成：

```text
storage/matches.csv
storage/units.csv
storage/matches.jsonl
```

导出操作需要管理密码。

导出数据不包含 `SAVE_PASSWORD`、Supabase secret 或数据库密码。

## Streamlit Community Cloud

将代码推送到 GitHub 后，Streamlit Community Cloud 会自动拉取新版本。

推荐使用 Python 3.12，并通过 `.python-version` 固定运行时版本，以提高 OpenCV / ONNX Runtime 兼容性。

正常路径始终保持：

```text
1 场比赛
= 2 张截图
= 2 次 RapidOCR inference
```

Streamlit widget rerun 不会重新执行 OCR。修改单个式神或属性只重新计算该单位的匹配证据。

## 版本演进

### V3.1

- 将 OCR 与 Streamlit widget rerun 分离。
- 避免编辑、密码输入等操作反复触发 OCR。

### V3.2

- 移除自动逐格 OCR fallback。
- 正常路径固定为每场 2 次整图 RapidOCR。
- 增加真实进度条和各阶段耗时诊断。

### V3.3 / V3.3.1

- 不再依赖整张截图的固定分辨率和长宽比。
- 根据表格纵横网格定位“阵容详情”主面板。
- 统一映射到 `1000 × 491` 内部坐标。
- 修复不同 OpenCV build 下 `HoughLinesP` 输出 shape 不一致的问题。
- 异常/空 Hough 输出会干净地返回定位失败，而不是导致应用崩溃。

### V3.4

- PostgreSQL/Supabase 成为线上持久化数据架构。
- 使用稳定 image-pair match ID，防止重复保存。
- 保存人工确认的御魂 icon 样本到 private `soul-icons`。
- 引入历史御魂图片 Top-K 相似度证据。
- 御魂推断升级为属性证据 + 图片证据的联合辅助判断。
- 改进式神名称识别：exact match 优先，并对长短名称包含关系进行安全消歧。
- 保留自动推断和最终人工确认之间的明确边界。

## 安全

- 所有写入、修改、删除和导出操作均受密码保护。
- 密码和 Supabase secret 只通过 Streamlit Secrets / 环境变量读取。
- 不将 secret 写入 PostgreSQL、SQLite、CSV、JSONL 或日志。
- 不要提交 `.streamlit/secrets.toml`。
- Supabase `soul-icons` bucket 应保持 private。

---

本项目的目标首先是建立**可靠、可人工校验、可持续积累的对弈竞猜数据集**。自动识别和匹配结果属于辅助证据；最终人工确认数据始终作为后续分析和模型改进的基准。

## 数据来源与鸣谢

本项目使用的 `data/duel-panels.json` 参考数据来源于开源项目：

- [sdsdsssssdsd/onmyoji-guessing-sim](https://github.com/sdsdsssssdsd/onmyoji-guessing-sim)

感谢原项目作者整理和公开《阴阳师》对弈竞猜相关的式神、御魂与面板参考数据。

本项目在该参考数据基础上实现截图主面板定位、OCR、式神名称识别、属性匹配、历史御魂图片匹配、人工核验以及数据持久化等功能。

`duel-panels.json` 的原始数据版权、许可和使用条款以原项目仓库中的说明为准。
