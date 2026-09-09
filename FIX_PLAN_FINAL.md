# MAD-LEO 终审修复方案（FIX_PLAN_FINAL）

范围：合并清单 45 项全部修复，**除外** GitHub URL / Figshare DOI（用户自行处理）。
问题编号沿用 `final-review-merged.md`（C1–C2, W1–W15, M1–M26）。

---

## 0. TV 实验输出影响矩阵（先回答核心问题）

| 数据改动 | TV 输出是否变 | 哪些表/图变 | 论文哪些数字变 | 已验证不变的部分 |
|---|---|---|---|---|
| W1 20 条坏 SLR 行加标记 | 基本不变 | `slr_distribution_summary.csv`（若统计加过滤，极端值字段变；中位不变） | 无 | **tier 754/194/186 不翻**（两窗口剔除坏行后仍有 371/444 个好观测，已实测）；**O-C 表不污染**（6 条带内坏行在计算窗外，实测 RMS 43/48 m 正常）；NP 精度 4–14 mm、2.6 mm、542/296/41 m 全不变 |
| W2 sub-Earth Starlink 行加标记 | 一处变 | `starlink_ephemeris_distribution.csv` 的 `radius_min_km`（ALL 行 5520→~6650；shell 43° 5520→正常值；中位不变） | 无（论文未引 radius_min） | 一致性实验不变（3 颗离轨星不在 sample25，已实测）；Fig 4d 不变（TLE 来源）；6,785/43M/107 h 不变 |
| W3 负 sigma 文档化 | 不变 | 无（只改 metadata.md） | 无 | 中位 4.2–14.1 mm 不变 |
| W11 state_fusion ω×r | **变** | `source_bias_calibration.csv` + fusion 两表重生成 | technique.tex:84 句：radial 68 m 不变；along 291 m 与 cross ~86 m 小幅移动（~5–7% 轴间混合，预计 ±20 m 内） | 结论（百米级 bias）方向不变 |
| W12 半径统一 6378.137→6371.0 | **变** | `starlink_tle_step_candidates.csv` 的 `median_altitude_km` 全列 +7.1 km | technique.tex:77："386 km"→"392 km"；"63% sub-480"需重算（阈值两侧 ~7 km 带内对象会移动，预期小变）；图 4d 本就是 6371 变体，不变 | 其余分布表/图已是 6371，零改动 |
| W9 Maxwell 参照 | 图变 | `sigma_calibration_curve.csv` **已有该列，表不变**；图 fig:consistency(b) 重画 | technique.tex:84："41%/82% vs Gaussian 68%/95%" → "vs Maxwell-norm 名义 19.9%/73.9%（模型过覆盖，作为 consistency bound 保守）" | 95.5/96.9/84.2% 不变 |
| 其余全部（C1/C2/W4–W10/W13–W15/M 项） | 不变 | （2.4 可选新增行） | 纯文字/bib 修改 | — |

**结论：五个头条数字群（1,134/1,139/274、tier 754/194/186、92.6%、σ 0.376/0.302、542/296/41 m）经实测全部不动。** 论文中实际要改的数字只有 4 个：along/cross bias（W11 重生成后取新值）、386→392 km、63% sub-480（重算）、41%/82% 的参照系表述（W9）。

---

## 阶段 1 — 数据层（dataset/，触发 manifest 重生成）

**1.1 SLR 证据加 `qc_status` 列**（W1）
- 补丁脚本（不重跑管线）：逐 parquet 读入 → 加列 `qc_status`，判据复用 `physical_qc.SLR_RANGE_MIN/MAX_M`（5e5–6e6 m）+ 显式跨目标名单（S3A 文件内 `compassi6b` 5 行 → `cross_target`；其余坏行 → `range_implausible`；正常行 `ok`）→ zstd 原地重写，行数不变。
- 不删行（"标记不删行"契约）；负 sigma **不进 flag**（provider verbatim，只在文档披露）。
- 交叉目标普查结论（已实测）：全 11 文件中仅 S3A 有 5 条跨目标（全部在坏行集合内）；79 条 `sentinel` 短名视为 S3A 变体名放行（无法证伪，范围合法）。

**1.2 Starlink 星历加 `quality_flag` 列**（W2）
- `ephemeris_state.parquet` 与 `sample25`：加列 `quality_flag`（''/`below_surface`，|r|<6,371 km 的 5,861 行）。sample25 实测不含这 3 颗，列为空但仍加（schema 一致）。

**1.3 重生成受影响实验表**
- `run_experiments.py distribution-validation`：slr/orbit/分布统计增加 qc/flag 过滤 + `excluded_rows` 计数列 → `slr_distribution_summary.csv`（预期中位不动，仅极端值/说明变）。
- `run_experiments.py starlink-distributions`：过滤 `below_surface` → `starlink_ephemeris_distribution.csv`（`radius_min_km` 变为正常值；中位不变）。

**1.4 manifest 与闸门**
- 重生成 `dataset/manifest.json`（SHA 变、行数不变）；`process.py selfcheck` 必须 `ok:true, unexpected:0`。

**1.5 文档**（W15、M2/M4/M8 打包）
- `metadata.md`：SLR schema 补 `qc_status`、`source_zero_fields` 两行；负 sigma 异常段（各 target 计数：jason-2 8,879 等 + 过滤建议 `sigma_m>0`，可选 `<1 m`）；正尾说明（TOPEX 1,499 m、Jason-3 175 m）；`num_returns` 可空 float 说明、`window_length` 可空；`annotation_label_source` 补第二枚举值；ITRF 实现 cm–dm caveat 一句；ISO-8601 混合精度一句；Starlink `below_surface` 说明（3 颗离轨物体、0.014%）。
- `dataset/README.md`："SLR normal points" → "normal points + full-rate records (2.1M/24M)"；删除或定义 E1–E6/S1–5 编号（M18-ii）。

## 阶段 2 — 分析代码 + TV 表

**2.1 state_fusion ω×r 修复**（W11）
- `analyzers/state_fusion.py:38-52`：叉乘前套 `processors.physical_qc.ecef_velocity_to_inertial`。
- 重跑 `run_experiments.py state-fusion` → diff `source_bias_calibration.csv`：radial 应不动，along/cross 取新值 → 回填论文。

**2.2 半径统一**（W12；建议方向：统一到 6371.0）
- `generate_starlink_step_candidates.py:42` 改 import `experiment_params.EARTH_RADIUS_KM`。
- 重跑 `starlink-step-candidates` → `median_altitude_km` +7.1 km；重算 sub-480 占比 → 回填论文（386→392 km）。
- `methods.tex:58` 句子拆成两句：SGP4/开普勒换算用 WGS-84（μ、6378.137）；**发布的高度列一律 a−6371.0 km（球形平均半径）**；koziai 内 WGS72 6378.135 为刻意共模设计（注一句）。

**2.3 σ 校准图 Maxwell 参照**（W9）
- `scripts/experiments/make_tv_figures.py:642` 附近：参照线改用/并列 `nominal_coverage_maxwell_norm` 列；重生成 `tv_consistency.pdf` → 回嵌。

**2.4（建议）表格增强支撑 C2/M17**
- `tv_hardening` 输出 `slope_estimation_comparison.csv` 增加分箱 OLS 行（<20 / 20–100 / 100–1000 / >1k m）+ bootstrap 95% CI（OLS/Deming），已有行数值不动 → C2 重写与 M17 直接有表可引。

**2.5（可选）bracket 带宽敏感性**（W13）
- `event_response` 以 6/12/24 h 三档重算（读 data/ raw，耗时较长）。**默认不做**，以文字依据句替代；若你要求做，加 `window_sensitivity` 同款新表。

**2.6 代码卫生**（M19/M20）
- `event_response.py:745` rotating_frame 一致性断言；`stable_windows.py`/`event_response.py`/step_candidates 改 import 共享常量；重跑受影响实验确认输出**逐字节不变**（纯重构验证）。

**2.7 全量回归**
- 17 个实验命令全跑一遍，diff `experiments/validation|starlink`：**只允许** 1.3/2.1/2.2/2.4 列出的表变，其余必须零 diff。

## 阶段 3 — 论文（arxiv/MAD_LEO_paper/）

**3.1 sample.bib**（C1、M22）
- 补 3 条真实文献（`shorten2024particle`、`montilla2022manoeuvre`、`guo2025spacetrack`——执行时查文献核实；若与现有未用条目 `lemmens2014maneuver`/`kelecy2007detection` 主张重合，优先改引已有键）；剪除未用条目（执行时以编译器 unused 列表为准）。

**3.2 technique.tex**（逐段）：C2 斜率机制重写（引 2.4 新行：>1 km 尾部 orbit≈0.6–0.8×TLE；衰减对 pooled 拟合可忽略，引表内 0.5899；衰减表述只留亚噪声底区间）；W4 O-C 尾部（jason-1 −32 km/10 观测系统性偏移 + cryo-2 1.13 km + TOPEX 1.04 km，给出时标问题定性）；W6 0.83→0.76（或声明 |both|>1 m 过滤）；W7 0.14→"Pearson 0.19"；W8 高度下限/量级重写（CryoSat-2 ~725 km、实测 −1.97 m/−0.80 m）；W9 Maxwell 句；W10 Flohrer 改"水平一致、增长率为本集特有"；W11 新 bias 数字；W12 392 km + 新占比；W13 12 h 依据两句（TLE 节奏 ~1–3 条/天 + Flohrer 引年龄上限）+ 选择偏倚一句（子样本偏年轻 TLE，残差中位数偏乐观）；M5 denominators 表补 "≥2 样本/带（≥1 周期均值）"；M9 Brouwer→Kozai；M10 "numerically close"或补两比例检验；M11 软化 tier 结论；M12 Kozai 尾部（5.6%>1 m，max 15.8 m，归因 B*/偏心率）；M13 "robust between one and two days; at 0.5 d collapses to 80 pairs"；M14 "of the same order as"+年龄差注；M15 "median quiet-pair rate ≈0.28 km/8 h (~15×)"；M16 >48 h 非单调小句；M21 Algorithm 1 两注（带边部分窗平均、开普勒 vs 交点周期 <1%）；M23/M24 补 `\ref{tab:denominators)}`、`\eqref{eq:tle-response}`、`\eqref{eq:tier-ks}`。

**3.3 其余文件**
- `usageNotes.tex`：W5 "3.7 cm/s"→"roughly 1 cm/s（Δv=(n/2)·Δa 逐事件中位 1.03 cm/s）"；M18-i 与 abstract 对齐（提大机动端系统差）。
- `methods.tex`：W12 半径句（2.2）；W13 依据；M6 7 天传播上限一句（影响 2 事件）；M3 Methods 补速度地固系一句 + 新增 Limitations 段（地固速度 ω×r、单 benchmark、11 星范围、operational 无标签、ITRF 实现）。
- `data.tex`：W15 字段表补 `qc_status`/`source_zero_fields`；A#10 补 Sentinel-3 `quality` 列。
- `abstract.tex`：M18-i。
- 清理 `images/c_q2_maneuver_anatomy_a/b.pdf`（M25）。

**3.4 编译**：pdflatex×2 + bibtex；`main.log` 零 undefined citation/reference、零 multiply-defined。

## 阶段 4 — 终验闸门

1. `selfcheck` ok + manifest 全匹配（README 片段）。
2. 实验表 diff 白名单核对（2.7）。
3. 旧值 grep 归零清单：`0.83`、`3.7 cm`、`0.14`、`780 km`、`386 km`、`29 m`（spread 句）、`Brouwer mean`、`Gaussian nominal`、`millimetres to centimetres`。
4. 新值论文↔表逐一对账（392 km、新 sub-480 占比、新 along/cross bias、0.76、0.19、Maxwell 数）。
5. 6 张 tv_*.pdf 目检（重点 fig:consistency b/c 两 panel）。

## 执行顺序与依赖

阶段1（1.1→1.2→1.3→1.4，1.5 并行）→ 阶段2（2.1/2.2/2.3/2.4 可并行，2.6 先行以冻结基线，2.7 收口）→ 阶段3（依赖 2 的最终数字）→ 阶段4。

## 已定默认（可推翻）

- 半径统一方向：**6371.0**（P4 单源已声明；只动 1 脚本+论文 2 数字）。备选 6378.137 需动 4 个生成器+分布表+图，不推荐。
- 负 sigma：**只文档不 flag**（verbatim 原则）。
- W13 带宽敏感性：**默认文字依据，不做新实验**。
- 20 坏行/sub-Earth 行：**标记保留，不删行**。
