# 红色字符识别 — 交接文档 (HANDOFF)

> 面向接手的 agent / 协作者。读完即可复现现状并继续推进。最后更新：2026-06-19（**最终版**）。
> **完整自洽技术报告见 `方法报告_98.96.md`**（无需历史背景即可复现）。当前最佳 = 平台 **98.96%**（已超目标 98.86），已定稿。
> 最终提交：`red_char/outputs/submission_ghl.csv`。

## 1. 任务

机器学习课程大作业问题三：给定 200×60 RGB 验证码图，每张恰好 5 个字符（字符集 `0-9A-Z`，36 类），每位有红/非红属性。输出**按顺序排列的全部红色字符**组成的字符串（长 0~5），生成 Kaggle `submission.csv`。评测指标 = 整串 **exact-match** 准确率。

- exact ≈ (每位联合 char&color 正确率)^5 → **字符识别是瓶颈**，颜色判定基本已解决。

## 2. 当前成绩

| 方案 | 验证集 exact | 平台分 |
|---|---|---|
| v2×4 集成（起点） | 0.9856 | 0.9700 |
| v2hi 高分辨率 ×3 | 0.9884 | 0.9794 |
| 伪标签 self-training ×3 | 0.9900 | 0.9810 |
| 12 主模型 + TTA + glyph reranker(gf1) | 0.9928 | 0.9840 |
| + glyph hires+GAP(gfg) | 0.9936 | 0.9856 |
| + glyph 红度隔离(gfr) | 0.9940 | 0.9860 |
| glyph 红度+红线增强(gfrl) | 0.9944 | 0.9872 |
| + 红色阈值/集成扩容微调 | 0.9944 | 0.9876 |
| **★ ghl: red2 + 重红线/浅红/遮挡增强 + 100ep 长训** | **0.9964** | **0.9896** |

**★ 最终提交文件**：`red_char/outputs/submission_ghl.csv`（平台 **0.9896**，已超目标 98.86）。配置 = 15 主模型(9 v2hi + 3 ps + 3 r2ps) + **ghl glyph ×3** + 选择性重排(primary_margin≤0.50, glyph_margin≥0.05, red_thr=0.20) + 水平 TTA。

**制胜一招(98.76→98.96)**：glyph 局部识别器用 **red2 强度鲁棒红度表示 + 重红线遮挡增强(p0.7,1-5条红线) + 浅红渐变增强(p0.4) + cutout(p0.3) + 100 epoch 长训**。在 held-out 6231 红字上错误 19→8(V 0.995→1.0, I 0.976→0.994)。**"长训"是关键**——同样增强只训 40ep 无效;补足 epoch 后才跳分。完整细节见 `方法报告_98.96.md` 第 4 节。

**教训**：2,500 val 在 ≥0.994 精度只到 ±1 样本噪声,**阈值/小集成微调不可信**(gfb val+1 平台-；阈值 0.65 平台-)。真增益必须有**大样本支撑**(held-out 6231 红字逐类、或端到端 val ≥+5 样本),ghl 正是如此才转化到平台。把"重线+长训"用到**主模型**(phl, char 0.990→0.994)虽真变强,但被 glyph 重排吸收,端到端平台不变(仍 98.96)。
备选：`submission_pseudo6.csv`（6 个伪标签学生集成，val 同为 0.99，更稳健，与 pseudo3 差 24/5000，未提交）。

## 3. 环境（重要）

- **必须用 conda 环境 `red_char`**：`/home/duxuanzheng/.conda/envs/red_char/bin/python`（torch 2.6.0+cu124，CUDA 可用）。`base` 是 CPU-only torch，**不要用**。
- 机器：8× A100-80GB，但是**共享 144 核机器**，常被其他用户压到 load ~400。
  - **启动训练务必限线程**，否则单进程抢 ~21 核、并发就把数据缓存饿死（曾卡 20 分钟 0% GPU）：
    `OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6`
  - 后台跑训练用 `setsid PY ... </dev/null & ` —— 直接 `nohup &` 的子进程会被父任务清理掉。
  - 跑前用 `nvidia-smi` 挑 4MiB 空闲的卡，`CUDA_VISIBLE_DEVICES=<id>`。

## 4. 数据（有坑）

- 完整数据来自父目录 **`大作业/verification-red-code.zip`（218MB，中央目录完好）**：50000 train + `train/labels.csv` + 5000 test + `submission_sample.csv`。已解压到 `red-char-recognition/红色字符识别/`（`config.py` 期望的位置）。
- ⚠️ `red-char-recognition/verification-red-code.zip`（61MB）是**截断损坏副本**（无中央目录、缺 labels.csv），**别用**。
- `labels.csv` 格式：`filename,color,all_label`，如 `00000.png,ruuur,DPVKD`（r=红 u=非红）。
- 划分：`config.SEED=42` 固定，95/5 → train 47500 / val 2500。验证集划分由 `deterministic_split_indices` 独立固定，**改训练 seed 不会动验证集**（保证集成公平 + 诊断可比）。

## 5. 方法演进与结论（含负面结果，勿重复踩坑）

1. **基线 v2**（残差+SE+CoordConv，6M）：单模型 val 0.979，char_acc 平台 0.988。
2. **多 seed 集成**：+0.68%。可靠但边际递减。
3. **v2hi（关键突破，model.py `RedCharNetV2Hi`）**：把降采样从 4 次减到 3 次（60×200→7×25 而非 3×12），每个字符保留 ~5px 细节（v2 只 2-3px），再用 1×1 卷积压通道控制 FC 规模。**单个 v2hi ≈ 整个 v2×4 集成**。攻 O/0、G/C 这类细节歧义有效。
4. **伪标签 self-training（train_pseudo.py，平台主力）**：6-seed v2hi 教师给 5000 test 图打标签，保留 `min-over-5pos char_conf≥0.92 且 color_conf≥0.90` 的（注意：6 模型平均 softmax 峰值只到 ~0.97，阈值用 0.99x 会保留 0 张，必须 ~0.92），保留约 3950 张并入训练。**向测试分布自适应**，平台 0.9794→0.9810。验证集纯净（只用真标签、不增强）保证指标诚实。
5. **红色阈值 τ=0.40（predict.py `--red-threshold`）**：颜色错误系统性单向 r→u（漏红），τ=0.40 在 val 上是一整段平台，无害微正。
6. ❌ **v3 预训练 ResNet18**：char_acc 0.968 ≪ 0.988，域差距 + 丢 CoordConv 位置信息。**别用预训练主干**。
7. ❌ **强增强（仿射加大+透视+高斯模糊，40ep）**：训练验证一起降（val 0.9804），char 差距不变。结论：那 0.77% train-val 差距**不是可约过拟合，而是不可约字形歧义**（G/C@0.97、V/I@0.95 这类）。且**模糊是错的**——它抹掉了我们要区分的细节。
8. ❌ **第二轮伪标签迭代**：不复利（教师错误被继承），仍 0.99。
9. **高分辨率局部 glyph reranker（关键，平台主力之一）**：
   - 从每个名义槽中心提取重叠 `60×64` 原图裁切；主模型(12 ckpt: ps/r2ps/v2hi)+水平 `0/-4/+4px` TTA 给候选；局部模型只在主模型 Top-3 候选内**选择性重排**（仅当主 Top-2 margin 小且 glyph margin 够大时覆盖；颜色由主模型定）。
   - glyph 架构迭代（每步端到端 val / 平台均升）：
     - `gf1` 原始(2 池化→15×16, Flatten, rgb)：val 0.9928 / 平台 0.9840
     - `gfg` **hires(1 池化→30×32) + GAP 小头**（flat 大 FC 会过拟合，GAP 才对）：0.9936 / 0.9856
     - `gfr` **红度输入 `input_mode=red`**（附加 `relu(R−max(G,B))` 红度图+暗度图，隔离非红线）：0.9940 / 0.9860
     - `gfrl` 红度 + 红线增强 `--red-line-aug 0.5`：0.9944 / 0.9872
     - **`ghl` ★最终制胜：`--input-mode red2`(每块归一化红度,救浅红) + 重红线增强 `--red-line-aug 0.7`(1~5条) + 浅红渐变 `--faint-aug 0.4` + `--cutout 0.3` + `--epochs 100`(长训是关键)**：held-out 6231 红字错误 19→8(V 0.995→1.0, I 0.976→0.994),端到端 val 0.9944→**0.9964 / 平台 0.9896**。完整细节见 `方法报告_98.96.md` §4。
   - 最终阈值：`--primary-margin-max 0.40 --glyph-margin-min 0.05 --red-threshold 0.20`，glyph 用 `gfrl1/gfrl2` 两 seed。
10. **K 折 OOF 框架（已建，用于无偏调参/诊断）**：`kfold.py` 5 折 + `train.py/train_glyph.py --fold` + `oof_predict.py`（拼全量 50000 无偏预测）+ `tune_oof.py`（向量化网格搜阈值）。因 2500 val 已饱和，后续任何调参/改进**必须**用 OOF 这种大样本判断。
11. ❌ **本轮新增负面结果（勿重复）**：
    - **去线 U-Net**（`synth.py`/`denoise.py`/`train_denoise.py`/`make_denoised.py`）：合成域差距→真图上有伪影、伤字符与颜色。直接去线后识别**更差**(char 0.987/color 0.997)；6 通道(原图+去线)融合**持平**。结论：合成训练的去线网无法突破字符天花板。
    - **glyph 宽裁切**(104px，给上下文判线)：邻字干扰 > 上下文收益，红字 acc 暴跌 0.978。需中心坐标通道才可能用,未做。
    - **红线增强用到主模型**：被强集成稀释，无增益。
    - **I/1 竖笔画群过采样(gfb)**：val 涨 1 样本(0.9944→0.9948)但**平台反降 0.9872→0.9864**——典型 val 噪声不转化。
    - 旧负面仍成立：v3 预训练、强增强/模糊、第二轮伪标签、glyph 全字符、flat 大 FC hires、混淆组专用头/1-NN。

### 诊断（已探明）
- 干净训练集 char 0.99987 / 验证集 char 0.99224，差 ~0.77%；强增强使两者同降证实**此差距是不可约字形歧义，非过拟合**。
- 最终系统(gfrl)验证集 **14/2500 全为字符错、零颜色错**；集中在 S 群(S↔6/8/E/C)、竖笔画群(I/1/V/Z 互混)、G↔C。看图：约半数人可辨(线遮挡)、半数近极限。
- **平台已在 98.72 附近平台期**。要冲更高，最可能需**验证码生成器/确切字体**做零域差距合成训练（推测同学 99.9% 的来源）；纯模型侧的高把握招已用尽。

## 6. 代码 / 产物清单（目录 `red_char/`）

| 文件 | 作用 |
|---|---|
| `config.py` | 路径/超参唯一来源；`MODEL`、增强参数、`AUG_HEAVY` 等 |
| `dataset.py` | Dataset、编解码、确定性划分、`build_*` |
| `model.py` | `RedCharNet`(v1)/`RedCharNetV2`(v2)/`RedCharNetV2Hi`(v2hi, 也支持 `v2hi6` 6通道)/`RedCharNetV3`(v3) + `build_model` |
| `augment.py` | `TrainAugment`，保色几何增强 + 强增强(heavy) + **红线增强(`red_line_p`)**，严禁色相/饱和/通道/灰度 |
| `train.py` | 主模型训练；CLI `--model --epochs --seed --tag --lr --heavy-aug --denoised --concat-denoised --red-line-aug --fold --overfit-sanity` |
| `train_pseudo.py` | 伪标签 self-training；`--teacher <ckpts> --char-th --color-th --heavy-aug --fold` |
| `predict.py` | 集成推理出 submission；`--checkpoints --red-threshold --output` |
| `eval_ensemble.py` | 集成 val/train 上 exact/char/color；`--split val/train/both --denoised --concat-denoised` |
| `analyze_errors.py` / `char_confusion.py` / `dump_val_errors.py` | 误差/混淆诊断、导出错误图 |
| `glyph.py` | 局部字符数据集/模型(`GlyphNet`: `hires`/`head_mode=gap`/`input_mode=red`/`crop_width`/`boost_chars`) + 裁切/推理 |
| `train_glyph.py` | 训练 glyph；`--hires --head-mode gap --input-mode red2 --red-line-aug --faint-aug --cutout --crop-width --boost-chars --all-glyphs --fold` |
| `eval_reranker.py` / `predict_reranker.py` | 主模型+TTA+选择性重排：验证 / 生成 submission |
| `glyph_perclass.py` | glyph 逐类(I/1/S/G…)准确率，验证定向改进是否真有效 |
| `kfold.py` / `oof_predict.py` / `tune_oof.py` | K 折 OOF 框架（全量 50000 无偏调参/诊断） |
| `synth.py` / `denoise.py` / `train_denoise.py` / `make_denoised.py` | 去线 U-Net（**负面**，留档） |
| `evaluate.py` / `train_pair_glyph.py` / `eval_pair_reranker.py` / `eval_knn_reranker.py` | 旧/负面实验，留档 |

**Checkpoints**（`outputs/checkpoints/best_*.pt`，含 EMA 权重）：
- **★最终系统(98.96)成员**：主模型 15 个 = `best_v2hs1..9.pt` + `best_ps1/2/3.pt` + `best_r2ps1/2/3.pt`；glyph = **`best_ghl1/2/3.pt`**(red2+重增强+100ep)
- glyph 迭代(早→晚)：`best_gf1`(原始)→`best_gfg1-3`(hires+gap)→`best_gfr1/2`(红度)→`best_gfrl1-6`(红度+红线)→**`best_ghl1-6`(red2+重红线/浅红/cutout+长训★)**
- `best_phl1-6.pt`：主模型也用 ghl 配方(red-line 0.6 + 100ep)训的版本,char 0.990→0.994 真变强,但端到端被 glyph 重排吸收(平台仍 98.96),**未进最终提交**,留档
- 负面留档：`best_gfb*`(I/1过采样,平台反降)、`best_gfw*`(宽裁切)、`best_gfc*`(cutout-only)、`best_hps*`(强增强伪标签)、`best_v3s*`(预训练)、`best_dn*/cd*`(去线/6通道)、`best_gfh*`(flat-hires)、`best_v2hiff`类(端到端去噪前端)、各折 `best_f*s1/gff*`

## 7. 复现关键命令

```bash
cd red_char
PY=/home/duxuanzheng/.conda/envs/red_char/bin/python
CK=outputs/checkpoints

PRIM="$CK/best_ps1.pt $CK/best_ps2.pt $CK/best_ps3.pt $CK/best_r2ps1.pt $CK/best_r2ps2.pt $CK/best_r2ps3.pt $CK/best_v2hs1.pt $CK/best_v2hs2.pt $CK/best_v2hs3.pt $CK/best_v2hs4.pt $CK/best_v2hs5.pt $CK/best_v2hs6.pt $CK/best_v2hs7.pt $CK/best_v2hs8.pt $CK/best_v2hs9.pt"
GHL="$CK/best_ghl1.pt $CK/best_ghl2.pt $CK/best_ghl3.pt"

# ★ 评测最终系统(val，输出 selective exact≈0.9964)
CUDA_VISIBLE_DEVICES=0 $PY eval_reranker.py --x-tta --top-k 3 --checkpoints $PRIM --glyph-checkpoints $GHL

# ★ 生成最终提交（平台 0.9896）= submission_ghl.csv
CUDA_VISIBLE_DEVICES=0 $PY predict_reranker.py --x-tta --selective --top-k 3 \
  --primary-margin-max 0.50 --glyph-margin-min 0.05 --red-threshold 0.20 \
  --checkpoints $PRIM --glyph-checkpoints $GHL \
  --output outputs/submission_ghl.csv

# ★ 训练制胜 glyph（ghl：red2 + 重红线/浅红/遮挡增强 + 100ep 长训）
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 setsid $PY -u \
  train_glyph.py --hires --head-mode gap --input-mode red2 \
  --red-line-aug 0.7 --faint-aug 0.4 --cutout 0.3 --epochs 100 --seed 1 --tag _ghl1 \
  > outputs/logs/ghl1.log 2>&1 </dev/null &

# 训练一个 v2hi seed（限线程 + setsid 后台）
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 setsid $PY -u \
  train.py --model v2hi --epochs 40 --seed 1 --tag _v2hs1 \
  > outputs/logs/x.log 2>&1 </dev/null &

# 伪标签 self-training（教师=6 个 v2hi）
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 setsid $PY -u \
  train_pseudo.py --teacher $CK/best_v2hs1.pt ... $CK/best_v2hs6.pt \
  --model v2hi --epochs 40 --seed 11 --tag _ps1 --char-th 0.92 --color-th 0.90 \
  > outputs/logs/ps1.log 2>&1 </dev/null &
```

## 8. 最终状态与（若继续）下一步

**已定稿**：最终最佳 = **`submission_ghl.csv`，平台 0.9896**（已超目标 98.86）。完整自洽方法见 `方法报告_98.96.md`。

制胜路径回顾(平台)：97.00(v2集成) → 97.94(v2hi高分辨率) → 98.10(伪标签) → 98.76(glyph重排+红度+红线增强) → **98.96(ghl: red2 + 重红线/浅红/cutout 增强 + 100ep 长训)**。

若后续仍要推进（按优先级 / 全部合规）：
1. **凡改进必用大样本判断**：2500 val 在 ≥0.994 只到 ±1 样本噪声;用 held-out 6231 红字逐类 或 端到端 val ≥+5 样本(ghl 即如此才转化到平台)。阈值/小集成微调不可信(gfb、阈值0.65 均平台反降)。
2. **"重线+长训"配方**对 glyph 有效(ghl);用到主模型(phl)char 真升 0.990→0.994 但被 glyph 重排吸收,端到端不变。可探:更强/更深 glyph、glyph 伪标签、第二级候选仲裁器(OOF 标定)。
3. 已证无效(勿重复)：预训练大模型、强增强含模糊、合成去线 U-Net、端到端去噪前端、宽裁切、I/1过采样、合成数据混训、全量重训(被重排吸收)、阈值/TTA 微调。
4. **绝不**获取/使用隐藏 test 答案调模型(测试集泄露+学术不端);仅用 test 图像做伪标签/TTA(合规)。

## 9. 红线（务必遵守）

- **绝不获取/使用隐藏 test 集的真答案**来做错误分析或调模型 —— 那是测试集泄露(test-set contamination) + 学术不端，榜分会虚高且不可复现。要做"对症下药"，**用带真标签的验证集 / K 折**，那是 test 误差的无偏估计。
- 增强**严禁**色相/饱和度/通道交换/灰度化 —— 红色是监督信号。
- 数据目录只读，划分函数单一来源，写 csv 一律 `dtype=str, keep_default_na=False` + LF 行尾（防 pandas 把 `NA` 当缺失、防空 label 变 nan）。
