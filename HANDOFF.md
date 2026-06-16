# 红色字符识别 — 交接文档 (HANDOFF)

> 面向接手的 agent / 协作者。读完即可复现现状并继续推进。最后更新：2026-06-16（**最终版**）。
> 完整方法另见 `方法报告_98.72.md`（技术报告）。当前最佳 = 平台 **98.72%**，已定稿。

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
| **+ glyph 红度+红线增强(gfrl) ★最终最佳** | **0.9944** | **0.9872** |
| I/1 过采样(gfb) | 0.9948 | 0.9864（**反降，作废**） |

**★ 最终提交文件**：`red_char/outputs/submission_reranker_gfrl.csv`（平台 **0.9872**）。
**重要教训**：2,500 验证集在 ≥0.994 这个精度上已分辨不出真实差异（1~2 个样本=噪声）。`gfb` 在 val 涨 1 个样本(0.9944→0.9948)却在平台**反降**(0.9872→0.9864)——**val 的 ±1 样本不可信，平台是唯一裁判，且已到平台期**。后续若要继续，必须用 K 折 OOF（全量 50000）或更大留出集来判断改进，切勿凭 2500 val 的小数点末位提交。
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
     - `gfrl` **红度 + 红线增强 `--red-line-aug 0.5`**（训练叠加合成红线，抗残留红线）★最终：0.9944 / **0.9872**
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
| `train_glyph.py` | 训练 glyph；`--hires --head-mode gap --input-mode red --red-line-aug --crop-width --boost-chars --boost-factor --all-glyphs --fold` |
| `eval_reranker.py` / `predict_reranker.py` | 主模型+TTA+选择性重排：验证 / 生成 submission |
| `glyph_perclass.py` | glyph 逐类(I/1/S/G…)准确率，验证定向改进是否真有效 |
| `kfold.py` / `oof_predict.py` / `tune_oof.py` | K 折 OOF 框架（全量 50000 无偏调参/诊断） |
| `synth.py` / `denoise.py` / `train_denoise.py` / `make_denoised.py` | 去线 U-Net（**负面**，留档） |
| `evaluate.py` / `train_pair_glyph.py` / `eval_pair_reranker.py` / `eval_knn_reranker.py` | 旧/负面实验，留档 |

**Checkpoints**（`outputs/checkpoints/best_*.pt`，含 EMA 权重）：
- **★最终系统成员**：主模型 12 个 = `best_ps1/2/3.pt`+`best_r2ps1/2/3.pt`+`best_v2hs1..6.pt`；glyph = **`best_gfrl1/2.pt`**
- v2 多 seed：`best_seed1/2/3.pt`（早期）
- glyph 迭代：`best_gf1.pt`(原始)、`best_gfg1-3.pt`(hires+gap)、`best_gfr1/2.pt`(红度)、`best_gfrl1/2.pt`(红度+红线★)
- 负面留档：`best_gfb1/2.pt`(I/1过采样,平台反降)、`best_gfw1/2.pt`(宽裁切)、`best_hps*`(强增强伪标签)、`best_v3s*`(预训练)、`best_dn*/cd*`(去线/6通道)、`best_gfh*`(flat-hires)、`best_pair*`、`denoiser_v2.pt`、各折 `best_f*s1/gff*.pt`

## 7. 复现关键命令

```bash
cd red_char
PY=/home/duxuanzheng/.conda/envs/red_char/bin/python
CK=outputs/checkpoints

PRIM="$CK/best_ps1.pt $CK/best_ps2.pt $CK/best_ps3.pt $CK/best_r2ps1.pt $CK/best_r2ps2.pt $CK/best_r2ps3.pt $CK/best_v2hs1.pt $CK/best_v2hs2.pt $CK/best_v2hs3.pt $CK/best_v2hs4.pt $CK/best_v2hs5.pt $CK/best_v2hs6.pt"

# ★ 评测最终系统(val，输出 selective exact≈0.9944)
CUDA_VISIBLE_DEVICES=0 $PY eval_reranker.py --x-tta --top-k 3 \
  --checkpoints $PRIM --glyph-checkpoints $CK/best_gfrl1.pt $CK/best_gfrl2.pt

# ★ 生成最终提交（平台 0.9872）= submission_reranker_gfrl.csv
CUDA_VISIBLE_DEVICES=0 $PY predict_reranker.py --x-tta --selective --top-k 3 \
  --primary-margin-max 0.40 --glyph-margin-min 0.05 --red-threshold 0.20 \
  --checkpoints $PRIM --glyph-checkpoints $CK/best_gfrl1.pt $CK/best_gfrl2.pt \
  --output outputs/submission_reranker_gfrl.csv

# 训练最终 glyph（红度+红线增强）
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 setsid $PY -u \
  train_glyph.py --hires --head-mode gap --input-mode red --red-line-aug 0.5 \
  --epochs 30 --seed 1 --tag _gfrl1 > outputs/logs/gfrl1.log 2>&1 </dev/null &

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

**已定稿**：最终最佳 = `submission_reranker_gfrl.csv`，平台 **0.9872**。用户已决定停止纯模型侧改进（I/1 过采样平台反降验证了平台期）。

若后续仍要推进（按优先级 / 全部合规）：
1. **凡改进必用 K 折 OOF（全量 50000）判断**，禁止凭 2500 val 末位小数提交——`gfb` 已证 val +1 样本会在平台反降。
2. **全量数据重训最终主模型 + glyph**：把 2500 验证集并回训练（多 5% 真标签），同配方固定 epoch + EMA(last)，可能小涨；无 val 时以当前模型为回退。
3. **真正可能大跳**：识别/获取验证码**生成器或确切字体**，做零域差距的合成数据训练（推测同学 99.9% 来源）。这是纯模型侧之外唯一高概率路径。
4. **红线方向延伸**只在 glyph 有效；主模型、宽裁切、去线均已证无效，勿重复。
5. 红线：**绝不**获取/使用隐藏 test 答案调模型（测试集泄露+学术不端）；对症分析用带真标签的验证集/K 折。

## 9. 红线（务必遵守）

- **绝不获取/使用隐藏 test 集的真答案**来做错误分析或调模型 —— 那是测试集泄露(test-set contamination) + 学术不端，榜分会虚高且不可复现。要做"对症下药"，**用带真标签的验证集 / K 折**，那是 test 误差的无偏估计。
- 增强**严禁**色相/饱和度/通道交换/灰度化 —— 红色是监督信号。
- 数据目录只读，划分函数单一来源，写 csv 一律 `dtype=str, keep_default_na=False` + LF 行尾（防 pandas 把 `NA` 当缺失、防空 label 变 nan）。
