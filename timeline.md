# 红色字符识别实验 Timeline

> 目的：记录为了提高红色字符整串 exact-match 命中率所做的每一次工程实现、训练、评估和失败尝试。后续任何新训练、新脚本、新集成搜索或提交文件生成，都必须在本文件追加记录。

## 维护规则

- 每次改进都记录：时间、动作、理由、命令或关键参数、验证结果、结论和下一步。
- 失败尝试也必须记录，特别是没有提升的方案，避免重复走弯路。
- 训练记录至少包含：seed、run-name、epochs、augment、特殊 loss/结构参数、best epoch、val exact、char/color/joint 指标、checkpoint 路径。
- 集成搜索记录至少包含：checkpoint 列表、权重搜索粒度、最佳权重、val exact、错误数或剩余瓶颈。
- 最终提交前记录 submission 生成命令、使用的 checkpoint/权重、格式自检结果和本地验证指标。

## 2026-06-15 至 2026-06-16

### 1. 任务方案与环境确认

**动作**：依据 `DEV_PLAN_红色字符识别.md` 实现多任务 CNN 路线：共享 CNN 主干，分别预测 5 个字符类别和 5 个红/非红颜色标签。

**理由**：任务最终只提交红色字符组成的字符串，但训练集中非红位置也有字符监督。多任务按位预测能同时利用字符和颜色信息，比手工颜色阈值或 CRNN/CTC 更适合固定 5 位验证码。

**服务器环境检查**：

- 服务器路径：`/home/lmt_ssd/red-char-recognition/red_char`
- GPU：2 张 RTX 3090，均空闲，24GB 显存
- 数据盘：`/home/lmt_ssd` 可用约 2.6T
- Conda 环境：使用已有 `intermimic-lab`
- PyTorch：`2.7.0+cu128`
- CUDA available：True
- 后补依赖：`pandas`、`tqdm`

**结论**：无需新建训练环境，直接复用 `intermimic-lab`。

### 2. 基础代码与数据验收

**动作**：上传并验证基础文件：

- `config.py`
- `dataset.py`
- `eda.py`
- `evaluate.py`
- `metrics.py`
- `model.py`
- `predict.py`
- `train.py`
- `requirements.txt`

**数据验收结果**：

- 代码文件：9 个
- 训练图像：50,000 张
- 测试图像：5,000 张
- `labels.csv` SHA256：`73b36f9349afb334ccc96302454ed553821bc4d9af343c5e64b6c5edf13bd6cf`
- `submission_sample.csv` SHA256：`a66b0f8d2fed62940cdd4caa964b2b0676cd913b0e9603d55a4478474fb439ac`
- 红/非红总数约各半：`{'r': 124917, 'u': 125083}`
- 每个位置红色比例约 0.5
- 红字数分布：1 到 4 个均约 12.5K
- 字符类别：36 类全覆盖
- 空 label 原始字节确认：`b'00034.png,'`

**验证**：

- `eda.py` passed，样例图输出到 `outputs/eda/samples.png`
- `dataset.py` self-test passed
- `model.py` 输出 shape：`[2,5,36]` 和 `[2,5,2]`
- 参数量：5,990,302
- 64 样本 overfit sanity 通过，loss 下降到约 `0.010`

**结论**：数据、编码、模型结构、训练闭环没有基础 bug。

### 3. 首次完整训练：baseline 30 epochs

**动作**：使用无增强 baseline 完整训练 30 epochs。

**理由**：先建立可靠基线，确认模型路线是否足够强，再决定是否增加增强或集成。

**结果**：

- 训练初期：epoch 1 exact `0.0376`
- epoch 3 exact `0.8328`
- best checkpoint：epoch 26
- best val exact：`0.9700`
- char_acc：`0.97888`
- color_acc：`0.99960`
- 错误数：75
- 错误类型：`char_only 70`，`color_only 5`

**结论**：颜色头已经接近完美，瓶颈主要是字符识别。需要轻量增强和模型多样性。

### 4. 加入轻量训练增强

**动作**：本地修改并上传轻量增强：

- RandomAffine：小角度旋转、小平移
- 轻微噪声
- 禁止 hue/saturation/color jitter，避免破坏红色标签语义
- 新增 `test_augmentation.py`

**理由**：baseline 主要错字符，且训练 loss 很低，轻量几何增强可能提升对字符抖动和干扰线的鲁棒性。

**验证**：

- 服务器单元测试 3 项通过

**训练结果：增强 30 epochs**：

- epoch 24 exact：`0.9760`
- epoch 30 exact：`0.9760`

**结论**：增强有效，从 97.00% 提升到 97.60%，但仍不够。

### 5. 延长增强训练到 50 epochs

**动作**：将增强训练延长到 50 epochs。

**理由**：30 epochs 仍在缓慢提升，且每个 epoch 约 10 秒，继续训练成本低。

**结果：seed42 / default path**：

- checkpoint：`outputs/checkpoints/best.pt`
- best epoch：50
- val_loss：`0.06932735087871551`
- exact：`0.9784`
- char_acc：`0.9848800099372864`
- color_acc：`0.9999199965476989`
- joint_pos_acc：`0.9848000111579895`
- 错误数：54

**结论**：延长训练继续提升，但单模型距离 99% 仍有明显差距。

### 6. 多 seed 训练与隔离 run-name

**动作**：修改训练脚本，支持：

- `--seed`
- `--run-name`
- 固定 `SPLIT_SEED=42`
- 不同 seed 输出到 `outputs/runs/<run-name>/`

**理由**：多 seed 集成需要不同训练随机性，但验证集划分必须完全一致，否则 checkpoint 不可直接比较或集成。

**验证**：

- `test_training_seed_does_not_change_fixed_validation_split` 通过
- `test_named_run_uses_isolated_output_paths` 通过
- `test_default_run_keeps_legacy_output_paths` 通过

**训练记录**：

| seed | run-name | checkpoint | best epoch | exact | char_acc | color_acc | 备注 |
|---:|---|---|---:|---:|---:|---:|---|
| 43 | `augment_seed43` | `outputs/runs/augment_seed43/checkpoints/best.pt` | 43 | 0.9796 | 0.98552 | 0.99976 | 正常完成 |
| 44 | `augment_seed44_clean` | `outputs/runs/augment_seed44_clean/checkpoints/best.pt` | 45 | 0.9812 | 0.98568 | 0.99984 | 最高单模型 |

**事故记录**：

- 曾误启动重复 seed44，形成脏目录 `outputs/runs/augment_seed44/`
- 后续明确不再使用该目录，保留 clean run：`augment_seed44_clean`

**结论**：多 seed 单模型最高到 98.12%，集成有继续提升空间。

### 7. 等权 logits 集成

**动作**：新增：

- `ensemble.py`
- `ensemble_search.py`
- evaluate/predict 支持 `--checkpoints`

**理由**：多个 seed 的错误不完全重叠，logits 平均通常可提升验证码整串准确率。

**验证**：

- 服务器测试 7 项通过

**等权搜索结果（三模型）**：

候选：

- `outputs/checkpoints/best.pt`
- `outputs/runs/augment_seed43/checkpoints/best.pt`
- `outputs/runs/augment_seed44_clean/checkpoints/best.pt`

最佳组合：

- checkpoint：seed42 + seed44_clean
- exact：`0.9852`
- char_acc：`0.98864`
- color_acc：`0.99992`

**结论**：集成明显有效，从 98.12% 单模型提升到 98.52%，但仍未到 99%。

### 8. 字符/颜色分头加权集成搜索

**动作**：新增：

- `weighted_ensemble_search.py`
- `--char-weights`
- `--color-weights`
- 向量化 `encode_red_sequences`

**理由**：错误几乎都是字符错，颜色接近满分。字符头和颜色头最佳模型权重可能不同，分头调权可能比统一等权更优。

**验证**：

- 本地 11 项测试通过
- 服务器 11 项测试通过
- smoke weighted search 正常生成 CSV

**三模型 step=0.05 搜索结果**：

- checkpoints：seed42、seed43、seed44_clean
- best exact：`0.9860`
- char_acc：`0.9890399575`
- color_acc：`1.0`
- joint_pos_acc：`0.9890399575`
- char_weights：`0.45|0.1|0.45`
- color_weights：`0|0.3|0.7`
- 复核错误数：35

**结论**：分头加权有效，从 98.52% 提升到 98.60%。瓶颈完全变成字符识别。

### 9. TTA 尝试

**动作**：新增轻量 TTA：

- `tta.py`
- 五视图：原图、左右平移 2 像素、上下平移 1 像素
- 白色填充，不环绕
- `evaluate.py` / `predict.py` 支持 `--tta`

**理由**：剩余 35 个错误是少量字符混淆，微小平移可能改善字符边界和干扰线扰动下的 logits。

**验证**：

- 本地 15 项测试通过
- 服务器 15 项测试通过
- 本地真实 checkpoint TTA 评估链路正常

**三模型加权 + TTA 结果**：

- exact：`0.9856`
- char_acc：`0.98904`
- color_acc：`1.0`
- 错误数：36

**结论**：TTA 没有提升，反而略降。后续最终推理暂不使用 `--tta`，除非后续模型显示相反结果。

### 10. 增加 seed45 / seed46 模型

**动作**：继续训练两个增强 seed：

- `augment_seed45`
- `augment_seed46`

**理由**：现有三模型集成已接近瓶颈，需要增加独立模型多样性，期望纠正不同样本。

**训练结果**：

| seed | run-name | checkpoint | best epoch | exact | char_acc | color_acc |
|---:|---|---|---:|---:|---:|---:|
| 45 | `augment_seed45` | `outputs/runs/augment_seed45/checkpoints/best.pt` | 43 | 0.9804 | 0.98568 | 0.99984 |
| 46 | `augment_seed46` | `outputs/runs/augment_seed46/checkpoints/best.pt` | 44/49/50 附近 | 0.9796 | 约 0.9848 | 0.9999 |

**等权 5 模型子集搜索**：

最佳组合：

- size：4
- checkpoints：seed42 + seed43 + seed44_clean + seed45
- exact：`0.9868`
- char_acc：`0.9896000124`
- color_acc：`0.9999199965`

**结论**：seed45 对集成有贡献，seed46 暂时贡献有限。

### 11. top4 / top5 加权搜索

**top4 step=0.1**：

- checkpoints：seed42、seed43、seed44_clean、seed45
- exact：`0.9876`
- char_acc：`0.9893599749`
- color_acc：`1.0`
- char_weights：`0.2|0|0.5|0.3`
- color_weights：`0|0.2|0.3|0.5`

**top5 step=0.1**：

- checkpoints：seed42、seed43、seed44_clean、seed45、seed46
- exact：`0.9876`
- char_acc：`0.9895199537`
- color_acc：`1.0`
- char_weights：`0.3|0|0.4|0.2|0.1`
- color_weights：`0|0|0.2|0.4|0.4`

**top4 step=0.05**：

- checkpoints：seed42、seed43、seed44_clean、seed45
- exact：`0.9879999757`
- char_acc：`0.9892799854`
- color_acc：`1.0`
- char_weights：`0.15|0.05|0.45|0.35`
- color_weights：`0|0.15|0.3|0.55`
- 复核错误数：30
- 错误类型：`char_only 30`

**结论**：当前最好验证集 exact 约 98.80%，离 99% 还差至少 5 张。颜色完全不是瓶颈，继续调颜色或 TTA 价值低。

### 12. 红色字符位置加权训练改动

**动作**：修改训练 loss，新增：

- `compute_loss(..., red_char_weight=...)`
- `train.py --red-char-weight`
- checkpoint 的 `metrics` 和 `config` 记录 `red_char_weight`

**理由**：最终指标只看红色字符组成的字符串。此前训练对红色位和非红色位字符 loss 等权，但所有剩余错误均为红串相关字符错误。提高红色位置字符 loss 权重，可把训练火力更集中到最终评价目标上。

**实现原则**：

- 默认 `red_char_weight=1.0`，完全兼容旧训练
- 加权后按权重和归一化，避免整体 loss 尺度突然变大
- 颜色 loss 不变

**本地验证**：

- 18 项单元测试通过
- `py_compile` 通过
- `git diff --check` 通过
- 1 epoch / 1 step smoke training 通过
- smoke checkpoint 中 `metrics.red_char_weight=2.5`
- smoke checkpoint 中 `config.red_char_weight=2.5`

**服务器状态**：

- `metrics.py`、`train.py`、`test_ensemble.py` 已上传
- 服务器测试已通过：`Ran 18 tests in 0.066s, OK`
- 下一步训练红字加权模型

**计划训练**：

| seed | run-name | red_char_weight | 目的 |
|---:|---|---:|---|
| 47 | `red_weight_seed47` | 2.5 | 验证红色位置字符加权是否改善剩余 char_only 错误 |
| 48 | `red_weight_seed48` | 2.5 或 3.0 | 增加红字加权模型多样性 |

**训练完成后的日志尾部观察**：

| seed | run-name | red_char_weight | 末尾最佳可见 epoch | 可见 exact | 可见 char_acc | 可见 color_acc | 观察 |
|---:|---|---:|---:|---:|---:|---:|---|
| 47 | `red_weight_seed47` | 2.5 | 50 | 0.9820 | 0.9854 | 0.9999 | 单模型超过此前 `augment_seed44_clean` 的 0.9812，说明红字加权有效 |
| 48 | `red_weight_seed48` | 2.5 | 45 | 0.9804 | 0.9849 | 0.9997 | 单模型不如 seed47，但可能提供集成多样性 |

**阶段结论**：红字加权模型没有显著提高整体 char_acc，但 seed47 的 exact-match 达到新的单模型最高值。下一步应把 red-weight checkpoints 纳入集成搜索，判断它们是否能纠正 top4 加权集成剩余的 30 个 char_only 错误。

**6 候选等权子集搜索**：

候选：

- `outputs/checkpoints/best.pt`
- `outputs/runs/augment_seed43/checkpoints/best.pt`
- `outputs/runs/augment_seed44_clean/checkpoints/best.pt`
- `outputs/runs/augment_seed45/checkpoints/best.pt`
- `outputs/runs/red_weight_seed47/checkpoints/best.pt`
- `outputs/runs/red_weight_seed48/checkpoints/best.pt`

最佳等权组合：

- size：5
- checkpoints：seed42 + seed44_clean + seed45 + red_weight_seed47 + red_weight_seed48
- exact：`0.9872000005722046`
- char_acc：`0.9893600130081177`
- color_acc：`0.9999199965476989`
- val_loss：`0.04062287650704384`

**结论**：红字加权模型进入了最佳等权组合，但等权结果 98.72% 仍低于当前 top4 加权最佳 98.80%。下一步应对该 5 模型组合做分头加权搜索；不要直接用 6 模型全网格加权，因为现有脚本会写出约 900 万行结果，内存和 CSV 都不划算。

**redmix5 step=0.1 分头加权搜索**：

候选顺序：

1. `outputs/checkpoints/best.pt`
2. `outputs/runs/augment_seed44_clean/checkpoints/best.pt`
3. `outputs/runs/augment_seed45/checkpoints/best.pt`
4. `outputs/runs/red_weight_seed47/checkpoints/best.pt`
5. `outputs/runs/red_weight_seed48/checkpoints/best.pt`

最佳结果：

- exact：`0.9879999756813049`
- char_acc：`0.9896799921989441`
- color_acc：`1.0`
- joint_pos_acc：`0.9896799921989441`
- char_weights：`0.2|0.3|0.2|0.2|0.1`
- color_weights：`0|0|0.5|0.4|0.1`

**结论**：redmix5 加权追平当前最高 exact 98.80%，并且位置级 char_acc 高于此前 top4 step=0.05 的 0.98928。说明红字加权模型修正了更多单位置字符，但这些修正尚未转化为更高整串 exact。下一步应导出该组合的 30 个左右错误，与 top4 错误集合比较；若错误集合不同，可做局部/贪心搜索，而不是全量 5 模型 step=0.05 暴搜。

**redmix5 复核与错误集合**：

- 复核 exact：`0.9880000005722046`
- char_acc：`0.9896800081253052`
- color_acc：`0.9999999953269958`
- 错误数：30
- 错误类型：`char_only 30`
- 与 top4 step=0.05 的 30 错集合相比，文件集合几乎完全重合；主要差异是 top4 错 `41741.png`，redmix5 错 `41547.png`。

**结论**：red_weight 模型带来了一些位置级字符修正，但没有形成足够的整串互补。继续使用现有 `weighted_ensemble_search.py` 做 5 模型 step=0.05 全网格会产生约 1.13 亿个组合行，CSV 和内存都不划算。下一步若继续挖集成，应新增“只保留 top-k 候选、不写全量 CSV”的 beam/局部搜索脚本；若仍不能提升到 99%，需要训练结构不同的模型，而不是继续堆同结构 seed。

### 13. beam / 局部权重搜索脚本

**动作**：新增 `beam_weight_search.py`。

**理由**：现有 `weighted_ensemble_search.py` 会把所有权重组合写入 CSV。对于 5 模型、`step=0.05`，字符权重和颜色权重各有 10626 个候选，组合约 1.13 亿行，不适合继续暴搜。新脚本只保留 top-k，先粗搜，再围绕最佳权重局部细搜。

**搜索策略**：

- 粗搜：默认 `--coarse-step 0.1`
- 局部细搜：默认 `--fine-step 0.02`
- 局部半径：默认 `--radius 0.08`
- 仅保留：默认 `--top-k 20`
- 仍然分开搜索字符权重和颜色权重
- 输出 top-k CSV，而非全量组合 CSV

**本地验证**：

- 新增测试：
  - parser 支持 coarse/fine/radius/top-k 参数
  - 局部权重向量归一化且位于中心附近
  - top-k 行按 exact、char_acc、color_acc、joint_pos_acc 排序
- 本地 21 项单元测试通过
- `py_compile` 通过
- `git diff --check` 通过
- 本地 smoke 搜索通过：
  - checkpoints：重复使用 `outputs/runs/smoke_seed43/checkpoints/best.pt`
  - `coarse-step=0.5`
  - `fine-step=0.5`
  - `top-k=3`
  - 输出：`outputs/logs/smoke_beam_weight_search.csv`

**计划服务器验证命令**：

```bash
cd /home/lmt_ssd/red-char-recognition/red_char && conda run --no-capture-output -n intermimic-lab python beam_weight_search.py --checkpoints \
outputs/checkpoints/best.pt \
outputs/runs/augment_seed44_clean/checkpoints/best.pt \
outputs/runs/augment_seed45/checkpoints/best.pt \
outputs/runs/red_weight_seed47/checkpoints/best.pt \
outputs/runs/red_weight_seed48/checkpoints/best.pt \
--coarse-step 0.1 \
--fine-step 0.02 \
--radius 0.08 \
--top-k 20 \
--output outputs/logs/beam_weight_search_redmix5.csv
```

**服务器 redmix5 beam 搜索结果**：

- checkpoints：seed42 + seed44_clean + seed45 + red_weight_seed47 + red_weight_seed48
- coarse vectors：1001
- fine char vectors：3951
- fine color vectors：1070
- best stage：`fine`
- exact：`0.9883999824523926`
- char_acc：`0.9898399710655212`
- color_acc：`1.0`
- joint_pos_acc：`0.9898399710655212`
- char_weights：`0.12|0.22|0.26|0.26|0.14`
- color_weights：`0|0|0.46|0.48|0.06`
- log：`outputs/logs/beam_weight_search_redmix5.csv`

**结论**：beam/局部搜索有效，将当前最佳从 98.80% 提升到约 98.84%，约少错 1 张。仍未达到 99%，但证明在 redmix5 权重附近还有小幅可挖空间。下一步应复核该权重并导出错误集合；若错误数为 29，离 99% 仍差至少 4 张。

**beam 权重复核**：

- exact：`0.9884000005722046`
- char_acc：`0.989840005683899`
- color_acc：`0.9999999953269958`
- joint_pos_acc：`0.989840005683899`
- 错误数：29

**结论**：复核与搜索结果一致。当前距离 99% 还差至少 4 张验证样本。下一步必须查看这 29 个错误是否仍与此前 30 错高度重叠；若高度重叠，说明同结构集成已接近上限，应训练结构差异模型。

**beam 29 错错误集合观察**：

- 仍然全部表现为字符识别瓶颈，颜色预测保持正确。
- 与此前 top4/redmix5 的 30 错集合高度重合。
- 相比 top4 step=0.05，beam 权重解决了 `20448.png`、`40273.png`、`41741.png`。
- beam 权重同时新增了 `16488.png`、`41547.png`。
- 净减少 1 张错误，达到 29 错。

**结论**：局部权重搜索已经接近同结构集成上限。要继续靠集成提升到 99%，需要加入结构不同、错误模式不同的模型，而不是继续训练同一个 `RedCharNet` 的 seed。

### 14. wide 结构模型

**动作**：新增 `--model-size base|wide`。

**理由**：当前最佳已经是 29 错，且错误集合与此前 30 错高度重合。同结构 seed、红字加权和局部权重搜索都已经榨过，继续堆同结构模型互补性有限。wide 模型通过更宽通道和更大 neck 学习不同的笔画/干扰线判别边界，有机会纠正顽固字符混淆。

**结构设计**：

| model_size | channels | neck_dim | dropout | 参数量 |
|---|---|---:|---:|---:|
| `base` | 32-64-128-256 | 512 | 0.3 | 5,990,302 |
| `wide` | 48-96-192-384 | 768 | 0.4 | 13,402,126 |

**兼容性设计**：

- `train.py` 默认 `--model-size base`，不影响旧训练命令。
- checkpoint 的 `metrics` 和 `config` 记录 `model_size`。
- `ensemble.load_models` 根据 checkpoint 的 `config.model_size` 自动构造对应模型。
- 旧 checkpoint 没有 `model_size` 时默认按 `base` 加载。

**本地验证**：

- 新增测试：
  - `--model-size wide` 参数解析
  - wide 输出 shape 仍为 `(B,5,36)` 与 `(B,5,2)`，且参数量大于 base
  - `load_models` 能根据 wide checkpoint 自动构造 wide 模型
- 本地 24 项单元测试通过
- `py_compile` 通过
- `git diff --check` 通过
- `python model.py` 通过：
  - base 参数量：5,990,302
  - wide 参数量：13,402,126
- wide 1-step 训练 smoke 通过：
  - `run_name=smoke_wide`
  - `red_char_weight=2.5`
  - `model_size=wide`
  - checkpoint 中 `metrics.model_size=wide`
  - checkpoint 中 `config.model_size=wide`

**服务器验证**：

- 单元测试通过：`Ran 24 tests in 0.557s, OK`
- `python model.py` 通过：
  - base 输出 shape：`[2,5,36]` / `[2,5,2]`，参数量 5,990,302
  - wide 输出 shape：`[2,5,36]` / `[2,5,2]`，参数量 13,402,126

**计划训练**：

| seed | run-name | model_size | red_char_weight | 目的 |
|---:|---|---|---:|---|
| 49 | `wide_seed49` | wide | 2.5 | 训练第一个结构差异模型，检验是否带来互补错误 |
| 50 | `wide_seed50` | wide | 2.5 | 增加 wide 结构内部多样性 |

**训练结果**：

| seed | run-name | model_size | red_char_weight | best epoch | exact | char_acc | color_acc | checkpoint |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 49 | `wide_seed49` | wide | 2.5 | 37 | 0.9864 | 0.98752 | 0.99992 | `outputs/runs/wide_seed49/checkpoints/best.pt` |
| 50 | `wide_seed50` | wide | 2.5 | 46 | 0.9836 | 0.98704 | 0.99992 | `outputs/runs/wide_seed50/checkpoints/best.pt` |

**观察**：

- wide49 单模型 exact 达到 98.64%，明显超过此前所有 base 单模型。
- wide50 单模型 exact 为 98.36%，弱于 wide49，但 char_acc 仍高于多数 base 单模型。
- wide 模型的 val_loss 明显更低，说明结构容量确实改变了拟合状态。
- 下一步应把 wide49/50 加入集成子集搜索和 beam 搜索，检验是否能纠正当前 beam 最佳剩余的 29 张。

**7 候选等权子集搜索**：

候选：

- `outputs/checkpoints/best.pt`
- `outputs/runs/augment_seed44_clean/checkpoints/best.pt`
- `outputs/runs/augment_seed45/checkpoints/best.pt`
- `outputs/runs/red_weight_seed47/checkpoints/best.pt`
- `outputs/runs/red_weight_seed48/checkpoints/best.pt`
- `outputs/runs/wide_seed49/checkpoints/best.pt`
- `outputs/runs/wide_seed50/checkpoints/best.pt`

最佳组合：

- size：5
- checkpoints：`augment_seed45` + `red_weight_seed47` + `red_weight_seed48` + `wide_seed49` + `wide_seed50`
- val_loss：`0.039212664031982425`
- exact：`0.9892000005722046`
- char_acc：`0.9904799989700317`
- color_acc：`0.9999199965476989`
- joint_pos_acc：`0.9904000001907348`

**结论**：wide 结构带来强互补性，等权集成直接从 98.84% 提升到 98.92%，仅差约 2 张达到 99%。最佳组合完全移除了 seed42 和 seed44_clean，说明 wide 与红字加权模型形成了新的更优集成核心。下一步对该 5 模型组合做 beam 分头加权搜索。

**widemix5 beam 分头加权搜索**：

候选顺序：

1. `outputs/runs/augment_seed45/checkpoints/best.pt`
2. `outputs/runs/red_weight_seed47/checkpoints/best.pt`
3. `outputs/runs/red_weight_seed48/checkpoints/best.pt`
4. `outputs/runs/wide_seed49/checkpoints/best.pt`
5. `outputs/runs/wide_seed50/checkpoints/best.pt`

最佳结果：

- stage：`fine`
- exact：`0.990399956703186`
- char_acc：`0.9904799461364746`
- color_acc：`1.0`
- joint_pos_acc：`0.9904799461364746`
- char_weights：`0.06|0.04|0|0.66|0.24`
- color_weights：`0.06|0.56|0.14|0.24|0`
- log：`outputs/logs/beam_weight_search_widemix5.csv`

**结论**：验证集搜索结果首次超过 99%，达到约 99.04%。下一步必须用 `evaluate.py` 独立复核同一 checkpoint 顺序和权重，并导出错误数；复核通过后再生成最终 submission。

**widemix5 独立复核**：

- evaluate exact：`0.9904000005722046`
- char_acc：`0.9904800097465515`
- color_acc：`0.9999999953269958`
- joint_pos_acc：`0.9904800097465515`
- 错误数：24
- checkpoint 顺序：
  1. `augment_seed45`
  2. `red_weight_seed47`
  3. `red_weight_seed48`
  4. `wide_seed49`
  5. `wide_seed50`
- char_weights：`0.06 0.04 0 0.66 0.24`
- color_weights：`0.06 0.56 0.14 0.24 0`

**结论**：独立评估与 beam 搜索一致，验证集 exact 达到 99.04%，满足“希望达到 99%”目标。下一步生成最终 `outputs/submission.csv` 并运行内置格式自检。

**submission 生成**：

- 输出文件：`outputs/submission_widemix5_9904.csv`
- 使用 checkpoint 顺序：`augment_seed45` + `red_weight_seed47` + `red_weight_seed48` + `wide_seed49` + `wide_seed50`
- char_weights：`0.06 0.04 0 0.66 0.24`
- color_weights：`0.06 0.56 0.14 0.24 0`
- 预测完成：20 个 batch
- 预测长度分布：`{1: 1268, 2: 1305, 3: 1232, 4: 1195}`
- 格式自检通过并写出文件
- 警告：
  - empty-label count 可疑：当前为 0
  - length-5 count 可疑：当前为 0

**结论**：文件格式可提交，但长度分布暴露了一个泛化风险：训练/验证集中没有 0 红和 5 红样本，当前 argmax 推理没有产生长度 0 或长度 5 的预测。若测试集确实包含 0 红或 5 红样本，平台分数会低于 99.04% 验证集分数。提交前建议额外生成一个基于颜色置信度校准 0/5 长度的版本，用于和原始版本双提交对比。

**计划服务器命令**：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 conda run --no-capture-output -n intermimic-lab python train.py --epochs 50 --augment --seed 49 --run-name wide_seed49 --red-char-weight 2.5 --model-size wide > outputs/runs/wide_seed49/console.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n intermimic-lab python train.py --epochs 50 --augment --seed 50 --run-name wide_seed50 --red-char-weight 2.5 --model-size wide > outputs/runs/wide_seed50/console.log 2>&1 &
```

**下一步命令候选**：

```bash
cd /home/lmt_ssd/red-char-recognition/red_char && conda run --no-capture-output -n intermimic-lab python -m unittest -v test_ensemble.py test_augmentation.py
```

若测试通过，启动：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 conda run --no-capture-output -n intermimic-lab python train.py --epochs 50 --augment --seed 47 --run-name red_weight_seed47 --red-char-weight 2.5 > outputs/runs/red_weight_seed47/console.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n intermimic-lab python train.py --epochs 50 --augment --seed 48 --run-name red_weight_seed48 --red-char-weight 2.5 > outputs/runs/red_weight_seed48/console.log 2>&1 &
```

## 当前最佳方案快照

截至 2026-06-16，验证集最佳结果：

- exact：`0.9879999757`
- 约 30 / 2500 错
- checkpoints：seed42、seed43、seed44_clean、seed45
- char_weights：`0.15 0.05 0.45 0.35`
- color_weights：`0 0.15 0.3 0.55`
- 不使用 TTA

当前瓶颈：

- 剩余错误全部为 `char_only`
- 颜色预测已经达到或接近 100%
- 需要通过红字加权训练、结构差异模型或更强字符专注策略继续提升

## Kaggle 测试机反馈与本地后处理实验

截至 2026-06-17，`submission_widemix5_9904.csv` 在 Kaggle 测试集得分为 **98 分**，低于验证集 `99.04%`。

**现象分析**：

- 最终提交长度分布：`{1: 1268, 2: 1305, 3: 1232, 4: 1195}`
- `submission_sample.csv` 长度分布：`{0: 150, 1: 1316, 2: 1304, 3: 1156, 4: 1070, 5: 4}`
- 内置预测阶段曾警告：empty-label count 和 length-5 count 可疑。

**假设**：

测试集可能包含真实的 0 红字符样本，而当前模型/验证切分没有覆盖 0 红和 5 红极端长度，导致最终提交没有产生空标签，从而拉低 Kaggle 分数。

**本地尝试**：

- 新增 `red_char/calibrate_extreme_lengths.py`
- 根据测试图像红色像素证据，将红像素最低的一批样本校准为空标签。
- 生成候选：
  - `submissions/submission_widemix5_empty050.csv`
  - `submissions/submission_widemix5_empty100.csv`
  - `submissions/submission_widemix5_empty150.csv`
  - `submissions/submission_widemix5_empty200.csv`

**理由**：

空红样本在图像上应具有极低红色像素数量。该策略不改动大部分字符预测，只校准极端长度先验，是在缺少正式 checkpoint/logits 的本地条件下，最快可提交验证的低成本尝试。

**提交状态**：

- 已在本地项目虚拟环境 `.kaggle_venv` 安装 Kaggle CLI。
- 尝试提交 `submission_widemix5_empty150.csv`：
  - message：`score98 empty150 red-pixel calibration`
  - 结果：Kaggle public score 为 `0.95040`
- 对照 baseline：
  - `submission_widemix5_9904.csv`：`0.98000`
  - `submission_widemix5_empty150.csv`：`0.95040`

**结论**：

将红像素最低的 150 个样本强制置为空标签会显著降分，说明测试集 public 部分并不存在大量需要置空的样本，或者该红像素阈值不能可靠识别空红样本。停止继续提交 `empty050 / empty100 / empty200`，避免浪费提交配额；下一步转向本地训练/模型推理改进，而不是极端长度后处理。

## 本地训练恢复

本地环境检查：

- PyTorch：`2.11.0+cu128`
- CUDA：可用
- GPU 数量：1
- 本地已有 checkpoint 仅为 epoch=1 smoke 产物，不能复用为最终提交模型。

**下一步尝试**：

启动本地正式训练 `local_wide_seed51`：

- model_size：`wide`
- seed：`51`
- augment：启用
- red_char_weight：`2.5`
- epochs：`50`

**理由**：

服务器阶段 wide 模型在验证集上提供了明显增益，且与 base/red-weight 模型形成互补；本地具备 CUDA 后，优先复现 wide + red-weight 的单模型能力，再决定是否继续训练多个 seed 做本地 ensemble。

**本地启动排障与正式启动（2026-06-17）**：

- 先跑本地验证：
  - `python -m unittest -v test_ensemble.py test_augmentation.py`
  - 结果：25 项测试通过
  - `py_compile` 通过
  - `git diff --check` 通过
- 首次后台训练尝试使用 `Start-Process -ArgumentList @(...)`，进程快速退出且无日志。
- 诊断发现：PowerShell `Start-Process` 的数组参数会破坏带空格的 `python -c` 参数；随后改为单字符串参数。
- 继续发现：`run_training.ps1` 使用 UTF-8 无 BOM 时，Windows PowerShell 5 会把脚本内中文绝对路径解析乱码。
- 修正：包装脚本改用 `$PSScriptRoot` 和相对路径，避免中文路径字面量。
- 继续发现：PowerShell 5 会把 native stderr/tqdm 输出当成 `NativeCommandError`，导致训练被包装脚本中断。
- 最终修正：包装脚本改为通过 `cmd.exe /c` 执行 `python -u train.py ... 1> console.log 2> console.err.log`，PowerShell 只负责启动和记录 exit code。
- 正式训练启动：
  - run-name：`local_wide_seed51`
  - PID：包装 PowerShell `69412`，Python `23680`
  - 命令参数：`--epochs 50 --augment --seed 51 --run-name local_wide_seed51 --red-char-weight 2.5 --model-size wide --num-workers 0 --no-cache-in-ram`
  - 日志：`red_char/outputs/runs/local_wide_seed51/console.log`
  - stderr：`red_char/outputs/runs/local_wide_seed51/console.err.log`

## 第二阶段本地训练与架构实现（2026-06-18）

### 本地 wide baseline 实测

由于 Codex 当前 shell 中普通后台子进程会在 shell 返回后被回收，正式训练改用 `Start-Process -Wait` 方式前台托管，并把 stdout/stderr 重定向到 run 目录。

本地 `wide + light + red_char_weight=2.5 + seed51` 训练结果：

- `local_wide_seed51_cache_5ep_20260618`
  - best epoch：`5`
  - val exact：`0.9532`
- `local_wide_seed51_cache_10ep_20260618`
  - best epoch：`10`
  - val exact：`0.9752`
- `local_wide_seed51_cache_20ep_20260618`
  - best epoch：`19`
  - val exact：`0.9792`
- `local_wide_seed51_cache_50ep_20260618`
  - best epoch：`39`
  - val exact：`0.9836`
  - best checkpoint：`red_char/outputs/runs/local_wide_seed51_cache_50ep_20260618/checkpoints/best.pt`

对 `10ep best + 20ep best` 做本地等权小集成：

- 搜索日志：`red_char/outputs/logs/local_wide_seed51_10_20_ensemble_search.csv`
- best val exact：`0.9800`

结论：同一 seed 的不同 epoch checkpoint 有少量互补，但单靠同结构 wide 无法达到第二阶段目标 `99.2%`，仍需要架构多样性。

### 第二阶段源码改动

已在 `main` 上提交：

- commit：`14b2624 Add stage2 red character architectures`
- 范围：
  - `red_char/model.py`
  - `red_char/config.py`
  - `red_char/dataset.py`
  - `red_char/train.py`
  - `red_char/test_ensemble.py`
  - `red_char/test_augmentation.py`

具体能力：

- `--model-size` 扩展为：`base | wide | k5 | resblock | deep3`
- 新增 `k5`：前两个卷积块使用 5x5 卷积，参数量约 `6,106,526`
- 新增 `resblock`：残差块替代普通 ConvBlock，参数量约 `6,034,366`
- 新增 `deep3`：3 个卷积块、保留更大空间特征图，参数量约 `24,182,398`
- 新增 `AUGMENT_PRESETS`：
  - `light`：保持旧行为，degrees=3, translate=5%, noise=0.01
  - `medium`：degrees=5, translate=8%, noise=0.02
  - `strong`：degrees=8, translate=10%, noise=0.03, black/white erasing
- `train.py` 新增 `--augment-preset`，checkpoint `config` 和 `metrics` 记录 `augment_preset`
- 保留此前 `--num-workers` 支持，便于 Windows 本地训练使用 `--num-workers 0`

### 验证结果

代码验证：

```powershell
cd D:\Learing\一路北航\机器学习\ML-TeamHW\red_char
python -m unittest discover -p "test*.py"
python model.py
python train.py --help
```

结果：

- `unittest discover`：32 项测试通过
- `model.py` 输出 shape 全部正确：
  - base：`torch.Size([2, 5, 36])`, `torch.Size([2, 5, 2])`, params `5,990,302`
  - wide：params `13,402,126`
  - k5：params `6,106,526`
  - resblock：params `6,034,366`
  - deep3：params `24,182,398`
- `train.py --help` 已显示：
  - `--model-size {base,wide,k5,resblock,deep3}`
  - `--augment-preset {light,medium,strong}`

64 样本过拟合 sanity：

- `sanity_k5_seed52_20260618`
  - 命令：`python -u train.py --overfit-sanity --seed 52 --run-name sanity_k5_seed52_20260618 --model-size k5 --num-workers 0 --cache-in-ram`
  - 结果：epoch 94 通过，loss `< 0.01` 且 exact `1.0`
- `sanity_resblock_seed54_20260618`
  - 命令：`python -u train.py --overfit-sanity --seed 54 --run-name sanity_resblock_seed54_20260618 --model-size resblock --num-workers 0 --cache-in-ram`
  - 结果：epoch 92 通过，loss `< 0.01` 且 exact `1.0`
- `sanity_deep3_seed56_20260618`
  - 命令：`python -u train.py --overfit-sanity --seed 56 --run-name sanity_deep3_seed56_20260618 --model-size deep3 --num-workers 0 --cache-in-ram`
  - 结果：epoch 92 通过，loss `< 0.01` 且 exact `1.0`

下一步训练优先级：

1. `local_k5_seed52`
2. `local_k5_seed53`
3. `local_resblock_seed54`
4. `local_resblock_seed55`
5. `local_deep3_seed56`

训练完成后再汇总 `local_wide_seed51_cache_50ep_20260618` 与新架构 checkpoints 做 ensemble/beam 搜索。

### 本地 k5 训练、beam 搜索与 Kaggle 提交（2026-06-18）

本地 `k5 + light + red_char_weight=2.5 + seed52` 训练已完成：

- run-name：`local_k5_seed52`
- 命令：`python -u train.py --epochs 50 --augment --seed 52 --run-name local_k5_seed52 --red-char-weight 2.5 --model-size k5 --augment-preset light --num-workers 0 --cache-in-ram`
- best epoch：`41`
- val exact：`0.9768`
- char acc：`0.9828`
- color acc：`0.99984`
- joint pos acc：`0.98272`
- best checkpoint：`red_char/outputs/runs/local_k5_seed52/checkpoints/best.pt`

等权集成搜索：

- 候选：
  - `local_wide_seed51_cache_50ep_20260618/checkpoints/best.pt`
  - `local_wide_seed51_cache_20ep_20260618/checkpoints/best.pt`
  - `local_wide_seed51_cache_10ep_20260618/checkpoints/best.pt`
  - `local_k5_seed52/checkpoints/best.pt`
- 搜索日志：`red_char/outputs/logs/local_stage2_wide_k5_ensemble_search.csv`
- 最佳组合：`wide50 + wide20 + k5`
- best val exact：`0.9848`

beam 分头加权搜索：

- 命令：`python beam_weight_search.py --checkpoints outputs/runs/local_wide_seed51_cache_50ep_20260618/checkpoints/best.pt outputs/runs/local_wide_seed51_cache_20ep_20260618/checkpoints/best.pt outputs/runs/local_k5_seed52/checkpoints/best.pt --coarse-step 0.1 --fine-step 0.02 --radius 0.08 --top-k 20 --output outputs/logs/local_stage2_wide_k5_beam_weight_search.csv`
- 最优 char weights：`0.52 | 0.24 | 0.24`
- 最优 color weights：`0 | 0.38 | 0.62`
- best val exact：`0.9860`
- char acc：`0.98904`
- color acc：`1.0`
- joint pos acc：`0.98904`
- 搜索日志：`red_char/outputs/logs/local_stage2_wide_k5_beam_weight_search.csv`

submission 生成与格式检查：

- 输出：`submissions/submission_local_stage2_wide50_wide20_k5_beam.csv`
- 根目录提交副本：`submission.csv`
- 预测长度分布：`{1: 1268, 2: 1305, 3: 1234, 4: 1193}`
- 格式检查通过：表头、行数、id 顺序、字符集、无 BOM、无 CRLF、无 nan 均通过
- 备注：脚本基于旧 `submission_sample` 的空标签/长度 5 期望给出 warning；但用户已确认测试集不存在 0 红和 5 红样本，本次长度分布不按该 warning 判失败。

Kaggle 提交结果：

- 命令：`kaggle competitions submit -c verification-red-code -f submission.csv -m "local stage2 wide50 wide20 k5 beam"`
- ref：`53801763`
- status：`SubmissionStatus.COMPLETE`
- publicScore：`0.97820`

结论：

- 这一本地 `wide50 + wide20 + k5` beam 组合没有超过历史 `submission_widemix5_9904.csv` 的 `0.98000`。
- 本地 val 从 `0.9848` 到 `0.9860` 有提升，但 Kaggle public 反而下降，说明单个 k5 seed 暂时没有提供足够测试集泛化收益。
- 下一步不应围绕该组合继续微调权重；更值得继续训练 `resblock/deep3` 或补齐第二个 k5 seed，再与历史 widemix5/服务器 checkpoint 做全量搜索。

### 参考高分分支并迁移 v2hi 主模型（2026-06-18）

用户提示项目中存在他人高分分支可参考，但 `0.98520` 提交属于他人，不作为本方案基线。本次只参考代码思路，不使用他人 submission。

已 fetch 到远端分支：

- `origin/feature/glyph-reranker-98.72`
- 分支报告显示平台最佳 `0.9872`，核心路线为 `v2hi` 高分辨率主模型 + pseudo-label + glyph reranker。
- 该分支会删除本分支的 `timeline.md`、测试与 submission 记录，不能直接 merge；只读参考关键实现。

本分支已按 TDD 迁移最小闭环：

- 先在 `red_char/test_ensemble.py` 增加 `v2hi` 架构和 parser 期望。
- RED 验证失败原因：
  - `build_model("v2hi")` 抛出 `ValueError: unknown model_size: v2hi`
  - `--model-size v2hi` 不在 argparse choices 中
- 实现：
  - `red_char/config.py`：`MODEL_SIZES` 增加 `v2hi`
  - `red_char/model.py`：新增 `SqueezeExcite`、`ResidualSEStage`、`CoordConv2d`、`V2HiRedCharNet`
  - `v2hi` 结构保留 3 次下采样，最终特征图约 `7x25`，再用 `1x1` 降通道后 Flatten，目标是保留更多字符细节。
- GREEN 验证：
  - `python -m unittest test_ensemble.EnsembleTests.test_stage2_model_sizes_preserve_output_shapes test_ensemble.EnsembleTests.test_train_parser_accepts_v2hi_model_size`：通过
  - `python model.py`：通过，`v2hi` 参数量 `7,213,792`
  - `python -m unittest discover -p "test*.py"`：33 项测试通过

v2hi 64 样本过拟合 sanity：

- run-name：`sanity_v2hi_seed61_20260618`
- 命令：`python -u train.py --overfit-sanity --seed 61 --run-name sanity_v2hi_seed61_20260618 --model-size v2hi --num-workers 0 --cache-in-ram`
- 结果：epoch `88` 通过，loss `< 0.01` 且 exact `1.0`

下一步：

- 启动正式 `local_v2hi_seed61` 训练，先用 `40 epochs + light augment + cache_in_ram + num_workers=0`。
- 如果单模型 val 达到或超过 wide seed51，再纳入现有 wide/k5 做 ensemble/beam；否则继续参考高分分支的 red-threshold / glyph reranker。

### v2hi 正式训练、集成与 Kaggle 提交（2026-06-18）

训练脚本补充：

- 新增 `train.py --resume <checkpoint>`，用于从 `last.pt` 继续训练。
- TDD 验证：
  - `test_train_parser_accepts_resume_checkpoint`
  - `test_restore_training_state_loads_checkpoint_payload`
- 验证命令：
  - `python -m unittest test_ensemble.EnsembleTests.test_train_parser_accepts_resume_checkpoint test_ensemble.EnsembleTests.test_restore_training_state_loads_checkpoint_payload`
  - `python -m unittest discover -p "test*.py"`
- 结果：35 项测试通过。

`local_v2hi_seed61` 训练：

- 初始命令：`python -u train.py --epochs 40 --augment --seed 61 --run-name local_v2hi_seed61 --red-char-weight 2.5 --model-size v2hi --augment-preset light --num-workers 0 --cache-in-ram`
- 由于工具 1 小时超时，初始训练只完成到 epoch `12`，best val exact `0.9532`。
- resume 命令：`python -u train.py --epochs 40 --augment --seed 61 --run-name local_v2hi_seed61 --red-char-weight 2.5 --model-size v2hi --augment-preset light --num-workers 0 --cache-in-ram --resume outputs/runs/local_v2hi_seed61/checkpoints/last.pt`
- 训练完成到 epoch `40`。
- best epoch：`37`
- best val exact：`0.9852`
- char acc：`0.98736`
- color acc：`0.99992`
- joint pos acc：`0.98728`
- best checkpoint：`red_char/outputs/runs/local_v2hi_seed61/checkpoints/best.pt`

等权集成搜索：

- 候选：
  - `local_v2hi_seed61/checkpoints/best.pt`
  - `local_wide_seed51_cache_50ep_20260618/checkpoints/best.pt`
  - `local_wide_seed51_cache_20ep_20260618/checkpoints/best.pt`
  - `local_k5_seed52/checkpoints/best.pt`
- 搜索日志：`red_char/outputs/logs/local_stage2_v2hi_wide_k5_ensemble_search.csv`
- 最佳组合：`v2hi + wide50 + k5`
- best val exact：`0.9876`

beam 分头加权搜索：

- 命令：`python beam_weight_search.py --checkpoints outputs/runs/local_v2hi_seed61/checkpoints/best.pt outputs/runs/local_wide_seed51_cache_50ep_20260618/checkpoints/best.pt outputs/runs/local_k5_seed52/checkpoints/best.pt --coarse-step 0.1 --fine-step 0.02 --radius 0.08 --top-k 20 --output outputs/logs/local_stage2_v2hi_wide_k5_beam_weight_search.csv`
- 最优 char weights：`0.36 | 0.38 | 0.26`
- 最优 color weights：`0 | 0.36 | 0.64`
- best val exact：`0.9880`
- char acc：`0.99024`
- color acc：`0.99992`
- joint pos acc：`0.99016`
- 搜索日志：`red_char/outputs/logs/local_stage2_v2hi_wide_k5_beam_weight_search.csv`

红色阈值检查：

- 对阈值 `0.10` 到 `0.50` 扫描，val exact 均为 `0.9880`。
- 结论：这组模型颜色判定稳定，短板仍是字符识别；无需为了本次提交修改 `predict.py` 的红色阈值逻辑。

submission 生成与 Kaggle 结果：

- 输出：`submissions/submission_local_stage2_v2hi_wide_k5_beam.csv`
- 根目录提交副本：`submission.csv`
- 预测长度分布：`{1: 1268, 2: 1306, 3: 1232, 4: 1194}`
- Kaggle 命令：`kaggle competitions submit -c verification-red-code -f submission.csv -m "local stage2 v2hi wide k5 beam"`
- ref：`53806255`
- status：`SubmissionStatus.COMPLETE`
- publicScore：`0.98040`

结论：

- `v2hi` 迁移有效：单模型 val `0.9852`，集成 val `0.9880`，Kaggle public 从本分支历史最佳 `0.98000` 小幅提升到 `0.98040`。
- 距离用户目标 `99+` 仍明显不足。
- 下一步应继续沿高分分支主线推进，而不是调当前 3 模型权重：
  1. 补 `v2hi` 第二/第三 seed，验证是否带来稳定集成收益；
  2. 迁移 glyph 局部 reranker 的最小闭环；
  3. 若时间允许，再做 pseudo-label self-training。

### 第二个 v2hi seed、四模型 beam 与 Kaggle 提交（2026-06-18）

重要基线说明：

- `submission_sample.csv -> 0.98520` 是他人提交，不作为本分支成绩或融合来源。
- 本轮仍只使用本地训练 checkpoint 生成 submission；他人高分分支只作为代码思路参考。

`local_v2hi_seed62` 训练：

- 初始命令：`python -u train.py --epochs 40 --augment --seed 62 --run-name local_v2hi_seed62 --red-char-weight 2.5 --model-size v2hi --augment-preset light --num-workers 0 --cache-in-ram`
- 初始训练因工具 2 小时超时中断，实际已跑到 epoch `36`，checkpoint 正常写入。
- resume 命令：`python -u train.py --epochs 40 --augment --seed 62 --run-name local_v2hi_seed62 --red-char-weight 2.5 --model-size v2hi --augment-preset light --num-workers 0 --cache-in-ram --resume outputs/runs/local_v2hi_seed62/checkpoints/last.pt`
- 训练完成到 epoch `40`。
- best epoch：`39`
- best val exact：`0.9836`
- char acc：`0.98760`
- color acc：`1.00000`
- joint pos acc：`0.98760`
- best checkpoint：`red_char/outputs/runs/local_v2hi_seed62/checkpoints/best.pt`

四模型等权搜索：

- 候选：
  - `local_v2hi_seed61/checkpoints/best.pt`
  - `local_v2hi_seed62/checkpoints/best.pt`
  - `local_wide_seed51_cache_50ep_20260618/checkpoints/best.pt`
  - `local_k5_seed52/checkpoints/best.pt`
- 搜索日志：`red_char/outputs/logs/local_stage2_v2hi2_wide_k5_ensemble_search.csv`
- 最佳组合：四模型全量等权。
- best val exact：`0.9884`
- char acc：`0.99056`
- color acc：`0.99992`
- joint pos acc：`0.99048`

四模型 beam 分头加权搜索：

- 命令：`python beam_weight_search.py --checkpoints outputs/runs/local_v2hi_seed61/checkpoints/best.pt outputs/runs/local_v2hi_seed62/checkpoints/best.pt outputs/runs/local_wide_seed51_cache_50ep_20260618/checkpoints/best.pt outputs/runs/local_k5_seed52/checkpoints/best.pt --coarse-step 0.1 --fine-step 0.02 --radius 0.08 --top-k 20 --output outputs/logs/local_stage2_v2hi2_wide_k5_beam_weight_search.csv`
- 最优 char weights：`0.20 | 0.20 | 0.42 | 0.18`
- 最优 color weights：`0 | 0.34 | 0 | 0.66`
- best val exact：`0.9896`
- char acc：`0.99096`
- color acc：`1.00000`
- joint pos acc：`0.99096`
- 搜索日志：`red_char/outputs/logs/local_stage2_v2hi2_wide_k5_beam_weight_search.csv`

独立 evaluate 复核：

- 命令：`python evaluate.py --checkpoints outputs/runs/local_v2hi_seed61/checkpoints/best.pt outputs/runs/local_v2hi_seed62/checkpoints/best.pt outputs/runs/local_wide_seed51_cache_50ep_20260618/checkpoints/best.pt outputs/runs/local_k5_seed52/checkpoints/best.pt --char-weights 0.2 0.2 0.42 0.18 --color-weights 0 0.34 0 0.66`
- val exact：`0.9896000001907349`
- char acc：`0.9909600052833557`
- color acc：`0.9999999953269958`
- 导出验证错误数：`26`

submission 生成与 Kaggle 结果：

- 输出：`submissions/submission_local_stage2_v2hi2_wide_k5_beam.csv`
- 根目录提交副本：`submission.csv`
- 预测长度分布：`{1: 1268, 2: 1306, 3: 1232, 4: 1194}`
- Kaggle 命令：`kaggle competitions submit -c verification-red-code -f submission.csv -m "local stage2 v2hi2 wide k5 beam"`
- ref：`53809652`
- status：`SubmissionStatus.COMPLETE`
- publicScore：`0.98080`

结论：

- 第二个 `v2hi` seed 让本地 beam 从 `0.9880` 提升到 `0.9896`，Kaggle public 从 `0.98040` 提升到 `0.98080`，方向有效但增幅很小。
- 目前本分支已确认的自有最好 Kaggle public 是 `0.98080`；`0.98520` 属于他人提交，不纳入本分支成绩。
- 距离 `99+` 目标仍有显著差距。下一轮不应继续只堆同类 v2hi seed；优先迁移并本地验证高分分支的 `glyph reranker` 最小闭环，或用高置信 pseudo-label 做 self-training，再提交验证。

### 本地 glyph reranker 最小闭环与 Kaggle 提交（2026-06-19）

目标：

- 继续参考 `origin/feature/glyph-reranker-98.72` 的代码思路，但不使用他人 submission 或 checkpoint。
- 在 main 分支实现一个本地可训练、可评估、可提交的 glyph reranker 最小闭环。

代码实现：

- 新增 `red_char/glyph.py`
  - `extract_glyph_crops()`：从 5 个名义字符位置切局部 crop。
  - `extract_glyph_crop()`：只切单个位置，避免 `GlyphDataset.__getitem__` 每次重复切 5 个 crop。
  - `GlyphDataset`：把全图训练样本展开为 position-level glyph 样本，可选 red-only。
  - `GlyphNet`：局部字符分类器，支持 `input_mode=rgb|red`、`hires`、`head_mode=flat|gap`。
  - `load_glyph_model()` / `glyph_probabilities()`：供评估和预测复用。
- 新增 `red_char/train_glyph.py`
  - 支持 `--run-name`、`--input-mode`、`--hires`、`--head-mode`、`--crop-width`、`--num-workers`、`--resume`、`--augment/--no-augment`。
  - checkpoint 写入 `outputs/runs/<run-name>/checkpoints/{best,last}.pt`。
- 新增 `red_char/eval_reranker.py`
  - 支持四模型主干的 `--char-weights` / `--color-weights`。
  - 提供 `rerank()` 和 `selective_rerank()`。
- 新增 `red_char/predict_reranker.py`
  - 用主模型 ensemble + glyph 模型生成 submission。
  - 复用 `predict.py` 的 submission 写入和格式校验。
- 新增 `red_char/test_glyph_reranker.py`
  - 覆盖 crop shape、单 crop 与批量 crop 一致性、GlyphNet 输出 shape、checkpoint 加载、rerank/selective_rerank 行为、三个脚本 parser。

TDD / 验证：

- 初始 RED：
  - `python -m unittest test_glyph_reranker.py` 因缺少 `glyph` / `eval_reranker` 模块失败。
  - parser 扩展阶段分别因缺少 `predict_reranker.py`、`train_glyph.py`、`--resume`、`--no-augment` 失败。
- GREEN：
  - `python -m unittest test_glyph_reranker.py`：9 项通过。

训练与排障：

- 慢速候选：`local_glyph_seed63_red_hires_gap`
  - 命令：`python -u train_glyph.py --epochs 20 --seed 63 --run-name local_glyph_seed63_red_hires_gap --input-mode red --hires --head-mode gap --crop-width 72 --num-workers 0 --cache-in-ram`
  - 工具 2 小时超时，完整日志只到 epoch `6`。
  - best val_acc：`0.95314`
  - 结论：`red + hires + gap` 太慢且局部精度不足，不继续作为提交候选。
- 性能修复：
  - `GlyphDataset.__getitem__` 从“切 5 个 crop 再取 1 个”改为 `extract_glyph_crop()` 只切目标位置。
  - 无增强 batch 512 单步训练探针约 `0.5s` 级别，可进入正式训练。
- 可用候选：`local_glyph_seed65_red_gap_noaug`
  - 初始命令：`python -u train_glyph.py --epochs 10 --seed 65 --run-name local_glyph_seed65_red_gap_noaug --input-mode red --head-mode gap --crop-width 64 --num-workers 0 --cache-in-ram --batch-size 512 --no-augment`
  - 10 epoch best val_acc：`0.99230`
  - 续训命令：`python -u train_glyph.py --epochs 20 --seed 65 --run-name local_glyph_seed65_red_gap_noaug --input-mode red --head-mode gap --crop-width 64 --num-workers 0 --cache-in-ram --batch-size 512 --no-augment --resume outputs/runs/local_glyph_seed65_red_gap_noaug/checkpoints/last.pt`
  - best epoch：`13`
  - best red-glyph val_acc：`0.99326`
  - 后续 epoch 出现不稳定/退化，提交使用 `best.pt`。

本地 reranker 评估：

- 主模型仍使用四模型 beam：
  - `local_v2hi_seed61`
  - `local_v2hi_seed62`
  - `local_wide_seed51_cache_50ep_20260618`
  - `local_k5_seed52`
- 主模型权重：
  - char：`0.20 | 0.20 | 0.42 | 0.18`
  - color：`0 | 0.34 | 0 | 0.66`
- glyph checkpoint：`red_char/outputs/runs/local_glyph_seed65_red_gap_noaug/checkpoints/best.pt`
- base exact：`2474/2500 = 0.9896`
- alpha rerank 最优：`top_k=2, alpha=0.70`
- rerank exact：`2475/2500 = 0.9900`
- selective rerank 最优：无提升，仍为 `2474/2500 = 0.9896`

submission 生成与 Kaggle 结果：

- 输出：`submissions/submission_local_stage2_glyph_rerank_alpha070.csv`
- 根目录提交副本：`submission.csv`
- 预测长度分布：`{1: 1268, 2: 1306, 3: 1232, 4: 1194}`
- Kaggle 命令：`kaggle competitions submit -c verification-red-code -f submission.csv -m "local stage2 glyph rerank alpha070"`
- ref：`53817507`
- status：`SubmissionStatus.COMPLETE`
- publicScore：`0.98200`

结论：

- 本地 glyph reranker 最小闭环有效：本地 exact 从 `0.9896` 到 `0.9900`，Kaggle public 从 `0.98080` 到 `0.98200`。
- 这仍低于目标 `99+`，但说明局部字符 reranker 比继续堆 v2hi seed 更有 public 收益。
- 下一步优先方向：
  1. 训练第二个 glyph seed，并做 glyph checkpoint ensemble；
  2. 尝试 `all_glyphs + noaug` 或轻量 medium augmentation，但避免已验证过慢的 `hires`；
  3. 在 reranker 上增加 per-position / confusion-group 选择，而不是全局 alpha。

### 第二个 glyph seed 与更保守 rerank 提交（2026-06-19）

目标：

- 延续上一轮有效的 glyph reranker 路线，训练第二个本地 glyph seed。
- 不使用他人 submission 或 checkpoint；`0.98520` 仍不计入本分支成绩。

训练：

- run-name：`local_glyph_seed66_red_gap_noaug`
- 命令：`python -u train_glyph.py --epochs 20 --seed 66 --run-name local_glyph_seed66_red_gap_noaug --input-mode red --head-mode gap --crop-width 64 --num-workers 0 --cache-in-ram --batch-size 512 --no-augment`
- best epoch：`15`
- best red-glyph val_acc：`0.99406`
- checkpoint：`red_char/outputs/runs/local_glyph_seed66_red_gap_noaug/checkpoints/best.pt`

本地 reranker 搜索：

- 主模型仍使用四模型 beam：
  - `local_v2hi_seed61`
  - `local_v2hi_seed62`
  - `local_wide_seed51_cache_50ep_20260618`
  - `local_k5_seed52`
- 主模型权重：
  - char：`0.20 | 0.20 | 0.42 | 0.18`
  - color：`0 | 0.34 | 0 | 0.66`
- base exact：`2474/2500 = 0.9896`
- seed65 单 glyph：
  - best alpha：`top_k=2, alpha=0.70`
  - rerank exact：`2475/2500 = 0.9900`
  - override positions：`31`
- seed66 单 glyph：
  - best alpha：`top_k=2, alpha=0.40`
  - rerank exact：`2475/2500 = 0.9900`
  - override positions：`14`
- seed65 + seed66 glyph ensemble：
  - best alpha：`top_k=2, alpha=0.65`
  - rerank exact：`2475/2500 = 0.9900`
  - override positions：`26`
- selective rerank：三组 glyph 候选均未超过 base exact。

提交选择：

- 因 seed66 用更少 override 达到同样本地 exact，选择提交 seed66 单 glyph 的保守 rerank。
- 暂不提交 seed65+seed66 ensemble，因为本地没有比单 seed 更高，且 override 更多。

submission 生成与 Kaggle 结果：

- 输出：`submissions/submission_local_stage2_glyph66_rerank_alpha040.csv`
- 根目录提交副本：`submission.csv`
- 预测长度分布：`{1: 1268, 2: 1306, 3: 1232, 4: 1194}`
- Kaggle 命令：`kaggle competitions submit -c verification-red-code -f submission.csv -m "local stage2 glyph66 rerank alpha040"`
- ref：`53818428`
- status：`SubmissionStatus.COMPLETE`
- publicScore：`0.98260`

结论：

- 第二个 glyph seed 继续带来 public 提升：`0.98200 -> 0.98260`。
- 本分支当前自有最好 Kaggle public：`0.98260`。
- 距离 `99+` 仍有明显差距，但 glyph reranker 是当前最有效方向。
- 下一步优先尝试：
  1. 再训练一个 no-augment glyph seed，观察 public 是否继续随更保守 rerank 上升；
  2. 训练 `all_glyphs + noaug`，看非红位置字符样本是否能改善字符边界；
  3. 做 confusion-group / per-position rerank 搜索，避免全局 alpha 只改少量位置。

### all-glyph glyph seed67 与额度受限提交尝试（2026-06-19）

目标：

- 按上一轮结论验证 `all_glyphs + noaug`：用全部 5 个字符位扩充 glyph 监督，但验证仍只看红字位。
- 继续排除他人 `submission_sample.csv = 0.98520`，不作为本分支 baseline。

训练：

- run-name：`local_glyph_seed67_all_gap_noaug`
- 命令：`python -u train_glyph.py --epochs 20 --seed 67 --run-name local_glyph_seed67_all_gap_noaug --input-mode red --head-mode gap --crop-width 64 --num-workers 0 --cache-in-ram --batch-size 512 --no-augment --all-glyphs`
- train_glyphs：`237500`
- val_glyphs：`6231`
- best epoch：`18`
- best red-glyph val_acc：`0.99519`
- checkpoint：`red_char/outputs/runs/local_glyph_seed67_all_gap_noaug/checkpoints/best.pt`

本地 reranker 搜索：

- 主模型仍使用四模型 beam：
  - `local_v2hi_seed61`
  - `local_v2hi_seed62`
  - `local_wide_seed51_cache_50ep_20260618`
  - `local_k5_seed52`
- 主模型权重：
  - char：`0.20 | 0.20 | 0.42 | 0.18`
  - color：`0 | 0.34 | 0 | 0.66`
- base exact：`2474/2500 = 0.9896`
- seed67 all-glyph 单模型：
  - best alpha：`top_k=2, alpha=0.70`
  - rerank exact：`2476/2500 = 0.9904`
  - override positions：`36`
- seed66 + seed67 all-glyph ensemble：
  - best alpha：`top_k=2, alpha=1.50`
  - rerank exact：`2478/2500 = 0.9912`
  - override positions：`45`
- seed65 + seed66 + seed67 all-glyph ensemble：
  - best alpha：`top_k=2, alpha=0.70`
  - rerank exact：`2477/2500 = 0.9908`
  - override positions：`29`
- selective rerank：三组 glyph 候选均未超过 base exact。

submission 生成：

- 输出：`submissions/submission_local_stage2_glyph66_glyph67all_rerank_alpha150.csv`
- 根目录提交副本：`submission.csv`
- 预测长度分布：`{1: 1268, 2: 1306, 3: 1232, 4: 1194}`
- CSV 校验：
  - 数据行：`5000`
  - 空标签：`0`
  - 非法字符：`0`
  - 长度分布：`{1: 1268, 2: 1306, 3: 1232, 4: 1194}`

Kaggle 提交尝试：

- 命令：`kaggle competitions submit -c verification-red-code -f submission.csv -m "local stage2 glyph66 glyph67all rerank alpha150"`
- 结果：未生成新 ref。
- Kaggle API 返回：`Submission not allowed: Your team has used its daily Submission allowance (5) today, please try again tomorrow UTC (3.2 hours from now).`
- 当前自有 Kaggle public best 仍为：`53818428 = 0.98260`。

结论：

- 本地 exact 已提升到 `0.9912`，是目前本分支最高本地结果。
- Kaggle 未提交成功的原因是每日提交额度耗尽，不是 CSV 格式错误。
- 额度恢复后优先提交 `submission_local_stage2_glyph66_glyph67all_rerank_alpha150.csv` 对应的根目录 `submission.csv`。

### 参考分支红线增强移植与 seed68 备选（2026-06-19）

背景：

- 只读查看 `origin/feature/glyph-reranker-98.72`，该分支报告显示高分路线中 `red input + GAP + red-line augmentation` 有效。
- 该分支分数与提交不作为本分支成绩，只参考局部实现思路。

源码改动：

- `dataset.TrainAugmentation` 增加 `red_line_p`，在几何增强后、噪声前叠加 1-3 条随机红色干扰线。
- `glyph.GlyphDataset` 增加 `red_line_p`，仅训练集增强时传入；验证集仍不增强。
- `train_glyph.py` 增加 `--red-line-aug`，并写入 glyph checkpoint 元数据。
- `test_glyph_reranker.py` 增加：
  - `TrainAugmentation(red_line_p=1.0)` 红线像素测试；
  - `train_glyph.py --red-line-aug` CLI 解析测试。
- 验证：`python -m unittest test_glyph_reranker`，结果 `10 OK`。

训练：

- run-name：`local_glyph_seed68_red_gap_redline050`
- 命令：`python -u train_glyph.py --epochs 20 --seed 68 --run-name local_glyph_seed68_red_gap_redline050 --input-mode red --head-mode gap --crop-width 64 --num-workers 0 --cache-in-ram --batch-size 512 --red-line-aug 0.5`
- 首次训练 1 小时超时，中断在 epoch 12；随后从 `last.pt` resume 到 epoch 20。
- best epoch：`16`
- best red-glyph val_acc：`0.99535`
- checkpoint：`red_char/outputs/runs/local_glyph_seed68_red_gap_redline050/checkpoints/best.pt`

本地 reranker 搜索：

- base exact：`2474/2500 = 0.9896`
- seed68 redline 单模型：
  - best alpha：`top_k=5, alpha=2.30`
  - rerank exact：`2476/2500 = 0.9904`
  - override positions：`222`
- seed66 + seed68 redline：
  - best alpha：`top_k=3, alpha=2.75`
  - rerank exact：`2476/2500 = 0.9904`
  - override positions：`143`
- seed67 all-glyph + seed68 redline：
  - best alpha：`top_k=2, alpha=1.15`
  - rerank exact：`2478/2500 = 0.9912`
  - override positions：`40`
- seed66 + seed67 all-glyph + seed68 redline：
  - best alpha：`top_k=2, alpha=1.45`
  - rerank exact：`2477/2500 = 0.9908`
  - override positions：`39`
- seed65 + seed66 + seed67 all-glyph + seed68 redline：
  - best alpha：`top_k=2, alpha=0.95`
  - rerank exact：`2476/2500 = 0.9904`
  - override positions：`32`
- selective rerank：均未超过 base exact。

备选 submission：

- 输出：`submissions/submission_local_stage2_glyph67all_glyph68redline_rerank_alpha115.csv`
- 预测长度分布：`{1: 1268, 2: 1306, 3: 1232, 4: 1194}`
- 与当前排队提交 `submission_local_stage2_glyph66_glyph67all_rerank_alpha150.csv` 相比：
  - 测试集差异行数：`12`
  - 均为等长字符替换，无长度变化。

结论：

- seed68 redline 提高了单 glyph val_acc，但未把 rerank exact 推过 `2478/2500`。
- 额度恢复后的优先提交仍建议先交 `glyph66 + glyph67all alpha150`，因为它继承了 seed66 已有 public 正向趋势。
- 若第一条不涨，再提交 `glyph67all + glyph68redline alpha115` 作为同分、较少 override 的备选。

### all-glyph seed69 追加验证（2026-06-19）

目标：

- 利用 Kaggle 每日额度等待窗口，追加一个 `all_glyphs + noaug` seed，验证 all-glyph 路线是否能继续随 seed 增益。

训练：

- run-name：`local_glyph_seed69_all_gap_noaug`
- 命令：`python -u train_glyph.py --epochs 20 --seed 69 --run-name local_glyph_seed69_all_gap_noaug --input-mode red --head-mode gap --crop-width 64 --num-workers 0 --cache-in-ram --batch-size 512 --no-augment --all-glyphs`
- train_glyphs：`237500`
- val_glyphs：`6231`
- best epoch：`15`
- best red-glyph val_acc：`0.99294`
- checkpoint：`red_char/outputs/runs/local_glyph_seed69_all_gap_noaug/checkpoints/best.pt`

本地 reranker 搜索：

- seed69 单模型：
  - best alpha：`top_k=2, alpha=0.05`
  - rerank exact：`2474/2500 = 0.9896`
  - override positions：`2`
- seed67 all-glyph + seed69：
  - best alpha：`top_k=2, alpha=1.20`
  - rerank exact：`2477/2500 = 0.9908`
  - override positions：`38`
- seed66 + seed67 all-glyph + seed69：
  - best alpha：`top_k=2, alpha=2.65`
  - rerank exact：`2478/2500 = 0.9912`
  - override positions：`50`
- seed67 all-glyph + seed68 redline + seed69：
  - best alpha：`top_k=2, alpha=1.00`
  - rerank exact：`2478/2500 = 0.9912`
  - override positions：`33`

结论：

- seed69 单体明显弱于 seed67/68，没有带来超过 `2478/2500` 的本地 exact。
- 不替换当前优先提交的 `glyph66 + glyph67all alpha150`。
- `glyph67all + glyph68redline + glyph69 alpha100` 是低 override 备选，但因包含弱 seed69，优先级低于 `glyph67all + glyph68redline alpha115`。

### Kaggle 额度恢复后的三版提交（2026-06-19）

提交 1：glyph66 + glyph67 all-glyph

- 根目录提交副本：`submission.csv`
- 对应归档：`submissions/submission_local_stage2_glyph66_glyph67all_rerank_alpha150.csv`
- Kaggle 命令：`kaggle competitions submit -c verification-red-code -f submission.csv -m "local stage2 glyph66 glyph67all rerank alpha150"`
- ref：`53823004`
- status：`SubmissionStatus.COMPLETE`
- publicScore：`0.98320`

提交 2：glyph67 all-glyph + glyph68 redline

- 对应归档：`submissions/submission_local_stage2_glyph67all_glyph68redline_rerank_alpha115.csv`
- Kaggle 命令：`kaggle competitions submit -c verification-red-code -f submission.csv -m "local stage2 glyph67all glyph68redline rerank alpha115"`
- ref：`53823111`
- status：`SubmissionStatus.COMPLETE`
- publicScore：`0.98320`

提交 3：glyph67 all-glyph + glyph68 redline + glyph69

- 对应归档：`submissions/submission_local_stage2_glyph67all_glyph68redline_glyph69_rerank_alpha100.csv`
- Kaggle 命令：`kaggle competitions submit -c verification-red-code -f submission.csv -m "local stage2 glyph67all glyph68redline glyph69 rerank alpha100"`
- ref：`53823220`
- status：`SubmissionStatus.COMPLETE`
- publicScore：`0.98260`

当前结论：

- 本分支自有最好 Kaggle public：`0.98320`。
- 最好提交并列：
  - `53823004`：`glyph66 + glyph67all alpha150`
  - `53823111`：`glyph67all + glyph68redline alpha115`
- 低 override 的 `glyph67all + glyph68redline + glyph69 alpha100` 未转化为 public 提升，退回 `0.98260`，说明弱 seed69 平滑并不可靠。
- 根目录 `submission.csv` 已恢复为 `submission_local_stage2_glyph66_glyph67all_rerank_alpha150.csv`。
- 当前仍未达到 `99+` 目标；下一阶段需要比单一 2500 val 更稳的判断，例如 K-fold/OOF 或测试分布适配，而不是继续只按 2500 val 末位提交。

### reranker 主模型 x-tta 负面实验（2026-06-19）

目标：

- 参考 `origin/feature/glyph-reranker-98.72` 的三视图水平 TTA 思路，在本分支 reranker 内独立验证是否改善当前四主模型 + glyph reranker。
- 不复用旧五视图 TTA 路线；第二阶段计划已记录五视图平移 TTA 曾略降。

源码改动：

- `eval_reranker.py` 新增 `average_primary_logits(..., x_tta=True)`：
  - 三个水平视图：`dx = 0, -4, +4`
  - 默认关闭，不改变既有行为。
- `eval_reranker.py` / `predict_reranker.py` 增加 `--x-tta` CLI 参数。
- `predict_reranker.py` 生成 submission 时复用与 eval 相同的 `average_primary_logits`，避免评估/预测不一致。
- `test_glyph_reranker.py` 增加：
  - eval parser 支持 `--x-tta`
  - predict parser 支持 `--x-tta`
  - `average_primary_logits` 确认会跑 3 个水平视图并平均 logits。

验证：

- 目标测试：`python -m unittest test_glyph_reranker.GlyphRerankerTests.test_reranker_eval_parser_accepts_primary_weights test_glyph_reranker.GlyphRerankerTests.test_average_primary_logits_uses_three_horizontal_tta_views test_glyph_reranker.GlyphRerankerTests.test_predict_reranker_parser_accepts_submission_options`
- 结果：`3 OK`

本地 reranker 搜索：

- 主模型仍使用四模型 beam：
  - `local_v2hi_seed61`
  - `local_v2hi_seed62`
  - `local_wide_seed51_cache_50ep_20260618`
  - `local_k5_seed52`
- 主模型权重：
  - char：`0.20 | 0.20 | 0.42 | 0.18`
  - color：`0 | 0.34 | 0 | 0.66`
- `g66 + g67_all + x-tta`：
  - base exact：`2473/2500 = 0.9892`
  - best alpha：`top_k=2, alpha=1.40`
  - rerank exact：`2477/2500 = 0.9908`
  - override positions：`46`
- `g67_all + g68_redline + x-tta`：
  - base exact：`2473/2500 = 0.9892`
  - best alpha：`top_k=2, alpha=0.70`
  - rerank exact：`2475/2500 = 0.9900`
  - override positions：`32`
- `g66 + g67_all + g68_redline + x-tta`：
  - base exact：`2473/2500 = 0.9892`
  - best alpha：`top_k=2, alpha=1.15`
  - rerank exact：`2476/2500 = 0.9904`
  - override positions：`38`
- selective rerank：均未超过 base exact。

结论：

- x-tta 在当前本地验证集是负收益：base 从 `2474` 降到 `2473`，最佳也只到 `2477/2500`。
- 不生成 submission，不消耗 Kaggle 额度。
- 保留 `--x-tta` 作为可复现实验开关，但后续默认不启用。
- 下一步优先转向测试分布适配/伪标签，而不是继续在单一 val 上做小幅 rerank 调参。

### 伪标签测试分布适配入口（2026-06-19）

目标：

- 不把他人的 `submission_sample.csv -> 0.98520` 计入本分支成果。
- 参考其他分支思路只作为代码/方法参考，本分支继续以自有 Kaggle public 为准。
- 新增本地伪标签训练入口，用当前四个自有 teacher checkpoint 给 test 集生成高置信伪标签，再训练 pseudo student，检查是否能提升 ensemble/reranker 泛化。

源码改动：

- 新增 `train_pseudo.py`：
  - `select_confident_pseudo_samples(...)`：仅保留 5 个字符位和 5 个颜色位都超过阈值的 test 样本。
  - `generate_pseudo_labels(...)`：复用当前 teacher ensemble 和 char/color 权重生成伪标签。
  - `run_training(...)`：将真训练集 train split 与 test 伪标签集拼接训练，验证仍只看原始 val split。
  - 支持 `--resume`，中断后可从 `last.pt` 恢复 optimizer/scheduler/epoch。
- 新增 `test_pseudo_training.py`：
  - 覆盖高置信筛选逻辑。
  - 覆盖伪标签训练 CLI 参数，包括 `--resume`。

验证：

- `python -m unittest test_pseudo_training`
- 结果：`2 OK`
- `python -m unittest discover -p "test*.py"`
- 结果：`48 OK`

伪标签保留量预检：

- teacher checkpoints：
  - `local_v2hi_seed61`
  - `local_v2hi_seed62`
  - `local_wide_seed51_cache_50ep_20260618`
  - `local_k5_seed52`
- char weights：`0.20 | 0.20 | 0.42 | 0.18`
- color weights：`0 | 0.34 | 0 | 0.66`
- 阈值扫描：
  - `char=0.92, color=0.90`：`4644/5000`
  - `char=0.90, color=0.90`：`4676/5000`
  - `char=0.88, color=0.88`：`4699/5000`
  - `char=0.85, color=0.85`：`4729/5000`

当前动作：

- 采用较保守的 `char_threshold=0.92, color_threshold=0.90` 启动 `local_pseudo_v2hi_seed70`。
- 若本地验证不能超过当前有效 ensemble/rerank，先不提交 Kaggle。

首轮训练修正：

- 从零训练 `local_pseudo_v2hi_seed70` 完成 1 epoch：
  - `pseudo_count=4645`
  - `val exact=0.0220`
  - 结论：scratch pseudo student 收敛太慢，不适合作为当前冲榜主线。
- `train_pseudo.py` 追加：
  - `--init-checkpoint`：只加载已有模型权重，用于从自有 teacher 微调。
  - `--lr`：支持伪标签微调使用更小学习率。
- 复测：
  - `python -m unittest test_pseudo_training`：`2 OK`
  - `python -m unittest discover -p "test*.py"`：`48 OK`
- 下一步改为从 `local_v2hi_seed61/checkpoints/best.pt` 初始化，低学习率微调。

伪标签微调与提交结果：

- 训练：`local_pseudo_v2hi_seed70_init61`
  - 初始化：`outputs/runs/local_v2hi_seed61/checkpoints/best.pt`
  - 学习率：`0.0002`
  - 阈值：`char=0.92, color=0.90`
  - 伪标签：`4645/5000`
  - 3 epoch best：`epoch=2, val exact=0.9836`
  - 结论：单模型低于原 `local_v2hi_seed61`，只能作为 ensemble 多样性候选。
- 本地 reranker：
  - 主模型：原四模型 + `local_pseudo_v2hi_seed70_init61/best.pt`
  - pseudo 低权重：char `0.05`，color `0.05`
  - char weights：`0.19 | 0.19 | 0.399 | 0.171 | 0.05`
  - color weights：`0 | 0.323 | 0 | 0.627 | 0.05`
  - glyph：`glyph66 + glyph67all`
  - best：`top_k=2, alpha=1.40, exact=2478/2500=0.99120`
  - 其他扫描：
    - pseudo `0.10/0.15/0.20`：均为 `2478/2500`
    - `glyph67all + glyph68redline`：`2478/2500`
    - `glyph66 + glyph67all + glyph68redline`：`2477/2500`
- 生成 submission：
  - `submissions/submission_local_stage2_pseudo5_g66_g67_rerank_alpha140.csv`
  - 根目录 `submission.csv` 已复制为该版本。
- Kaggle 提交：
  - 命令：`kaggle competitions submit -c verification-red-code -f submission.csv -m "local stage2 pseudo5 g66 g67 rerank alpha140"`
  - 实际 CLI：`.kaggle_venv/Scripts/kaggle.exe`
  - ref：`53825047`
  - status：`SubmissionStatus.COMPLETE`
  - publicScore：`0.98280`
- 结论：
  - 伪标签低权重虽然本地达到 `2478`，public 未超过本分支自有最好 `0.98320`。
  - 当前自有最好仍是 `53823004` / `53823111` 的 `0.98320`。
  - 伪标签训练入口保留为可复现实验，但当前不继续围绕 pseudo 权重消耗提交额度。

参考其他分支代码（不计入本分支成绩）：

- 参考分支：`origin/feature/glyph-reranker-98.72`
- 用户已说明该分支及 `0.98520` 相关成绩不是本分支成果；本轮只读参考，不合并、不复用其 submission。
- 观察到的可参考方向：
  - 高分路线依赖更多主模型、OOF/K-fold、选择性 glyph rerank、red-threshold、pair/glyph 专项脚本。
  - 该分支会删除/重写本分支 timeline 与 submissions，不适合直接 merge。
  - 其交接文档也强调 2500 val 在高分段噪声很大，后续应优先 OOF/更大验证，而不是继续凭末位样本反复提交。
- 本轮尝试：
  - 对当前自有四主模型 + `glyph66/glyph67all` 做 selective margin rerank 网格扫描。
  - 该只读扫描 5 分钟未完成，被工具超时终止；无结果，不作为提交依据。
- 后续建议：
  - 若继续第二阶段，应把下一步重点从单 split val 调参转到 OOF/K-fold 评估或全量真标签训练候选，而不是继续提交 `2478` 附近的同分变体。

### selective rerank 高效扫描与 resblock 正式训练（2026-06-19）

目标：

- 继续推进第二阶段计划，不把 `0.98520` 或其他分支成绩计入本分支。
- 先用本地验证筛掉无效 rerank 阈值，避免继续用 Kaggle public 做调参。
- 补齐第二阶段计划中尚未正式训练的 `resblock` 架构多样性模型。

selective rerank 本地扫描：

- 使用现有 `collect_probabilities` / `selective_rerank`，临时脚本改用向量化 exact 统计，避免上一轮字符串循环超时。
- 扫描颜色策略：`argmax` 与 red prob threshold `0.20~0.50`。
- 结果：
  - `orig_g66_g67`：
    - base：`2474/2500`
    - alpha rerank：`2478/2500`，`top_k=2, alpha=1.50`
    - selective：`2474/2500`
  - `orig_g67_g68`：
    - base：`2474/2500`
    - alpha rerank：`2478/2500`，`top_k=2, alpha=1.15`
    - selective：`2474/2500`
  - `pseudo5_g66_g67`：
    - base：`2474/2500`
    - alpha rerank：`2478/2500`，`top_k=2, alpha=1.40`
    - selective：`2474/2500`
- 结论：
  - selective margin rerank 和 red-threshold 在当前自有模型组合上没有本地收益。
  - 不生成 submission，不消耗 Kaggle 提交额度。

resblock 正式训练启动：

- 训练目标：`local_resblock_seed54`
- 参数：
  - `--epochs 50`
  - `--seed 54`
  - `--model-size resblock`
  - `--augment-preset light`
  - `--red-char-weight 2.5`
  - `--num-workers 0`
  - `--no-cache-in-ram`
- wrapper：
  - `red_char/outputs/runs/local_resblock_seed54/run_training.ps1`
  - 使用 `$PSScriptRoot` 定位 `red_char`，避免长命令传参导致后台静默退出。
- 训练完成后再评估是否加入当前四主模型 + glyph reranker；若本地不能超过 `2478/2500`，不提交 Kaggle。

resblock 训练结果与止损：

- 后台 wrapper 结果：
  - `run_training.ps1` 与 `run_training.cmd` 均只写出模型信息后退出，未进入完整训练循环。
  - 前台直接运行 `train.py` 可正常训练，因此本轮改用前台分段 resume。
- smoke check：
  - 命令：`python -u train.py --epochs 1 --augment --seed 54 --run-name local_resblock_seed54_smoke --red-char-weight 2.5 --model-size resblock --augment-preset light --num-workers 0 --no-cache-in-ram`
  - 结果：完成 1 epoch，验证训练入口可用。
- 正式训练：
  - 通过 `--resume outputs/runs/local_resblock_seed54/checkpoints/last.pt` 分段续跑。
  - 已完成到 epoch 18。
  - 当前最好：epoch 17，exact `0.96160`，char_acc `0.97296`，red_acc `0.99960`，color_acc `0.97256`。
- 决策：
  - 该模型明显低于第二阶段计划中进入 ensemble 的本地门槛 `0.975`，也低于现有主模型组合。
  - 本轮停止继续训练 `local_resblock_seed54`，不生成 submission，不提交 Kaggle。
  - 当前自有 Kaggle 最好仍为 `0.98320`；`0.98520` 及其他更高分支不计入本分支成果。
