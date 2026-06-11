# 红色字符识别（课程大作业问题三）— 开发计划

> 本计划由 Claude Code 规划，交由 Codex 实现，最后由 Claude Code 检查验收。
> 实现过程中如与本计划冲突，以本计划的验收标准为准；实现细节可灵活。

## Context（背景）

机器学习课程大作业问题三：给定 200×60 RGB 验证码式图片，每张恰好 5 个字符（字符集 0-9 + A-Z 共 36 类），每个字符位有红/非红属性。任务是输出图中**按顺序排列的所有红色字符组成的字符串**（长度 0~5），生成 submission.csv 提交 Kaggle，评价指标为整串 exact-match 准确率。

- 数据目录（**只读，绝不修改**）：`D:\Learing\一路北航\机器学习\ML-TeamHW\红色字符识别\`
  - `train\images\`：50,000 张 PNG（00000.png~49999.png）
  - `train\labels.csv`：表头 `filename,color,all_label`，如 `00000.png,ruuur,DPVKD`（r=红, u=非红）
  - `test\images\`：5,000 张 PNG（00000.png~04999.png）
  - `submission_sample.csv`：表头 `id,label`；**空 label 写作 `00034.png,`（裸逗号，无引号无 nan）**，LF 行尾
- 环境：Windows 11 本地，NVIDIA GPU，PyTorch，标准 Python 脚本项目（非 notebook）。
- 代码放新建目录：`D:\Learing\一路北航\机器学习\ML-TeamHW\red_char\`

### 数据探查已确认的事实（可直接信任）

- 每位红色比例均 ≈ 50%，红字数分布（训练）1~4 个各 ≈12.5K，**无类别不均衡问题**。
- 训练集没有 0 红 / 5 红样本，但测试集（由 sample 长度分布推断）有 0 红 ≈150、5 红 ≈4 → 按位独立预测天然可外推。
- 图像有浅色噪点背景、**贯穿全图的彩色干扰线（含偏红线条）**、字符位置/大小/旋转有抖动、存在橙色非红字符 → 红色判定必须由网络学习，**HSV 阈值规则不可靠**；**不要硬编码 5 等分切片**。
- 36 类字符训练集全覆盖。

## 模型方案（已定，勿改路线）

**多任务 CNN**：共享主干 + 5×36 字符分类头 + 5×2 红色判断头。理由：不浪费任何监督信号（非红位的字符标签也参与训练），形状/颜色解耦，定长 5 字符场景下远优于 CRNN/CTC。

### 网络结构（约 6.4M 参数）

输入 `3×60×200`，像素 /255 归一化（从零训练，无需 ImageNet 统计量）。

```
Backbone（4 个卷积块，每块 = [Conv3x3-BN-ReLU]×2 + MaxPool2d(2)）：
  Block1:   3 →  32   → 32×30×100
  Block2:  32 →  64   → 64×15×50
  Block3:  64 → 128   → 128×7×25
  Block4: 128 → 256   → 256×3×12
Neck:  Flatten(9216) → Linear(9216,512) + BN1d + ReLU + Dropout(0.3)
Heads: char_head  Linear(512,5×36) → view(B,5,36)
       color_head Linear(512,5×2)  → view(B,5,2)
```

**禁止全局平均池化压掉空间维**（按位预测依赖位置信息），必须 Flatten。

### 损失与超参初值

- `loss = mean_i CE(char_i) + λ·mean_i CE(color_i)`，λ=1.0（若误差分析显示颜色错误是瓶颈，λ→2.0 重训）。不加 label smoothing。
- AdamW lr=1e-3, weight_decay=1e-4；CosineAnnealingLR(T_max=epochs, eta_min=1e-5)，无 warmup。
- 30 epochs，batch 256，AMP 开启，按 **val exact-match** 存 best checkpoint。
- 增强：**首轮不加任何增强**建基线；若过拟合再加 RandomAffine(translate≤5%, degrees≤3) + 轻微高斯噪声。**绝对禁止 hue/saturation 抖动、通道交换、灰度化**（红色是标签信号）。

### 推理规则

第 i 位取 `argmax(char_i)`；若 `argmax(color_i)==红` 则计入，按位置序拼接 → 字符串（可为空）。

## 项目文件结构

```
red_char\
  requirements.txt   # torch(CUDA 版，注明官方 index-url 安装命令)、torchvision、pandas、numpy、Pillow、matplotlib、tqdm
  config.py          # 唯一来源：数据路径(pathlib)、CHARSET、SEED=42、VAL_RATIO=0.05、超参、device、num_workers
  eda.py             # 数据断言 + 样例拼图 → outputs/eda/
  dataset.py         # RedCharDataset + encode_labels/decode_prediction + 确定性划分 + build_dataloaders()
  model.py           # RedCharNet
  train.py           # 训练循环（验证/checkpoint/CSV 日志），含 --overfit-sanity 模式
  predict.py         # best.pt → 推理 test → outputs/submission.csv + 内置格式自检
  evaluate.py        # 验证集三指标 + 错误分类导出
  outputs\           # 运行时生成：checkpoints/ logs/ eda/ submission.csv
```

- `config.py` 是路径/字符集/超参的唯一来源，其余文件只 import。
- `encode_labels(color,all_label)→(char[5],color[5])` 与 `decode_prediction` 必须互逆。
- 训练/评测的验证集划分调用**同一个确定性函数**（固定 seed Generator），保证两处拿到完全相同的 2,500 张。
- 写入文件一律 UTF-8 无 BOM。

## 分阶段实施步骤（每阶段验收不过，禁止进入下一阶段）

### 阶段一：环境 + EDA

- 建 `red_char/`、requirements.txt、config.py；eda.py 做全量断言（50,000 行标签、color 仅 r/u 长 5、all_label 仅字符集长 5、抽样 500 张图全为 200×60 RGB），输出红色比例/红字数分布/36 类频次，随机 16 张拼图存 `outputs/eda/samples.png`；**以二进制读 submission_sample.csv 打印空 label 行原始字节**，确认 `id,` 格式。
- 验收：断言全过；`torch.cuda.is_available()==True`；统计与上文"已确认事实"一致。

### 阶段二：Dataset / DataLoader

- 实现编码/解码、Dataset（train 返回 `(img[3,60,200] float, char[5], color[5])`，test 返回 img+文件名）、95/5 确定性划分、build_dataloaders。
- Windows 注意：所有入口逻辑包在 `if __name__=="__main__":`；`num_workers=4, persistent_workers=True, pin_memory=True`，config 可一键降 0；建议实现 `cache_in_ram=True`（5 万张 ≈180MB，全载内存消除 IO）。
- 验收（写进 `python dataset.py` 自测块）：
  1. 编解码断言 ≥5 组：`("ruuur","DPVKD")→"DD"`、`("uuuuu",·)→""`、`("rrrrr","AB12C")→"AB12C"` 等；
  2. 一个 batch shape `[256,3,60,200]` float32、值域 [0,1]，targets `[256,5]`；
  3. 划分 47500/2500，两次调用文件名列表 hash 完全一致；
  4. 完整遍历一个 epoch 不报错（验证 Windows 多进程）。

### 阶段三：模型 + 小批量过拟合检查

- 实现 RedCharNet；train.py 提供 `--overfit-sanity`：固定 64 样本、dropout=0、训练 ~300 step。
- 验收：
  1. `RedCharNet()(randn(2,3,60,200))` 输出 `(2,5,36)` 与 `(2,5,2)`，参数量 5M~8M；
  2. 64 样本总 loss <0.01 且红色字符串 exact-match=100%。**不达标 = 有 bug，禁止进入阶段四。**

### 阶段四：完整训练

- 固定全部随机种子（random/numpy/torch）；每 epoch 在验证集算三指标（exact-match / 字符位 acc / 颜色位 acc），写 `outputs/logs/train_log.csv`；存 best.pt（含 state_dict、epoch、指标、config 快照）与 last.pt。
- 验收：
  1. 无 NaN，loss 持续下降；
  2. **val exact-match ≥95%**（预期 98%+；<90% 视为有 bug，按风险表排查而非盲目调参）；
  3. 交叉校验 exact-match ≈（按位联合正确率）^5。

### 阶段五：推理与 submission

- **以 submission_sample.csv 的 id 列为推理清单**（不要 listdir）；`model.eval()+no_grad()` batch 推理；`df.to_csv(index=False, lineterminator="\n")`，label 保持空字符串而非 NaN。
- 验收（内置自检函数，全过才提示成功）：
  1. 5001 行（含表头），表头 == `id,label`；
  2. id 列与样例逐行一致（含顺序）；
  3. 二进制读回：空预测行形如 `00123.png,`，无 `nan`、无引号、无 CRLF、无 BOM；
  4. 所有 label 仅含 36 字符集且长度 ≤5；
  5. 打印预测长度分布，与样例量级吻合（0:≈150, 5:≈4）；0 红预测数为 0 或 >500 则告警。

### 阶段六：本地评测与误差分析

- evaluate.py：best.pt + 同一划分函数 → 三指标；错误样本分类（仅字符错/仅颜色错/都错），导出 `outputs/eda/val_errors.csv`，可选错误样本拼图。
- 验收：数字与 train_log.csv 中 best epoch 完全一致（证明划分确定性与 checkpoint 链路正确）。
- 迭代决策树（不满意时一次只改一项）：① 颜色错为主→λ=2.0；② train-val gap 大→开轻量增强；③ 欠拟合→通道翻倍或 50 epochs；④ 顽固错误→3 个 seed 模型 logits 平均（predict.py 支持 checkpoint 列表）。

## 风险点与应对

| # | 风险 | 应对 |
|---|---|---|
| 1 | 偏红干扰线误导颜色头（已确认存在） | 让 CNN 学笔画颜色；颜色错误占比高则 λ=2.0 |
| 2 | 红与橙/紫边界模糊（已确认有橙色非红样本） | 边界由数据定义，模型学习；严禁色相类增强 |
| 3 | **pandas NA 陷阱**：label 可能恰好是 `NA`/`NAN` 被 read_csv 当缺失值 | 所有读 csv 一律 `dtype=str, keep_default_na=False`；写前断言无 NaN |
| 4 | Windows DataLoader 多进程问题 | main 保护；num_workers 可降 0；cache_in_ram 选项 |
| 5 | 训练/评测划分不一致致指标虚高 | 划分单一来源 + hash 比对 |
| 6 | id 排序/遗漏 | 复用样例 id 列作推理清单 |
| 7 | BN 模式遗漏 | 验证/推理统一封装 eval+no_grad |

## 验证方式（端到端）

1. 编解码互逆 5 组手工算例（阶段二）；
2. 64 样本过拟合一票否决检查（阶段三）；
3. 划分 hash 比对（阶段二/六）；
4. 三指标数学关系交叉校验（阶段四）；
5. submission 字节级自检（阶段五）；
6. evaluate 复现 best epoch 指标（阶段六）；
7. 最终人工抽查 5 张测试图 + 预测直觉核对，再提交 Kaggle；同分布合成数据，公榜应与本地 val 基本一致。

## Codex 执行顺序 TODO

1. [ ] 建 `red_char/` + requirements.txt + config.py，验证 CUDA
2. [ ] eda.py 跑通并核对全部统计数字
3. [ ] dataset.py 过阶段二 4 项验收
4. [ ] model.py + train.py 骨架，过 64 样本过拟合检查
5. [ ] 全量训练 30 epochs，val exact-match ≥95%
6. [ ] predict.py 生成 submission.csv，过 5 项格式自检
7. [ ] evaluate.py 误差分析，按决策树决定是否迭代
8. [ ] （由义人手动）提交 Kaggle，对比公榜与本地分数
