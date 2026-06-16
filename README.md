# 红色字符识别

机器学习课程大作业问题三：识别验证码图片中按顺序排列的红色字符，生成 Kaggle `submission.csv`。

## 当前方案

- 多任务 CNN：共享卷积主干 + 5 个字符分类头 + 5 个红色判断头。
- 输入：`3x60x200` RGB，像素归一化到 `[0, 1]`。
- 输出：按位置预测字符与红/非红，推理时只拼接预测为红色的位置。

## 改进版（默认开启，相对基线提升 exact-match）

在保持原多任务框架与提交格式不变的前提下，做了以下有依据的增强，全部可通过
`config.py` 或命令行开关切换，方便做对照实验（`exact ≈ 联合位准确率^5`，所以字符
准确率是瓶颈，改进围绕它展开）：

1. **更强主干 `RedCharNetV2`（`config.MODEL="v2"`）**：残差 + Squeeze-Excite 通道
   注意力 + CoordConv 坐标通道。残差改善梯度与字符形状特征；SE 帮助颜色头区分红 vs
   橙/紫干扰；CoordConv 注入绝对列位置，正好契合"读第 i 个字符"的按位预测。输出接口、
   Flatten neck 与原版完全一致，可一键切回 `v1` 做基线对比。
2. **在线数据增强（仅训练集）**：小幅平移/缩放/旋转 + 轻微高斯噪声。**严禁色相/饱和度/
   通道交换/灰度化**——红色就是监督信号。增强即时生成、缓存里只存干净图，验证/测试不增强。
3. **权重 EMA**：用 EMA 平滑权重评测并保存 `best.pt`，几乎零成本稳定涨点。
4. **训练细节**：线性 warmup + 余弦退火、梯度裁剪、字符头轻量 label smoothing(0.05，
   颜色头不平滑)、A100 上启用 TF32。
5. **推理集成**：`predict.py` 支持多 checkpoint（多 seed）softmax 概率平均；默认优先
   加载 EMA 权重。
6. **高分辨率局部重排（当前验证集最佳 0.9928）**：`glyph.py` 从五个槽位提取重叠
   `60x64` 裁切，局部模型保留 `15x16` 特征图，只在主集成 Top-3 候选中选择性重排。
   `eval_reranker.py` 同时验证了小幅水平 `-4/+4px` TTA 对当前训练增强是安全的。

> 注：`overfit-sanity` 自动关闭增强/EMA/label smoothing/dropout，保证纯记忆能力检查时
> 交叉熵可降到 0。

## 使用方式

```bash
cd red_char
python eda.py
python dataset.py
python model.py                       # 自检 v1/v2 形状与参数量
python train.py --overfit-sanity --no-cache-in-ram   # 一票否决：必须 exact=1.0
python train.py --epochs 40           # 默认 v2 + 增强 + EMA
python evaluate.py                    # 误差分析（默认用 EMA 权重）
python predict.py                     # 生成 submission（默认 best.pt，EMA）

# 对照基线（原始 v1，无增强/EMA）：
python train.py --epochs 30 --model v1 --no-augment --no-ema

# 多 seed 集成推理：
python predict.py --checkpoints outputs/checkpoints/best_seed1.pt outputs/checkpoints/best_seed2.pt

# 局部模型训练、验证和最终提交的完整命令见 ../HANDOFF.md。
```

数据目录 `红色字符识别/` 为只读本地数据，不纳入 Git 仓库。需要支持 `torchvision`
（`transforms.v2`）做在线增强。
