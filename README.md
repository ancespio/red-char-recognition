# 红色字符识别

机器学习课程大作业问题三：识别验证码图片中按顺序排列的红色字符，生成 Kaggle `submission.csv`。

## 当前方案

- 多任务 CNN：共享卷积主干 + 5 个字符分类头 + 5 个红色判断头。
- 输入：`3x60x200` RGB，像素归一化到 `[0, 1]`。
- 输出：按位置预测字符与红/非红，推理时只拼接预测为红色的位置。

## 本地小测试结果

在 RTX 4060 Laptop GPU 上跑 8 epoch 小训练：

- best epoch: 7
- val red-string exact-match: 95.00%
- char position accuracy: 96.86%
- color position accuracy: 99.94%

当前小训练生成的 submission 仅用于格式自检，不建议提交。正式训练前需重点检查 `predict.py` 打印的预测长度分布，尤其是空串数量和长度 5 数量。

## 使用方式

```powershell
cd red_char
python eda.py
python dataset.py
python model.py
python train.py --overfit-sanity --no-cache-in-ram
python train.py --epochs 30
python train.py --epochs 30 --augment
python train.py --epochs 50 --augment --seed 43 --run-name augment_seed43
python train.py --epochs 50 --augment --seed 47 --run-name red_weight_seed47 --red-char-weight 2.5
python train.py --epochs 50 --augment --seed 49 --run-name wide_seed49 --red-char-weight 2.5 --model-size wide
python evaluate.py
python evaluate.py --checkpoints outputs/checkpoints/best.pt outputs/runs/augment_seed44_clean/checkpoints/best.pt
python ensemble_search.py --checkpoints outputs/checkpoints/best.pt outputs/runs/augment_seed43/checkpoints/best.pt outputs/runs/augment_seed44_clean/checkpoints/best.pt
python weighted_ensemble_search.py --checkpoints outputs/checkpoints/best.pt outputs/runs/augment_seed43/checkpoints/best.pt outputs/runs/augment_seed44_clean/checkpoints/best.pt --step 0.05
python beam_weight_search.py --checkpoints outputs/checkpoints/best.pt outputs/runs/augment_seed44_clean/checkpoints/best.pt outputs/runs/augment_seed45/checkpoints/best.pt outputs/runs/red_weight_seed47/checkpoints/best.pt outputs/runs/red_weight_seed48/checkpoints/best.pt --coarse-step 0.1 --fine-step 0.02 --radius 0.08 --top-k 20
python evaluate.py --checkpoints outputs/checkpoints/best.pt outputs/runs/augment_seed43/checkpoints/best.pt outputs/runs/augment_seed44_clean/checkpoints/best.pt --char-weights 0.45 0.1 0.45 --color-weights 0 0.3 0.7 --tta
python predict.py
python predict.py --checkpoints outputs/checkpoints/best.pt outputs/runs/augment_seed43/checkpoints/best.pt outputs/runs/augment_seed44_clean/checkpoints/best.pt --char-weights 0.45 0.1 0.45 --color-weights 0 0.3 0.7 --tta
```

数据目录 `红色字符识别/` 为只读本地数据，不纳入 Git 仓库。
