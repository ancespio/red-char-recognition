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
python evaluate.py
python predict.py
```

数据目录 `红色字符识别/` 为只读本地数据，不纳入 Git 仓库。
