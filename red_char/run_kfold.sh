#!/bin/bash
# Train per-fold OOF models: 1 v2hi primary + 1 glyph per fold (5 folds).
# Wave 1: 5 v2hi (GPU0-4) + 3 glyph (GPU5-7). Wave 2: 2 glyph (GPU0-1).
set -u
cd /home/duxuanzheng/homework/机器学习/大作业/red-char-recognition/red_char
PY=/home/duxuanzheng/.conda/envs/red_char/bin/python
CAP="OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6"
L=outputs/logs

launch_v2hi(){ # gpu fold
  env $CAP CUDA_VISIBLE_DEVICES=$1 setsid $PY -u train.py --model v2hi --epochs 40 \
    --fold $2 --seed 1 --tag _f$2s1 > $L/kfold_f$2s1.log 2>&1 < /dev/null & }
launch_glyph(){ # gpu fold
  env $CAP CUDA_VISIBLE_DEVICES=$1 setsid $PY -u train_glyph.py --epochs 20 \
    --fold $2 --seed 1 --tag _gff$2 > $L/kfold_gff$2.log 2>&1 < /dev/null & }

echo "=== wave 1: 5 v2hi + 3 glyph ==="
launch_v2hi 0 0; launch_v2hi 1 1; launch_v2hi 2 2; launch_v2hi 3 3; launch_v2hi 4 4
launch_glyph 5 0; launch_glyph 6 1; launch_glyph 7 2
wait
echo "=== wave 2: remaining 2 glyph ==="
launch_glyph 0 3; launch_glyph 1 4
wait
echo "=== all kfold training done ==="
ls -1 outputs/checkpoints/best_f*s1.pt outputs/checkpoints/best_gff*.pt 2>/dev/null
