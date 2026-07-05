#!/usr/bin/env bash
# Data-efficiency sweep: FlutterForm vs MLP at small train fractions.
# Usage: bash scripts/data_efficiency.sh [PY] [DEVICE]
set -e
PY=${1:-python3}
DEV=${2:-auto}
D=data/tierA_50k.npz
O=results_cmp
mkdir -p $O
for f in 0.01 0.03 0.1 0.3; do
  echo "###### data-efficiency frac=$f ######"
  $PY train.py mode=tierA data=$D device=$DEV train.batch=256 model.d=16 \
      train.w_flutter=2 train.max_steps=12000 train.eval_every=3000 \
      train.frac=$f out=$O/ff_de$f | tail -1
  $PY train_baseline.py data=$D device=$DEV train.max_steps=12000 \
      train.frac=$f out=$O/bl_de$f | tail -1
  $PY eval.py data=$D device=$DEV ckpt=$O/ff_de$f/flutterform_tierA.pt \
      baseline=$O/bl_de$f/baseline.pt out=$O/de_${f}.json | tail -3
done
echo "DATA-EFFICIENCY DONE -> results_cmp/de_*.json"
