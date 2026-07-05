#!/usr/bin/env bash
# Data-efficiency sweep: FlutterForm vs MLP at small train fractions.
# Usage: bash scripts/data_efficiency.sh [PY] [DEVICE]
set -e
PY=${1:-python3}
DEV=${2:-auto}
D=data/tierA_50k.npz
O=results_cmp
mkdir -p $O
for f in 0.02 0.1 0.5; do
  echo "###### data-efficiency frac=$f ######"
  $PY train.py mode=tierA data=$D device=$DEV train.batch=256 model.d=20 \
      train.w_flutter=2 train.w_lowv=0.6 train.max_steps=12000 train.eval_every=6000 \
      train.frac=$f out=$O/ff_de$f
  $PY train_baseline.py data=$D device=$DEV train.max_steps=12000 \
      train.frac=$f out=$O/bl_de$f
  $PY eval.py data=$D device=$DEV ckpt=$O/ff_de$f/flutterform_tierA.pt \
      baseline=$O/bl_de$f/baseline.pt out=$O/de_${f}.json
  echo "== de_$f done: $(grep -o '\"vf_median_%\": [0-9.]*' $O/de_${f}.json | head -2 | tr '\n' ' ')"
done
echo "DATA-EFFICIENCY DONE -> results_cmp/de_*.json"
