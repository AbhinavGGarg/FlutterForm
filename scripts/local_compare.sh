#!/usr/bin/env bash
# Full FlutterForm-vs-MLP comparison matrix.
# Usage: bash scripts/local_compare.sh [PY] [STEPS] [WF] [DEVICE]
set -e
PY=${1:-.venv/bin/python}
STEPS=${2:-20000}
WF=${3:-2.0}
DEV=${4:-cpu}
D=data/tierA_50k.npz
O=results_cmp
mkdir -p $O
FF="$PY train.py mode=tierA data=$D device=$DEV train.batch=256 model.d=16 train.w_flutter=$WF train.eval_every=2000"
BL="$PY train_baseline.py data=$D device=$DEV"
EV="$PY eval.py data=$D device=$DEV"

echo "###### in-distribution ######"
$FF train.max_steps=$STEPS out=$O/ff_indist | tail -2
$BL train.max_steps=12000 out=$O/bl_indist | tail -1
$EV ckpt=$O/ff_indist/flutterform_tierA.pt baseline=$O/bl_indist/baseline.pt out=$O/indist.json | tail -6

echo "###### extrapolation: train mu<40, test mu>=40 ######"
$FF train.max_steps=$STEPS holdout.col=mu holdout.thresh=40 out=$O/ff_extrap | tail -2
$BL train.max_steps=12000 holdout.col=mu holdout.thresh=40 out=$O/bl_extrap | tail -1
$EV ckpt=$O/ff_extrap/flutterform_tierA.pt baseline=$O/bl_extrap/baseline.pt \
    holdout.col=mu holdout.thresh=40 out=$O/extrap.json | tail -6

echo "###### data efficiency ######"
for f in 0.02 0.1; do
  $FF train.max_steps=$STEPS train.frac=$f out=$O/ff_de$f | tail -1
  $BL train.max_steps=12000 train.frac=$f out=$O/bl_de$f | tail -1
  $EV ckpt=$O/ff_de$f/flutterform_tierA.pt baseline=$O/bl_de$f/baseline.pt out=$O/de_${f}.json | tail -2
done

$PY scripts/summarize.py $O
