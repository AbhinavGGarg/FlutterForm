#!/usr/bin/env bash
# Full FlutterForm-vs-baseline comparison on the 50k dataset.
# Produces results_cmp/*.json. Run from repo root on the GPU box.
set -e
D=data/tierA_50k.npz
mkdir -p results_cmp
FF="python3 train.py mode=tierA data=$D device=auto"
BL="python3 train_baseline.py data=$D device=auto"

echo "########## 1. in-distribution baseline ##########"
$BL train.max_steps=12000 out=results_cmp/bl_indist
python3 eval.py ckpt=results_d16/flutterform_tierA.pt data=$D \
    baseline=results_cmp/bl_indist/baseline.pt out=results_cmp/indist.json | tail -8

echo "########## 2. EXTRAPOLATION: train mu<40, test mu>=40 ##########"
$FF train.max_steps=25000 train.batch=256 model.d=16 \
    holdout.col=mu holdout.thresh=40 out=results_cmp/ff_extrap | tail -3
$BL train.max_steps=12000 holdout.col=mu holdout.thresh=40 out=results_cmp/bl_extrap | tail -2
python3 eval.py ckpt=results_cmp/ff_extrap/flutterform_tierA.pt data=$D \
    baseline=results_cmp/bl_extrap/baseline.pt \
    holdout.col=mu holdout.thresh=40 out=results_cmp/extrap.json | tail -8

echo "########## 3. DATA EFFICIENCY (baseline vs FF at small train frac) ##########"
for f in 0.01 0.03 0.1; do
  $BL train.max_steps=8000 train.frac=$f out=results_cmp/bl_f$f | tail -1
done
echo "(FlutterForm data-efficiency runs use train.frac via a follow-up patch)"
echo "DONE -> results_cmp/"
