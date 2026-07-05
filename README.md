# FlutterForm

**An eigen-structured modal-coupling transformer for interpretable, differentiable
flutter prediction & design.** Exea Labs ML-lab sprint project (Abhinav Garg).

Flutter — the aeroelastic instability where two structural modes coalesce and
oscillations grow unbounded — enters the design loop today as an opaque,
non-differentiable gate. FlutterForm hard-wires the *eigenstructure of flutter
itself* into a tiny glass-box model: mode tokens → pairwise outer-product
coupling attention that **is** the aerodynamic coupling operator Q(k) → a
differentiable p-k eigen-solve whose spectrum **is** the answer (flutter speed,
frequency, and which modes coalesce — with no mechanism labels).

## Status

**➡️ Full results and honest limitations: [RESULTS.md](RESULTS.md).** Headline: FlutterForm (6.6k params) loses to a black-box MLP *in-distribution* but **extrapolates ~3× better** to unseen mass ratios, recovers the coalescence mechanism (~73%), predicts the full flutter diagram, and enables **+37% p-k-verified inverse design** by backprop.

| Component | State |
|---|---|
| Physics core (Theodorsen, typical section, p-k, k-method) | ✅ built + **validated** (H&P 0.9%) |
| Tier-A dataset generator (`data/generate_pk.py`) | ✅ built + run (50k sections) |
| FlutterForm model (tokenizer → coupling → eigen head) | ✅ built, **6,620 params** (d=20), trains |
| `train.py mode=tierA` (smoke contract `train.max_steps=1`) | ✅ runs clean on CPU / MI300X |
| Capacity-matched MLP baseline + full eval suite | ✅ done (`eval.py`, `baseline.py`) |
| In-distribution + **extrapolation** results | ✅ done (extrapolation = the headline) |
| Data-efficiency curves | 🔄 running |
| Operator-consistency (learned Q vs Theodorsen) | ✅ done (honest negative — §7 of RESULTS) |
| **Inverse-design demo** (backprop the flutter boundary) | ✅ **+37% p-k-verified** |
| AGARD 445.6 external validation | ✅ reduced-model check (trend yes, magnitude over-predicts, transonic out of scope) |
| Tier-B 3-D wings (assumed modes + strip theory / DLM) | ⏳ future work |

**Physics validation** (`pytest tests/ && python scripts/validate_physics.py`):

- Theodorsen C(k) matches the classical table (C(0.1) = 0.8320 − 0.1723i) and limits.
- Static aero limit is singular exactly at the closed-form divergence speed.
- Two *independent* flutter solutions — p-k and classical k-method — agree at
  the flutter point to <0.5% across parameter space (they share no code path
  beyond the aero matrix).
- Literature anchor: reproduces the published Hodges & Pierce typical-section
  flutter point (V_F/bω_θ = 2.165, ω_F/ω_θ = 0.6545) to **0.9%**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/ -q                          # 20 gates: physics + model
python scripts/validate_physics.py        # literature anchor report
python data/generate_pk.py --n 400 --seed 42 --out data/tierA_dev.npz
python train.py mode=tierA train.max_steps=1     # smoke (proposal contract)
python train.py mode=tierA train.max_steps=300   # short real run (CPU, ~8 s)
```

## How it works

```
section params (μ, σ, x_θ, a, r², M∞)
   │
   ▼
ModalTokenizer            one token per structural mode (plunge, pitch)
   │
   ▼
EigenCouplingAttention    Ĝ_ij(k) = Σ_f φ_f(k) · (t_i ᵀ W_f t_j)   ← the outer
   │                      product between mode tokens IS the learned aero
   │                      coupling operator; φ(k) spans the physical
   │                      circulatory envelope (1/k, 1/k² — geometric prior)
   ▼
DifferentiablePK          G(k,V) = (kV)² Ĝ(k)  (exact physical V-scaling)
   │                      closed-form 2×2 eig, unrolled k fixed point —
   │                      no torch.linalg.eig, no CUDA/ROCm kernel deps
   ▼
p(V) per branch  →  V-g / V-f trajectories  →  flutter speed, frequency,
                    coalescing mode pair — and gradients through all of it
```

Key design decisions:

- **Structure is known, aero is learned.** Ms, Ks are assembled exactly from
  the section parameters; the network's entire job is the aerodynamic
  coupling operator — the object the consistency theorem targets.
- **Exact velocity scaling.** Every Theodorsen term scales as ω² once
  V = ωb/k, so the model predicts the frequency-factored Ĝ(k) and the head
  applies (kV)² analytically. The network never has to learn the trivial part.
- **Closed-form spectral readout.** 2×2 eigenvalues via trace/determinant:
  fully differentiable, batched, and portable (CPU / CUDA / ROCm identical).

## Layout

```
flutterform/physics/   theodorsen.py · section.py · pk.py · kmethod.py
flutterform/model/     tokenizer.py · coupling.py · eigenhead.py · net.py
flutterform/data.py    dataset wrapper over generated shards
data/generate_pk.py    Tier-A ground-truth generator
scripts/               validate_physics.py
tests/                 test_physics.py · test_model.py   (20 gates)
train.py               key=value config, smoke: mode=tierA train.max_steps=1
```

## GPU notes (CUDA or ROCm)

Pure PyTorch — no flash-attn, no xformers, no custom Triton kernels, no
`torch.linalg.eig` in the training path (closed-form 2×2 instead). The eigen
head runs identically on CPU, CUDA, and ROCm wheels; `train.py` defaults to
`device=auto` (CUDA if available). Dataset generation is CPU-bound
numpy/scipy and parallelizes across cores (`--workers`, seed-deterministic
regardless of worker count).

## License

MIT.
