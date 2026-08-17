# Three-seed, leakage-resistant 8k confirmation

## Result

The follow-up does not support a corpus-independent claim that Double
Attention is better than MHA4. It produces a repeatable improvement on
WikiText-2, but MHA4 has the best final loss on Python standard-library code.
The code learning curves explain why the earlier 2k screen disagreed: both
Double Attention variants learn faster initially, then are overtaken between
4k and 6k steps.

| corpus | model | validation loss | paired vs. MHA4 | test loss | paired vs. MHA4 |
|---|---|---:|---:|---:|---:|
| WikiText-2 | MHA4 | 3.574126 ± 0.008899 | 0 | 3.615561 ± 0.009814 | 0 |
| WikiText-2 | QK4-S4 untied | **3.536640 ± 0.003566** | **-0.037486 ± 0.007424** | **3.575900 ± 0.003238** | **-0.039661 ± 0.006608** |
| WikiText-2 | QK2-S2 tied | 3.545569 ± 0.004098 | -0.028557 ± 0.009515 | 3.585082 ± 0.003042 | -0.030479 ± 0.006966 |
| Python code | MHA4 | **2.345204 ± 0.014014** | 0 | **3.096425 ± 0.026286** | 0 |
| Python code | QK4-S4 untied | 2.371795 ± 0.009172 | +0.026591 ± 0.005843 | 3.181250 ± 0.023323 | +0.084825 ± 0.035738 |
| Python code | QK2-S2 tied | 2.381245 ± 0.007840 | +0.036041 ± 0.015916 | 3.180323 ± 0.013896 | +0.083899 ± 0.035567 |

Values are mean ± sample standard deviation over seeds 0, 1, and 2. Deltas
are paired within a seed; negative loss is better. Both Double Attention
variants win all three WikiText seeds on both validation and test and lose all
three Python-code seeds on both validation and test.

## Parameter and training cost

| model | parameters | vs. MHA4 | mean 8k training time | relative time | peak allocated |
|---|---:|---:|---:|---:|---:|
| MHA4 | 16,838,656 | 0 | 4.137 min | 1.00x | 319.149 MiB |
| QK4-S4 untied | 16,969,776 | +131,120 (+0.779%) | 9.525 min | 2.30x | 348.038 MiB |
| QK2-S2 tied | 16,969,752 | +131,096 (+0.779%) | 8.974 min | 2.17x | 331.756 MiB |

Training time is the mean of the logged training chunks over both corpora and
all three seeds. It excludes the final robust validation and test passes. Peak
memory is PyTorch's maximum allocated CUDA memory, not whole-device usage.
These runs used the original forward-only Triton path, whose backward
recomputed the PyTorch reference equations, while MHA4 used fused SDPA. Native
Triton backward was implemented after these runs, so this historical table must
not be used as a benchmark of the current kernel.

## Per-seed final deltas

| corpus | model | seed 0 val/test | seed 1 val/test | seed 2 val/test |
|---|---|---:|---:|---:|
| WikiText-2 | QK4-S4 untied | -0.035998 / -0.037451 | -0.045541 / -0.047090 | -0.030918 / -0.034442 |
| WikiText-2 | QK2-S2 tied | -0.026201 / -0.026768 | -0.039028 / -0.038515 | -0.020441 / -0.026154 |
| Python code | QK4-S4 untied | +0.024209 / +0.093684 | +0.022315 / +0.115300 | +0.033249 / +0.045491 |
| Python code | QK2-S2 tied | +0.028696 / +0.053480 | +0.025123 / +0.123005 | +0.054303 / +0.075212 |

Each cell is `validation delta / test delta` relative to the MHA4 run with the
same seed.

## Learning-curve reversal on code

| model | 2k mean delta (wins) | 4k mean delta (wins) | 6k mean delta (wins) | 8k mean delta (wins) |
|---|---:|---:|---:|---:|
| QK4-S4 untied | -0.030311 (3/3) | +0.004749 (1/3) | +0.029806 (0/3) | +0.027602 (0/3) |
| QK2-S2 tied | -0.029371 (3/3) | +0.020861 (1/3) | +0.043891 (0/3) | +0.030300 (0/3) |

This controlled run reproduces the old Python-code advantage at the 2k
checkpoint, despite using new leakage-resistant splits. The sign then changes.
The previous cross-corpus screen stopped at 2k, so its conclusion described
early convergence rather than final 8k generalization. Because the tokenizer
and splits also changed, absolute losses from the two protocols should not be
compared directly.

WikiText-2 does not show this reversal: from 500 through 8,000 steps, both
Double Attention variants beat MHA4 in all three seeds at every logged
checkpoint. At 8k, the mean single-traversal deltas are -0.044214 for QK4-S4
and -0.038662 for tied QK2-S2.

## Protocol

- two corpora: WikiText-2 raw and Python 3.12 standard-library source;
- explicit train, validation, and held-out test token files;
- official WikiText splits;
- deterministic Python file split by `sha256(relative_path) mod 20`: 18
  buckets train, one validation, one test;
- shared SentencePiece unigram vocabulary of 2,048 pieces, trained only on one
  million characters from each training split;
- seeds 0, 1, and 2;
- width 512, six layers, FFN width 1536, sequence length 64;
- BF16, effective batch 8, 600-step warmup, maximum learning rate `6e-4`;
- 12,000-step cosine schedule stopped at 8,000 steps;
- 4,096,000 training tokens per run;
- robust validation and test losses are each means of two deterministic
  64-batch random-sample evaluations with different RNG seeds;
- NVIDIA GeForce MX570 A; 18 completed runs.

| corpus | train tokens | validation tokens | test tokens |
|---|---:|---:|---:|
| WikiText-2 | 3,986,646 | 417,909 | 472,325 |
| Python code | 3,167,087 | 133,784 | 421,354 |

## Limits

- Three seeds establish consistency in this setup but are too few for a broad
  architecture claim or a reliable small-sample significance test.
- The file-hash Python split has a real distribution shift: validation is much
  easier than test. Only paired comparisons within the same split are used.
- Sequence length is 64. Longer-context behavior remains untested.
- This experiment compares almost parameter-matched models, not equal-FLOP or
  equal-wall-clock training. Double Attention receives the same tokens but
  substantially more compute time in the current implementation.
- The Python result may reflect optimization or regularization, not a hard
  representational limit. Longer schedules and learning-rate sweeps remain
  open tests; the subsequently implemented Triton backward changes runtime,
  not the recorded model outputs.

## Reproduction

From WSL in the repository root:

```bash
source scripts/wsl_env.sh
python -m pip install -e ".[data,gpu]"
python scripts/prepare_protocol_v2.py
bash scripts/run_confirmation_v2.sh
python scripts/summarize_confirmation_v2.py runs/confirmation_v2
```

`data/protocol_v2/metadata.json` records source revisions, split rules, token
counts, and hashes. JSON results and logs are below `runs/confirmation_v2/`;
large `.pt` checkpoints and logs are excluded from Git by `.gitignore`.
