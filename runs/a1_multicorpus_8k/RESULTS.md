# A1 versus QK1-S2 multicorpus 8k run

## Result

The 8,000-step result is domain-dependent.  MHA4 has the lowest robust
validation loss on Python documentation, Shakespeare, and Python code;
QK1-S2 tied is best on WikiText-2.  The broad early-training advantage in the
separate 2,000-step screen does not persist under the longer schedule.

| corpus | MHA4 | A1 tied | A1 vs. MHA4 | QK1-S2 tied | QK1-S2 vs. MHA4 | QK1-S2 vs. A1 |
|---|---:|---:|---:|---:|---:|---:|
| Python docs | **2.796927** | 2.824375 | +0.027448 | 2.823904 | +0.026978 | -0.000470 |
| WikiText-2 | 3.491300 | 3.474998 | -0.016303 | **3.470164** | **-0.021136** | -0.004834 |
| Shakespeare | **4.309338** | 4.316602 | +0.007263 | 4.361591 | +0.052253 | +0.044990 |
| Python code | **3.062775** | 3.127668 | +0.064894 | 3.130352 | +0.067578 | +0.002684 |
| Mean paired delta | 0 | +0.020826 | +0.020826 | +0.031418 | +0.031418 | +0.010592 |

Lower loss is better.  Each robust loss is the mean of two deterministic
64-batch validation evaluations.  The corresponding perplexities are:

| corpus | MHA4 | A1 tied | QK1-S2 tied |
|---|---:|---:|---:|
| Python docs | **16.394** | 16.850 | 16.842 |
| WikiText-2 | 32.829 | 32.298 | **32.142** |
| Shakespeare | **74.391** | 74.934 | 78.382 |
| Python code | **21.387** | 22.821 | 22.882 |

## Learning-curve deltas

Each cell below is `A1 vs. MHA4 / QK1-S2 vs. MHA4` using the logged
single-validation traversal at that checkpoint.  Negative is better than
MHA4.

| corpus | 2k | 4k | 6k | 8k |
|---|---:|---:|---:|---:|
| Python docs | +0.0092 / +0.0017 | +0.0144 / -0.0044 | +0.0260 / +0.0323 | +0.0223 / +0.0273 |
| WikiText-2 | -0.0010 / +0.0175 | -0.0272 / -0.0276 | -0.0178 / -0.0122 | -0.0090 / -0.0188 |
| Shakespeare | -0.0249 / +0.0141 | +0.0000 / +0.0138 | -0.0341 / -0.0308 | +0.0010 / +0.0708 |
| Python code | +0.0184 / +0.0220 | +0.0633 / +0.0592 | +0.0914 / +0.1047 | +0.0704 / +0.0655 |

## Interpretation

- A1 and QK1-S2 retain an advantage on WikiText-2, agreeing with the earlier
  three-seed confirmation that dictionary-routed attention works well on this
  domain.
- MHA4 overtakes both models on Python documentation and Python code.  The
  code gap is substantial: `0.064894` for A1 and `0.067578` for QK1-S2.
- The second softmax is not a reliable long-run improvement over A1.  QK1-S2
  wins on Python docs and WikiText-2 by `0.000470` and `0.004834`, but loses on
  Shakespeare and Python code.  The mean QK1-S2 minus A1 delta is `+0.010592`.
- The learned maps do not numerically collapse: final per-layer temperature
  gaps range from about `0.11` to `1.35`, and map weights range from `0.38` to
  `0.62`.  Their continued differentiation does not guarantee better final
  generalization.
- Shakespeare is overtrained in this setup.  Its 0.38M-token training split is
  traversed approximately 10.7 times; all models reach their best logged
  validation loss at 4k and degrade afterward.  At 4k MHA4 and A1 are tied to
  four decimal places (`4.184774` versus `4.184806`).

## Protocol

- seed 0; width 512; six layers; FFN width 1536;
- sequence length 64; effective batch 8; BF16;
- 600-step warmup; maximum learning rate `6e-4`;
- 12,000-step cosine schedule stopped at 8,000 steps;
- 4,096,000 training tokens per run;
- tied 256-by-512 dictionary shared across all six Double Attention layers;
- NVIDIA GeForce RTX 5060 Laptop GPU; 12 completed runs.

This is a new internally controlled cohort using the shared tokenizer recorded
in `data/multicorpus/metadata.json`.  It has no held-out test split and only one
seed, so the small WikiText and Python-document differences require
replication.  Absolute losses must not be merged with the earlier
`runs/multicorpus` or `runs/confirmation_v2` cohorts.

The earlier `runs/a1_multicorpus` experiment used a 3,000-step cosine schedule
stopped at 2,000 steps.  This experiment uses a 12,000-step schedule from the
start, so its 2k checkpoints are the valid early points for interpreting these
8k trajectories; the two cohorts' 2k endpoints are not directly comparable.
