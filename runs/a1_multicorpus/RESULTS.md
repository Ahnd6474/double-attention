# A1 versus QK1-S2 multicorpus screen

## Result

QK1-S2 tied has the lowest robust validation loss on all four corpora, followed
by A1 tied and MHA4.  Because A1 tied and QK1-S2 tied differ by only 18 learned
parameters, their paired difference isolates the effect of replacing one outer
attention softmax with a learned mixture of two temperatures over the same
routing geometry.

| corpus | MHA4 | A1 tied | A1 vs. MHA4 | QK1-S2 tied | QK1-S2 vs. MHA4 | QK1-S2 vs. A1 |
|---|---:|---:|---:|---:|---:|---:|
| Python docs | 3.493892 | 3.459328 | -0.034564 | **3.441027** | **-0.052865** | **-0.018301** |
| WikiText-2 | 4.287544 | 4.280093 | -0.007451 | **4.264392** | **-0.023152** | **-0.015701** |
| Shakespeare | 4.269197 | 4.217091 | -0.052107 | **4.201400** | **-0.067798** | **-0.015691** |
| Python code | 3.923876 | 3.888752 | -0.035124 | **3.886437** | **-0.037439** | **-0.002314** |
| Mean paired delta | 0 | -0.032311 | -0.032311 | **-0.045313** | **-0.045313** | **-0.013001** |

Lower loss is better.  Each robust loss is the mean of two deterministic
64-batch validation evaluations.  The perplexities are:

| corpus | MHA4 | A1 tied | QK1-S2 tied |
|---|---:|---:|---:|
| Python docs | 32.914 | 31.796 | **31.219** |
| WikiText-2 | 72.787 | 72.247 | **71.122** |
| Shakespeare | 71.464 | 67.836 | **66.780** |
| Python code | 50.596 | 48.850 | **48.737** |

## Interpretation

- A1 beats MHA4 on all four corpora at 2,000 steps while using 15,396,870
  parameters versus MHA4's 16,838,656, a reduction of 1,441,786 (8.56%).
- QK1-S2 adds only 18 parameters to A1 and improves every corpus.  Its gain over
  A1 is consistent on Python docs, WikiText-2, and Shakespeare
  (`0.015691`-`0.018301` loss), but much smaller on Python code (`0.002314`).
- The two learned maps did not collapse.  Across layers and corpora their final
  score scales remain separated by roughly `1.0`-`1.3`, and their learned mix
  weights range from about `0.41/0.59` to `0.57/0.43` rather than staying at the
  initial `0.50/0.50`.
- This supports an early-convergence benefit from the two-temperature mixture;
  it does not yet establish a converged or corpus-independent advantage.

## Protocol

- seed 0; width 512; six layers; FFN width 1536;
- sequence length 64; effective batch 8; BF16;
- 150-step warmup; maximum learning rate `6e-4`;
- 3,000-step cosine schedule stopped at 2,000 steps;
- 1,024,000 training tokens per run;
- tied 256-by-512 dictionary shared across all six Double Attention layers;
- NVIDIA GeForce RTX 5060 Laptop GPU; 12 completed runs.

The experiment uses one shared 2,048-piece SentencePiece unigram tokenizer
trained from one million characters of each corpus.  WikiText and Shakespeare
use the source revisions recorded in `data/multicorpus/metadata.json`.

## Comparison scope

This is a new internally controlled cohort.  Its tokenizer was rebuilt on
Windows and its Python-code corpus comes from the local Python 3.12.14 standard
library.  Its absolute losses must not be merged with `runs/multicorpus`, whose
stored runs were generated in an earlier environment with different token
counts.  Comparisons within this report are paired and valid.

Only one seed and an early 2,000-step checkpoint were measured.  The earlier
8,000-step confirmation showed that an early Python-code advantage can reverse
between 4,000 and 6,000 steps, so the Python-code ordering here is provisional.
The next decisive experiment is a three-seed 8,000-step comparison of MHA4,
A1 tied, and QK1-S2 tied using leakage-resistant train/validation/test splits.
