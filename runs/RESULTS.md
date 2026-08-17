# Experiment summary

## Main result

The untied QK4-S4 model is the best run in the current screen. Its robust
validation loss is `3.527659`, which is `0.029234` below MHA4, with only
131,120 more parameters (`+0.779%`). Every tested Double Attention variant
beats MHA4 in this one-seed experiment.

## Unified ranking

| rank | model | dictionary | parameters | vs. MHA4 params | robust loss | vs. MHA4 loss | perplexity |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | QK4-S4 | untied | 16,969,776 | +0.779% | **3.527659** | **-0.029234** | **34.044** |
| 2 | QK2-S2 | tied | 16,969,752 | +0.779% | 3.534702 | -0.022191 | 34.285 |
| 3 | QK4-S4 | tied | 16,904,240 | +0.389% | 3.535695 | -0.021198 | 34.319 |
| 4 | QK1-S2 | tied | 15,396,888 | -8.562% | 3.536660 | -0.020233 | 34.352 |
| 5 | QK2-S2 | untied | 17,100,824 | +1.557% | 3.538038 | -0.018855 | 34.399 |
| 6 | QK1-S2 | untied | 15,527,960 | -7.784% | 3.538389 | -0.018504 | 34.411 |
| 7 | QK2-S1 | untied | 17,100,806 | +1.557% | 3.541223 | -0.015670 | 34.509 |
| 8 | QK2-S1 | tied | 16,969,734 | +0.778% | 3.545101 | -0.011793 | 34.643 |
| 9 | A1 | untied | 15,527,942 | -7.784% | 3.545413 | -0.011480 | 34.654 |
| 10 | A1 | tied | 15,396,870 | -8.562% | 3.545568 | -0.011326 | 34.659 |
| 11 | MHA4 | n/a | 16,838,656 | 0.000% | 3.556893 | 0.000000 | 35.054 |

Lower loss and perplexity are better. Loss differences are absolute, not
percentages.

## What the ablations say

- **Best quality:** untied QK4-S4. Four independent Q/K routing branches and
  four outer maps produced the lowest loss.
- **Best tied model:** tied QK2-S2. It is `0.022191` below MHA4 and nearly
  parameter-matched (`+0.779%`).
- **Best parameter efficiency:** tied QK1-S2. It uses 1,441,768 fewer
  parameters than MHA4 (`-8.562%`) while improving loss by `0.020233`.
- **A1 already beats MHA4:** untied A1 improves loss by `0.011480` with
  1,310,714 fewer parameters (`-7.784%`). Tying A1 changes loss by only
  `+0.000155`, so it is effectively neutral at this resolution.
- **Tying is architecture-dependent:** it improves QK1-S2 and QK2-S2, is
  nearly neutral for A1, and degrades QK2-S1 and QK4-S4 in this seed.
- **Outer-map multiplicity matters:** QK1-S2 is substantially better than A1
  without a meaningful parameter increase. QK2-S2 also improves over QK2-S1.

## Tied versus untied

| variant | untied loss | tied loss | tied - untied | parameters removed by tying |
|---|---:|---:|---:|---:|
| A1 | 3.545413 | 3.545568 | +0.000155 | 131,072 |
| QK2-S1 | 3.541223 | 3.545101 | +0.003878 | 131,072 |
| QK1-S2 | 3.538389 | 3.536660 | -0.001729 | 131,072 |
| QK2-S2 | 3.538038 | 3.534702 | -0.003336 | 131,072 |
| QK4-S4 | 3.527659 | 3.535695 | +0.008036 | 65,536 |

## Triton performance

On the NVIDIA GeForce MX570 A, the standalone A1 attention forward pass at
batch 4 and sequence length 64 took:

| implementation | latency | relative speed |
|---|---:|---:|
| Triton | 0.511 ms | 3.13x |
| PyTorch reference | 1.599 ms | 1.00x |

This measures the attention forward operator, not end-to-end training. These
historical experiments used the original PyTorch-recomputation backward. A
native Triton backward has since been added; current forward+backward figures
are reported in the repository README.

## Experiment conditions

- seed 0;
- model width 512, six layers, FFN width 1536;
- sequence length 64, effective batch 8;
- BF16 on an NVIDIA GeForce MX570 A;
- 600-step warmup, maximum learning rate `6e-4`;
- 12,000-step cosine schedule stopped at 8,000 steps;
- 4,096,000 training tokens seen per run;
- robust loss is the mean of two deterministic validation traversals.

## Interpretation limits

These results are a screening experiment, not a final architecture claim. Only
one seed was run, and several differences are small enough to change order
under seed variation. The next useful test is a 3-seed replication of MHA4,
untied QK4-S4, tied QK2-S2, and tied QK1-S2.

The original project token-ID artifact was unavailable. This experiment uses a
deterministically rebuilt SentencePiece unigram-2048 corpus from the official
Python 3.14 text documentation. The current runs are internally comparable,
but their loss values must not be compared numerically with the older project
runs that used a different corpus artifact and training configuration.

Detailed reports: [untied screen](screen/RESULTS.md) and
[tied screen](tied/RESULTS.md).

## Cross-corpus follow-up

A separate 2,000-step shared-tokenizer screen tests the leading models on
Python documentation, WikiText-2, Shakespeare, and Python source code. QK4-S4
untied and QK2-S2 tied beat MHA4 on all four corpora. See the
[cross-corpus report](multicorpus/RESULTS.md) for the full table and limits.

## Three-seed confirmation

The recommended follow-up is now complete: 18 leakage-resistant runs compare
MHA4, QK4-S4 untied, and QK2-S2 tied across WikiText-2 and Python code with
seeds 0, 1, and 2. On held-out test data, QK4-S4 improves WikiText loss by
`0.039661 ± 0.006608`, while MHA4 beats it on Python code by
`0.084825 ± 0.035738`. The earlier 2k code result is reproduced at 2k but
reverses between 4k and 6k. See the
[three-seed confirmation report](confirmation_v2/RESULTS.md) for full results,
parameter/runtime comparisons, split controls, and reproduction commands.
