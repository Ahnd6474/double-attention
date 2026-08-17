# Tied-dictionary 8k GPU screen

All Double Attention variants in this screen use one tied dictionary `D` for
both the routing-key and routing-value paths. The MHA4 row is the same baseline
from the untied screen because it has no dictionary to tie.

## Results

| variant | parameters | robust loss | vs. MHA4 | perplexity |
|---|---:|---:|---:|---:|
| QK2-S2 tied | 16,969,752 | **3.534702** | **-0.022191** | **34.285** |
| QK4-S4 tied | 16,904,240 | 3.535695 | -0.021198 | 34.319 |
| QK1-S2 tied | 15,396,888 | 3.536660 | -0.020233 | 34.352 |
| QK2-S1 tied | 16,969,734 | 3.545101 | -0.011793 | 34.643 |
| A1 tied | 15,396,870 | 3.545568 | -0.011326 | 34.659 |
| MHA4 | 16,838,656 | 3.556893 | 0.000000 | 35.054 |

Lower loss is better. On this one-seed screen, every tied Double Attention
variant beats MHA4. QK2-S2 tied is the best tied model, while the untied QK4-S4
run remains the overall best result at `3.527659`.

## Tied versus untied

| variant | tied loss | untied loss | tied - untied | parameter reduction |
|---|---:|---:|---:|---:|
| A1 | 3.545568 | 3.545413 | +0.000155 | 131,072 |
| QK2-S1 | 3.545101 | 3.541223 | +0.003878 | 131,072 |
| QK1-S2 | 3.536660 | 3.538389 | -0.001729 | 131,072 |
| QK2-S2 | 3.534702 | 3.538038 | -0.003336 | 131,072 |
| QK4-S4 | 3.535695 | 3.527659 | +0.008036 | 65,536 |

The tied result is not uniformly better or worse: tying helps QK1-S2 and
QK2-S2 in this seed, is nearly neutral for A1, and hurts QK4-S4. These are
screening results, not statistical evidence; multiple seeds are required for a
reliable architecture ranking.

## Reproduction

```bash
source scripts/wsl_env.sh
bash scripts/run_tied_screen.sh
python scripts/summarize_runs.py runs/tied
```

Shared settings: seed 0, six LM layers, sequence length 64, effective batch 8,
8,000 optimizer steps, 12,000-step cosine schedule, BF16 on an NVIDIA GeForce
MX570 A, and the locally prepared Python documentation corpus. Raw JSON,
checkpoints, and logs are stored beside this report.
