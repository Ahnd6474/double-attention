# Layer-Local Dictionary Mixer: 1k-step screen

> Follow-up: the failure was traced to shared Q/K/V/output readouts rather
> than the layer-local feature state itself. See
> [`../lldm_diagnosis/RESULTS.md`](../lldm_diagnosis/RESULTS.md).

## Result

The proposed unified Layer-Local Dictionary Mixer (LLDM) fits faster during
the first 500 warmup steps, but its advantage reverses after warmup. At 1,000
steps the parameter-efficient LLDM-2 loses to both A1 and MHA4 on Python docs,
and loses to the nearly tied A1/MHA4 pair on Python code.

Increasing feature and relational width to match A1's parameter count recovers
some quality, but does not close the gap to A1. This rejects raw parameter
count as the sole explanation for the reversal.

| corpus | A1 tied | MHA4 | LLDM-2 | LLDM-2 parameter-matched |
|---|---:|---:|---:|---:|
| Python docs | **4.596955** | 4.707445 | 4.741970 | 4.694217 |
| Python code | **4.329754** | 4.336036 | 4.506595 | 4.485643 |

Lower robust validation loss is better. These are one-seed architecture
screens, not final quality estimates.

## Architecture sizes

| model | feature width | relational maps x rank | parameters |
|---|---:|---:|---:|
| LLDM-2 | 1,024 | 2 x 128 | 8,986,142 |
| LLDM-2 parameter-matched | 1,536 | 2 x 256 | 15,296,030 |
| A1 tied | n/a | 1 x 256 | 15,396,870 |
| MHA4 | FFN 1,536 | 4 x 128 | 16,838,656 |

LLDM-2 uses 41.6% fewer parameters than A1 and 46.6% fewer than MHA4. Its
0-to-500-step peak allocation was 325.6 MiB, versus 354.4 MiB for A1 and
366.0 MiB for MHA4. The parameter-matched LLDM uses 490.4 MiB because its two
1,536-by-256 relational projectors and their activations are retained for
backward. Runtime comparisons are excluded: LLDM has no custom kernel, and
the Windows runs mixed cold starts with checkpoint-resumed warm starts.

## Learning trajectory

At step 500, before the 600-step warmup completed, LLDM-2 robust loss was
4.998533 on Python docs versus 5.038580 for A1 and 5.119120 for MHA4. On Python
code it was 4.707773 versus 4.712811 and 4.794693. The early win therefore
appeared on both corpora, but did not survive the high-learning-rate phase
after warmup.

The parameter-matched model improved over LLDM-2 at 1,000 steps by 0.047753 on
Python docs and 0.020953 on Python code. It still lost to A1 by 0.097262 and
0.155889 respectively. Capacity explains only part of the deficit.

## Relational-path audit

Turning every learned context scale off in the 500-step LLDM-2 checkpoint
changed robust loss as follows:

| corpus | full LLDM-2 | context disabled | degradation |
|---|---:|---:|---:|
| Python docs | 4.998533 | 5.005122 | +0.006589 |
| Python code | 4.707773 | 4.772730 | +0.064956 |

The relational path is therefore active, particularly on Python code; the
model is not behaving as a token-independent FFN stack.

Assignment entropy was 98.8-99.4% of its maximum and mean pairwise token-Q
cosine similarity reached 0.97-0.99 in deeper layers. A parameter-matched
control subtracting the uniform assignment component (`P - 1/M`) reduced that
common component but worsened Python-docs robust loss from 4.694217 to
4.748549. Assignment centering is not a viable fix in this form.

## Interpretation

The experiment supports layer-local feature decomposition as an efficient
early-learning bias, but not the more aggressive decision to replace both
Transformer sublayers with one shared synthesis path. The remaining likely
bottlenecks are:

1. local and contextual updates share one output projection and one residual;
2. the relational value path is restricted to the same rank-r subspace used
   for routing;
3. local computation is a single `2 SiLU(xD)` expansion without an independent
   gate;
4. six fully independent dictionaries receive fewer cross-layer training
   signals than the globally shared A1 bank.

The next justified control is a conservative two-residual block that retains
layer-local `Z`, but gives local and contextual paths independent output
projections. Longer or multi-seed runs of the current unified block are not
recommended before that structural control.

## Protocol

- seed 0; width 512; six layers; sequence length 64; effective batch 8;
- BF16 on an NVIDIA GeForce RTX 5060 Laptop GPU;
- 600-step warmup; maximum learning rate 6e-4; 12,000-step cosine schedule;
- stopped at 1,000 steps (512,000 training tokens);
- two deterministic 64-batch validation evaluations for each robust loss;
- Python docs and Python code corpora with the existing 2,048-token vocabulary.
