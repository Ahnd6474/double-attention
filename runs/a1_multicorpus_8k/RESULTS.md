# A1 multicorpus 8k experiments

## Result

Removing either routing normalization does not improve A1. Across the four
corpora, removing Q/K normalization increases robust validation loss by
`+0.566180` on average, removing reconstructed-Dp normalization by `+0.165906`,
and removing both by `+0.282004`.

The post-SiLU logit-normalized variant avoids those large failures, but it does
not improve final robust loss consistently. Its mean paired delta from A1 is
`+0.003904`; lower is better.

Removing softmax and using signed direct-SiLU coefficients is strongly
domain-dependent: it loses to A1 on three corpora but improves Shakespeare by
more than `0.32`. The favorable arithmetic mean is therefore not a robust win.

| corpus | MHA4 | A1 | A1-SiLU | A1-SiLU-logitnorm | logitnorm vs. A1 |
|---|---:|---:|---:|---:|---:|
| Python docs | **2.796927** | 2.824375 | 2.825448 | 2.825169 | +0.000795 |
| WikiText-2 | 3.491300 | 3.474998 | **3.471369** | 3.479587 | +0.004590 |
| Shakespeare | **4.309338** | 4.316602 | 4.310566 | 4.330707 | +0.014105 |
| Python code | **3.062775** | 3.127668 | 3.141167 | 3.123794 | -0.003874 |
| Mean paired delta vs. A1 | -0.020826 | 0 | +0.001227 | +0.003904 | +0.003904 |

Each loss is the mean of two deterministic 64-batch validation evaluations.

## Normalization ablations

| corpus | A1 | no Q norm | delta | no Dp norm | delta | neither norm | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python docs | **2.824375** | 3.320714 | +0.496339 | 2.908207 | +0.083832 | 3.066154 | +0.241779 |
| WikiText-2 | **3.474998** | 4.425899 | +0.950901 | 3.536951 | +0.061953 | 3.678218 | +0.203220 |
| Shakespeare | **4.316602** | 4.470589 | +0.153987 | 4.555244 | +0.238643 | 4.327090 | +0.010488 |
| Python code | **3.127668** | 3.791162 | +0.663493 | 3.406864 | +0.279195 | 3.800198 | +0.672529 |
| Mean delta | 0 |  | **+0.566180** |  | **+0.165906** |  | **+0.282004** |

Q/K row normalization is therefore primarily a dictionary-softmax calibration
mechanism in this setup. The pre-attention LayerNorm does not make it
redundant: the learned Q/K projections reintroduce variable row norms, and raw
dot products make the fixed `beta=4` assignment much too sharp.

## Post-SiLU logit normalization

The new route uses raw Q/K for the dictionary dot product, then standardizes
after the nonlinearity:

```text
u_i = d_i^T q
h_i = 2 SiLU(u_i)
z_i = (h_i - mean(h)) / sqrt(mean((h - mean(h))^2) + eps) / sqrt(d_r)
p   = softmax(beta z)
q_D = normalize(sum_i p_i d_i)
```

The `1/sqrt(d_r)` factor is essential. Atom-axis standardization gives unit
logit standard deviation; at routing width 256, division by 16 restores the
approximately 0.25 post-beta standard deviation of the original normalized A1
route instead of feeding unit-scale logits into `beta=4`.

Logged single-traversal loss deltas for A1-SiLU-logitnorm versus A1 are:

| corpus | 2k | 4k | 6k | 8k |
|---|---:|---:|---:|---:|
| Python docs | -0.004154 | -0.021890 | -0.016077 | -0.033612 |
| WikiText-2 | +0.024331 | -0.020624 | -0.032128 | -0.044055 |
| Shakespeare | -0.058472 | -0.082251 | -0.041300 | -0.065696 |
| Python code | +0.048545 | +0.039848 | +0.045526 | +0.064376 |

The first three logged curves look better, but the independent two-pass robust
evaluation does not preserve those gains. With one seed and roughly 32k
validation tokens per pass, the small final differences are not statistically
resolved. Python code is the only robust-loss win over A1, at `-0.003874`.

Shakespeare is overtrained under the common 8k budget. The post-SiLU variant
reaches `4.102554` at 4k versus A1's best logged `4.184806`, then degrades to a
final robust `4.330707`. It is a useful early-stopping result, not an 8k win.

## Routing-width ablation

The routing bottleneck experiment increases A1's Q/K routing width from 256 to
512. To isolate capacity from temperature, `beta` increases from 4 to
`4 sqrt(2) = 5.656854`, and the outer score scale increases from 16 to
`sqrt(512) = 22.627417`. R512-D512 holds atom count fixed; R512-D1024 retains
the baseline's two-atoms-per-routing-dimension expansion ratio.

| corpus | MHA4 | R256-D512 | R512-D512 | delta | R512-D1024 | delta |
|---|---:|---:|---:|---:|---:|---:|
| Python docs | **2.796927** | 2.824375 | 2.820142 | -0.004233 | 2.818171 | -0.006204 |
| WikiText-2 | 3.491300 | 3.474998 | **3.473432** | -0.001566 | 3.477734 | +0.002736 |
| Shakespeare | 4.309338 | 4.316602 | 4.309947 | -0.006654 | **4.309005** | -0.007596 |
| Python code | **3.062775** | 3.127668 | 3.137728 | +0.010059 | 3.126655 | -0.001014 |
| Mean paired delta vs. R256 | -0.020826 | 0 |  | -0.000598 |  | -0.003019 |

The robust results support only a mild, corpus-dependent bottleneck. R512-D512
is effectively tied with R256 on average, while R512-D1024 improves mean loss by
`0.003019`. Both remain materially behind MHA4 on Python docs and code despite
having more parameters than MHA4.

The logged curves show a stronger optimization effect:

| corpus | R512-D512 delta at 2k | 4k | 6k | 8k |
|---|---:|---:|---:|---:|
| Python docs | -0.018202 | -0.030641 | -0.009136 | -0.036694 |
| WikiText-2 | +0.026132 | -0.025299 | -0.038927 | -0.042479 |
| Shakespeare | -0.046913 | -0.089044 | -0.096187 | -0.081627 |
| Python code | +0.053904 | +0.055692 | +0.062406 | +0.077944 |

Routing width helps Python docs and Shakespeare throughout training, and helps
WikiText-2 after 2k, but consistently hurts the Python-code curve. The
Shakespeare 4k loss improves from `4.184806` to `4.095762`, the clearest
bottleneck signal in the cohort. Increasing dictionary size from 512 to 1024
changes mean robust loss by only `-0.002421` relative to R512-D512, so atom
count is not the primary limitation.

| model | parameters | vs. A1 | peak allocated |
|---|---:|---:|---:|
| R256-D512 | 15,396,870 | 0 | 354.4 MiB |
| R512-D512 | 17,100,806 | +1,703,936 (+11.1%) | 384.4 MiB |
| R512-D1024 | 17,362,950 | +1,966,080 (+12.8%) | 405.4 MiB |

The width result therefore does not explain the remaining MHA gap by itself.
Independent attention-map diversity and the tied simplex dictionary map remain
stronger candidates than raw routing width alone.

## Direct SiLU assignment without softmax

These variants remove the dictionary softmax entirely while retaining both
the Q/K input L2 normalization and reconstructed-output L2 normalization:

```text
u_i = d_i^T normalize(q)
c_i = (2 / g) SiLU(g u_i)
q_D = normalize(sum_i c_i d_i)
```

`g=1` is the direct-SiLU variant. `g=4` is a more selective control; the
`2/g` factor gives both variants unit slope at the origin. Unlike softmax, the
coefficients are signed, do not sum to one, and do not use `beta`.

| corpus | MHA4 | A1 softmax | direct SiLU g=1 | delta vs. A1 | direct SiLU g=4 | delta vs. A1 |
|---|---:|---:|---:|---:|---:|---:|
| Python docs | **2.796927** | 2.824375 | 2.886226 | +0.061851 | 2.901494 | +0.077119 |
| WikiText-2 | 3.491300 | **3.474998** | 3.692929 | +0.217931 | 3.614578 | +0.139580 |
| Shakespeare | 4.309338 | 4.316602 | **3.973837** | -0.342765 | 3.988546 | -0.328055 |
| Python code | **3.062775** | 3.127668 | 3.157479 | +0.029810 | 3.138199 | +0.010531 |
| Mean loss | 3.415085 | 3.435911 | 3.427618 | -0.008293 | **3.410704** | -0.025206 |

The apparently favorable mean is entirely caused by the large Shakespeare
gain. Both direct variants lose to A1 on three of four corpora; their median
deltas are `+0.045831` (`g=1`) and `+0.043825` (`g=4`). The stronger gain
recovers `0.078351` on WikiText-2 and `0.019280` on Python code relative to
`g=1`, but hurts Python docs and Shakespeare slightly.

Softmax is therefore not merely suppressing coefficient magnitude. Its
atom-wise competition and adaptive normalization are useful on the larger,
more diverse corpora. Direct signed coefficients provide substantially faster
fitting on the repeatedly traversed Shakespeare corpus, but are not a robust
replacement for softmax under this protocol.

## Broader comparison

QK1-S2 remains best on WikiText-2 at `3.470164`, but its mean delta from A1 is
`+0.010592` because it loses on Shakespeare and Python code. MHA4 remains best
on Python docs and Python code; R512-D1024 and MHA4 are tied within noise on
Shakespeare. None of the A1 activation, normalization, or width variants
produces a corpus-independent win.

## Protocol

- seed 0; width 512; six layers; FFN width 1536;
- sequence length 64; effective batch 8; BF16;
- 600-step warmup; maximum learning rate `6e-4`;
- 12,000-step cosine schedule stopped at 8,000 steps;
- 4,096,000 training tokens per run;
- one tied dictionary shared across all six Double Attention layers;
- NVIDIA GeForce RTX 5060 Laptop GPU;
- 48 completed runs in the main comparison: MHA4, A1, QK1-S2, A1-SiLU,
  A1-SiLU-logitnorm, three normalization ablations, and two R512 variants on
  four corpora, plus two direct-SiLU variants without softmax. The later
  `softmax(z)` run was stopped by design and excluded.

This is a one-seed controlled cohort with no held-out test split. Absolute
losses must not be merged with the earlier `runs/multicorpus` or
`runs/confirmation_v2` cohorts.
