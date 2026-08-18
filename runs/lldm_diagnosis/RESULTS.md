# LLDM failure diagnosis

> Long-run follow-up: independent readouts fix the routing collapse but do not
> beat A1 or MHA4 at 8k. See [`../lldm_8k/RESULTS.md`](../lldm_8k/RESULTS.md).

## Conclusion

The layer-local shared feature state is not the main failure. The loss comes
from over-sharing the feature readouts after `Z` is constructed:

1. local and contextual features were forced through one synthesis matrix;
2. one low-rank matrix per map was reused for Q, K, V, and context lifting;
3. Q and K could differ only through diagonal feature gates.

Removing these constraints recovers the full loss deficit on Python docs
while preserving one layer-local `Z` shared by local and relational paths.

## Controlled 1k-step results

| Python docs model | parameters | robust loss | change |
|---|---:|---:|---:|
| LLDM parameter-matched, shared output/readout | 15,296,030 | 4.694217 | baseline |
| separate local/context output | 15,282,206 | 4.644286 | -0.049931 |
| separate output + independent Q/K | 15,253,022 | 4.606668 | -0.037618 |
| independent Q/K/V/context readouts | 15,461,918 | **4.573425** | -0.033243 |
| A1 tied | 15,396,870 | 4.596955 | reference |
| MHA4 | 16,838,656 | 4.707445 | reference |

The final independent-readout LLDM improves on the original parameter-matched
LLDM by 0.120792 and on A1 by 0.023530 in this seed.

On Python code, independent readouts improve LLDM from 4.485643 to 4.365082.
This is a large recovery but remains 0.035328 behind A1 (4.329754) and
0.029046 behind MHA4 (4.336036). The fix generalizes directionally, but the
ranking is corpus-dependent at 1,000 steps.

## Why the original readout collapses

The original relational path was

```text
q = normalize((P * gq) R)
k = normalize((P * gk) R)
v = (Z * gv) R
context = attention(q, k, v) R^T
output = (Z + lambda context) W
```

`P` has 98.5-99.4% of maximum assignment entropy. Its shared uniform component
is mapped through the same `R` for every token, while diagonal gates cannot
learn an independent rotation. In layers 2-6 this produces:

- token-Q cosine of 0.96-0.99;
- attention entropy of 95.9-99.8% of the causal maximum;
- Q effective-rank fractions of only 2.4-4.0% in most layers.

A1's corresponding attention entropy is 60.7-84.8% in layers 2-6. MHA4's is
12.7-38.1%. The original LLDM attention therefore behaves mostly like a
learned prefix average.

The gradient audit agrees. Original Q/K gate gradient-to-weight ratios are
3.5e-5 and 4.5e-5, while A1's independent Q/K projections are 4.8e-3 and
6.1e-3. A single context scalar receives a ratio above 2.0, so the model can
more easily tune global context strength than token-specific routing.

## What fixes it

The successful variant keeps the layer-local feature decomposition:

```text
Z = 2 SiLU(RMSNorm(X) D)
P = softmax(0.25 standardize(Z))
Pc = P - 1/M
```

but gives each relational map independent readouts:

```text
q_s = normalize(Pc Rq_s)
k_s = normalize(Pc Rk_s)
v_s = Z Rv_s
A_s = softmax(a_s q_s k_s^T + causal_mask)
C   = sum_s alpha_s (A_s v_s) U_s
X'  = X + Z Wlocal + lambda C
```

After 1,000 steps, token-Q cosine falls to 0.17-0.43, attention entropy falls
to 39.9-46.2%, and Q effective-rank fractions rise to 8.1-19.3%. Q/K
gradient-to-weight ratios become 4.8e-3 and 5.6e-3, almost exactly matching
A1. The learned context-scale gradient ratio also falls from 2.18 to 0.42.

Thus the shared dictionary hypothesis survives, but the shared-readout
hypothesis does not. `Z` can be common; Q, K, V, and synthesis need distinct
learned views of it.

## Rejected explanations

- **Parameter count alone:** matching A1's size recovers only part of the loss.
- **Uniform assignment alone:** subtracting `1/M` without independent readouts
  worsens Python-docs loss to 4.748549.
- **Insufficient assignment temperature:** increasing the assignment scale
  from 0.25 to 1.0 gives 4.748609, effectively the same failed trajectory as
  centering because Q/K L2 normalization removes the main scale difference.
- **Unused context:** disabling context at 1,000 steps worsens the shared-output
  model by 0.079308 and the separate-output model by 0.163319.

## Protocol

- seed 0; d=512; six layers; sequence length 64; effective batch 8;
- BF16 on NVIDIA GeForce RTX 5060 Laptop GPU;
- 600-step warmup, peak learning rate 6e-4, 12k cosine schedule;
- stopped at 1,000 steps / 512,000 training tokens;
- robust loss is the mean of two deterministic 64-batch evaluations;
- primary diagnosis on Python docs, directional confirmation on Python code.

These are controlled one-seed diagnostics. A longer multi-seed comparison is
required before treating the final independent-readout model as a new winner.
