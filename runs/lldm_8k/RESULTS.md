# Independent-readout LLDM: 8k training experiment

## Result

The independent-readout LLDM fixes the original routing collapse, but does not
beat A1 or MHA4 under the full 8,000-step protocol. Its short 1k diagnostic win
does not survive the shared-tokenizer corpus and longer training horizon.

| corpus | LLDM independent readouts | A1 tied | MHA4 | LLDM - A1 | LLDM - MHA4 |
|---|---:|---:|---:|---:|---:|
| Python docs | 2.867906 | 2.824375 | **2.796927** | +0.043531 | +0.070979 |
| Python code | 3.218380 | 3.127668 | **3.062775** | +0.090712 | +0.155605 |

Lower robust validation loss is better. Every value is the mean of two
deterministic 64-batch evaluations.

## Logged validation trajectory

### Python docs

| step | LLDM | A1 | MHA4 |
|---:|---:|---:|---:|
| 1k | 3.955748 | 3.925563 | **3.912198** |
| 2k | 3.572359 | 3.531293 | **3.522132** |
| 4k | 3.220656 | 3.198252 | **3.183835** |
| 6k | 3.037986 | 3.006472 | **2.980446** |
| 8k | 2.918924 | 2.872258 | **2.849996** |

### Python code

| step | LLDM | A1 | MHA4 |
|---:|---:|---:|---:|
| 1k | 4.286752 | **4.279184** | 4.284285 |
| 2k | 3.969523 | 3.849037 | **3.830632** |
| 4k | 3.533289 | 3.445762 | **3.382418** |
| 6k | 3.289295 | 3.202999 | **3.111613** |
| 8k | 3.092550 | 3.031483 | **2.961101** |

LLDM trails from the first comparable checkpoint. The Python-docs gap to A1
briefly narrows at 4k and then widens. On Python code, MHA4 separates strongly
after 1k.

## Train-split audit

The final minibatch on Python code made LLDM appear to have a much lower train
loss. A separate two-pass, 64-batch deterministic evaluation rejects that
interpretation:

| corpus | LLDM train | A1 train | MHA4 train |
|---|---:|---:|---:|
| Python docs | 2.814342 | 2.730943 | **2.694655** |
| Python code | 2.431933 | 2.342812 | **2.276310** |

LLDM has higher train loss as well as higher validation loss on both corpora.
The long-run deficit is therefore optimization/representation inefficiency,
not additional overfitting from layer-local dictionaries.

## Cost

| model | parameters | peak allocation | 8k recorded chunks |
|---|---:|---:|---:|
| LLDM independent readouts | 15,461,918 | 405.2 MiB | 681.0 s |
| A1 tied | 15,396,870 | 354.4 MiB | 314.4 s |
| MHA4 | 16,838,656 | 366.0 MiB | 234.4 s |

The fresh Python-docs runs are used for the resource comparison. LLDM has
0.42% more parameters than A1 but uses 14.3% more peak memory. Its unfused
PyTorch implementation is 2.17x slower than the optimized A1 path and 2.91x
slower than MHA4; these timings are implementation results, not an intrinsic
lower bound for the architecture.

## Interpretation

The earlier diagnosis remains valid: independent Q/K/V/context readouts are
necessary to prevent nearly uniform attention. They are not sufficient for a
competitive long-run model. The remaining likely bottleneck is the decision
to replace two sequential Transformer sublayers with one parallel feature
decomposition:

```text
X' = X + local(Z(X)) + context(Z(X))
```

Unlike a Transformer, the local computation cannot condition on the result of
the contextual update (or vice versa) within the same layer. Both paths also
depend on one normalized dictionary expansion, so a weak feature direction
limits both simultaneously.

The next justified experiment is a sequential shared-feature block, not a
wider parallel LLDM:

```text
Zctx = features(RMSNorm(X), D)
Xctx = X + context(Zctx)
Zlocal = features(RMSNorm(Xctx), D)  # same layer-local D, new coefficients
X' = Xctx + local(Zlocal)
```

This preserves the layer-local dictionary hypothesis while restoring
attention-then-FFN conditional computation. Further width or temperature
sweeps of the current parallel block are not supported by these results.

## Protocol

- seed 0; d=512; six layers; sequence length 64; effective batch 8;
- shared 2,048-token multicorpus tokenizer;
- BF16 on NVIDIA GeForce RTX 5060 Laptop GPU;
- 600-step warmup, maximum learning rate 6e-4, 12k cosine schedule;
- stopped at 8,000 steps / 4,096,000 training tokens;
- Python code resumed from the exact 1k optimizer checkpoint;
- Python docs started at step 0 because its earlier diagnostic used a different
  tokenizer artifact.
