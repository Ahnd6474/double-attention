# Double Attention experiments

This package implements the shared-dictionary, low-dimensional routing design
from the `Double Attention` project and the latest multiplicity ablations.

The combined result summary is in [`runs/RESULTS.md`](runs/RESULTS.md). The
cross-corpus follow-up is in
[`runs/multicorpus/RESULTS.md`](runs/multicorpus/RESULTS.md). The leakage-resistant
three-seed confirmation is in
[`runs/confirmation_v2/RESULTS.md`](runs/confirmation_v2/RESULTS.md). Its main
finding is conditional: Double Attention wins all three WikiText-2 seeds, but
MHA4 has the best final Python-code loss after the Double Attention models'
early 2k advantage reverses between 4k and 6k. Detailed 8k GPU screen reports
are in:

- [`runs/screen/RESULTS.md`](runs/screen/RESULTS.md) for the untied `Dk`/`Dv` sweep;
- [`runs/tied/RESULTS.md`](runs/tied/RESULTS.md) for the shared single-`D` sweep.
Query/key routing is separated from dense value content:

```text
Q,K = project(x)                         # low-dimensional, per Q/K branch
z    = normalize(Dv softmax(beta Dk^T normalize(Q or K)))
A_s  = softmax(gamma_s zq_s zk_s^T)     # one or more outer maps
Y    = sum_s alpha_s A_s V               # V is dense and shared
```

The default dictionary is untied (`Dk`, `Dv`) to reproduce the attached d=512
experiment. Set `untied_dictionary=False` for the single-D formulation in the
project report. Dictionaries can be global or shared by layer groups.

## Experiment presets

| preset | Q/K branches | outer softmaxes | routing width/branch | purpose |
|---|---:|---:|---:|---|
| `a1` | 1 | 1 | 256 | current single-map model |
| `a1-silu` | 1 | 1 | 256 | apply `2 SiLU` to dictionary logits before assignment softmax |
| `a1-silu-logitnorm` | 1 | 1 | 256 | raw Q/K, then `2 SiLU`, atom-axis standardization, and `1/sqrt(routing_dim)` scaling |
| `a1-silu-logitnorm-t1` | 1 | 1 | 256 | same post-SiLU standardization with unit final softmax scale: `softmax(z)` |
| `a1-r512-d512` | 1 | 1 | 512 | remove the Q/K routing bottleneck at fixed dictionary size, with calibrated temperatures |
| `a1-r512-d1024` | 1 | 1 | 512 | remove the bottleneck while retaining the 2x dictionary expansion ratio |
| `a1-r512-d1536` | 1 | 1 | 512 | 512-wide attention and 1,536 atoms with the standard learned FFN |
| `a1-r512-d1536-qffn` | 1 | 1 | 512 | full-rank `W GELU(D^T Q x)` using 512-wide attention |
| `a1-r512-d2855-qffn` | 1 | 1 | 512 | parameter-matched full-rank Q-D FFN with 2,855 atoms/hidden units |
| `a1-d1536` | 1 | 1 | 256 | 1,536 dictionary atoms with the standard learned FFN |
| `a1-d1536-qffn` | 1 | 1 | 256 | reuse attention Q and D as the FFN expansion: `W GELU(D^T Q x)` |
| `a1-no-softmax` | 1 | 1 | 256 | replace simplex assignment with signed `2 SiLU(D^T q)` dictionary coefficients |
| `a1-no-softmax-g4` | 1 | 1 | 256 | direct SiLU coefficients with gain 4 and unit slope at the origin |
| `a1-no-qnorm` | 1 | 1 | 256 | remove Q/K row normalization before dictionary assignment |
| `a1-no-dpnorm` | 1 | 1 | 256 | keep the reconstructed `Dp` magnitude |
| `a1-no-norm` | 1 | 1 | 256 | remove both routing row normalizations |
| `qk2-s1` | 2 | 1 | 256 | isolate independent Q/K projections |
| `qk1-s2` | 1 | 2 | 256 | isolate outer-softmax multiplicity |
| `qk2-s2` | 2 | 2 | 256 | combine both effects |
| `qk4-s4` | 4 | 4 | 128 | match MHA4 Q/K head width |

For `qk2-s1`, branch scores are summed with RMS normalization before a single
softmax. For `qk1-s2`, the routing geometry is deliberately shared and the two
maps have slightly different learnable temperatures; this avoids an exactly
symmetric duplicate while not adding another Q/K projection. Independent maps
read the same dense `V` and are mixed by learned softmax weights.

## Usage

```python
import torch
from double_attention import SharedDictionaryAttention, experiment_config

config = experiment_config("qk2-s2", backend="auto")
attention = SharedDictionaryAttention(config).cuda().half()
x = torch.randn(8, 512, 512, device="cuda", dtype=torch.float16)
y = attention(x)
```

For a six-layer LM experiment:

```python
from double_attention import DoubleAttentionLM, experiment_config

model = DoubleAttentionLM(
    vocab_size=2048,
    max_sequence_length=64,
    config=experiment_config("a1"),
    num_layers=6,
    dictionary_group_size=6,  # one global bank
    feedforward_dim=1536,
)
```

## Triton path

`backend="auto"` uses Triton for CUDA fp16/bf16 tensors and falls back to the
fully differentiable PyTorch reference elsewhere. The GPU path contains:

- one dictionary normalization per shared bank and stack forward, rather than
  repeating the same normalization graph in every layer that uses the bank;
- one combined Q/K projection GEMM while retaining the original
  `query.weight` and `key.weight` parameters for checkpoint compatibility;
- fused row L2-normalization forward and backward, with saved inverse norms so
  backward does not recompute row norms or retain the pre-normalized routing
  activation;
- an RTX 50-series fast path that fuses input row normalization with the
  standard 256-by-512 dictionary assignment GEMM for up to 1,024 routing rows,
  with automatic cuBLAS fallback outside the measured winning range;
- a fully fused RTX 50-series dictionary-route forward kernel for the standard
  128/256 routing widths and 512 atoms. It performs normalization, assignment,
  temperature softmax, reconstruction, and output normalization in one launch
  for up to 4,096 rows while writing the tensors required by backward;
- a matching fused dictionary-route backward kernel that combines output
  normalization backward, value projection, softmax backward, key projection,
  and input normalization backward. Only the two dictionary-parameter gradient
  GEMMs remain delegated to PyTorch/cuBLAS. The measured cutoff is 4,096 rows
  at routing width 128 and 2,048 rows at width 256;
- fused temperature-scaled dictionary softmax forward and backward;
- a Flash-style causal routed-attention forward kernel that never materializes
  the `[T, T]` score/probability matrices and reuses one dense value tensor for
  all outer maps;
- native Flash-style routed-attention backward kernels for `dQ`, `dK`, `dV`,
  and score-scale gradients. They reuse saved row log-sum-exp values and
  recompute only score tiles, without building a PyTorch autograd graph. When
  its total workspace is at most 16 MiB, one FP32 `dP = dO V^T` buffer is shared
  by the `dQ` and `dK` kernels; larger cases automatically use the matrix-free
  recomputation path. Scores and probabilities are never materialized.

Unsupported dictionary shapes and pre-RTX-50 GPUs automatically use the
multi-kernel path, where dictionary GEMMs remain delegated to PyTorch/cuBLAS.
The custom autograd functions do not support higher-order gradients.

On the NVIDIA GeForce MX570 A with BF16, batch 2, and sequence length 64, a
full attention-module forward+backward step measured:

| variant | Triton | PyTorch reference | speedup | Triton peak memory |
|---|---:|---:|---:|---:|
| QK4-S4 | 5.038 ms | 5.507 ms | 1.09x | 25.634 MiB |
| QK2-S2 | 4.817 ms | 5.203 ms | 1.08x | 24.384 MiB |

These are paired CUDA-event medians, not full-language-model training times.
At batch 1 and sequence length 512, QK4-S4 measured 4.082 ms versus 5.370 ms
(`1.32x`) and used 44.009 MiB versus 49.041 MiB. QK2-S2 measured 4.519 ms
versus 5.432 ms (`1.20x`) and used 36.759 MiB versus 38.525 MiB. The kernel
selects smaller tiles through length 128 and larger tiles for longer contexts,
and causal programs skip fully masked future/past tiles.

In a paired 50-iteration benchmark of the complete six-layer LM at the actual
experiment shape (BF16, batch 8, length 64), QK4-S4 measured 35.847 ms versus
42.791 ms (`1.19x`). QK2-S2 measured 37.088 ms versus 40.287 ms (`1.09x`).
This includes embeddings, FFNs, normalization, and the language-model head.

A same-seed, six-layer QK4-S4 smoke run on the WikiText-2 training split cut
the logged 0-to-500-step time from 38.749 seconds with the old recomputation
backward to 29.344 seconds with native Triton backward (`1.32x`). Step-500
validation loss remained close (4.988099 old, 4.980435 native). This short run
checks integration and trajectory consistency; it is not a replacement for the
three-seed quality experiment.

Install and test:

```bash
uv sync --extra test --extra gpu --extra data
uv run python scripts/check_environment.py
uv run pytest
uv run python benchmarks/benchmark_attention.py --variant qk2-s2 --backend triton
uv run python benchmarks/benchmark_attention.py --variant qk2-s2 --backend triton \
  --batch 2 --sequence 64 --dtype bf16 --backward
uv run python benchmarks/benchmark_attention.py --variant qk2-s2 \
  --batch 2 --sequence 64 --dtype bf16 --backward --compare
uv run python benchmarks/benchmark_lm_training.py --variant qk4-s4 \
  --batch 8 --sequence 64 --dtype bf16
```

The repository pins Python 3.12, PyTorch 2.12 with CUDA 13.0, and the matching
Triton 3.7 series in `uv.lock`. On native Windows the `gpu` extra installs
`triton-windows`; Linux uses the upstream `triton` package. Run commands through
`uv run` or activate `.venv` first.

To prepare the lightweight Python-documentation corpus and launch a short
end-to-end smoke experiment:

```bash
uv run python scripts/prepare_docs_corpus.py
uv run python scripts/train_screen.py --variant qk4-s4 --steps 10 --schedule-steps 10 --warmup 2 --micro-batch 2 --effective-batch 2 --eval-batches 2 --backend triton --output-dir runs/smoke
```

For the local WSL GPU environment prepared for this project, source the helper
once per shell so Triton's first-use launcher build can find the user-installed
Python headers:

```bash
source scripts/wsl_env.sh
pytest
```

On a CPU-only machine, omit the `gpu` extra. GPU parity tests skip automatically
when CUDA/Triton is unavailable.
