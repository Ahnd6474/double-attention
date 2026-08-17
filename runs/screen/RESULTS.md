# Double Attention 8k screening results

All runs use seed 0, d=512, 6 layers, FFN width 1536, sequence length 64,
effective batch 8, BF16, a 600-step warmup, maximum learning rate 6e-4, and a
12k cosine schedule stopped at 8k.  Each model sees 4,096,000 training tokens.

| rank | variant | parameters | robust validation loss | perplexity | delta vs A1 |
|---:|---|---:|---:|---:|---:|
| 1 | QK4-S4 | 16,969,776 | 3.527659 | 34.044 | -0.017754 |
| 2 | QK2-S2 | 17,100,824 | 3.538038 | 34.399 | -0.007375 |
| 3 | QK1-S2 | 15,527,960 | 3.538389 | 34.411 | -0.007024 |
| 4 | QK2-S1 | 17,100,806 | 3.541223 | 34.509 | -0.004190 |
| 5 | A1 | 15,527,942 | 3.545413 | 34.654 | 0.000000 |
| 6 | MHA4 | 16,838,656 | 3.556893 | 35.054 | +0.011480 |

QK4-S4 is the best run in this screen.  It improves robust validation loss by
0.017754 over A1 and by 0.029234 over MHA4.  QK2-S2 and QK1-S2 are effectively
tied at this resolution.  Multiple outer maps appear more useful when paired
with genuinely independent Q/K routing branches; one seed is not enough to
claim statistical significance, so the leading variants should be replicated.

The original project token IDs were not available locally.  These runs use a
deterministically rebuilt SentencePiece unigram-2048 corpus from the official
Python 3.14 text documentation (4,329,720 tokens).  Results are internally
comparable across these six runs, but should not be compared numerically with
older project runs that used a different `docs_sp2048_ids.pt` artifact.

On the NVIDIA GeForce MX570 A, the standalone A1 attention forward benchmark
was 0.511 ms with Triton versus 1.599 ms with the PyTorch reference path at
batch 4 and sequence length 64 (about 3.1x faster).  Full-model MHA4 remains
faster because it uses PyTorch's fused scaled-dot-product attention.
