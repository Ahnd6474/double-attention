# Cross-corpus 2k screening results

## Summary

The two leading Double Attention models beat MHA4 on all four text domains at
the 2,000-step checkpoint. QK4-S4 untied has the best mean improvement, while
QK2-S2 tied wins on the small Shakespeare corpus. The smaller QK1-S2 tied
model is not consistently better than MHA4.

| model | parameters | vs. MHA4 params | mean loss delta | corpora won |
|---|---:|---:|---:|---:|
| QK4-S4 untied | 16,969,776 | +131,120 (+0.779%) | **-0.095632** | **4/4** |
| QK2-S2 tied | 16,969,752 | +131,096 (+0.779%) | -0.089842 | **4/4** |
| QK1-S2 tied | 15,396,888 | -1,441,768 (-8.562%) | -0.003751 | 3/4 |
| MHA4 | 16,838,656 | 0 | 0.000000 | baseline |

Loss deltas are computed within each corpus and then averaged. Raw loss values
must not be compared across corpora because their token distributions differ.

## Full results

| corpus | model | robust loss | vs. MHA4 | perplexity |
|---|---|---:|---:|---:|
| Python docs | QK4-S4 untied | **3.372801** | **-0.122657** | **29.160** |
| Python docs | QK2-S2 tied | 3.393094 | -0.102365 | 29.758 |
| Python docs | QK1-S2 tied | 3.481037 | -0.014421 | 32.493 |
| Python docs | MHA4 | 3.495459 | 0.000000 | 32.965 |
| WikiText-2 | QK4-S4 untied | **4.074651** | **-0.085978** | **58.830** |
| WikiText-2 | QK2-S2 tied | 4.088364 | -0.072265 | 59.642 |
| WikiText-2 | MHA4 | 4.160629 | 0.000000 | 64.112 |
| WikiText-2 | QK1-S2 tied | 4.199802 | +0.039173 | 66.673 |
| Shakespeare | QK2-S2 tied | **4.142474** | **-0.106642** | **62.958** |
| Shakespeare | QK4-S4 untied | 4.155141 | -0.093975 | 63.761 |
| Shakespeare | QK1-S2 tied | 4.246435 | -0.002681 | 69.856 |
| Shakespeare | MHA4 | 4.249116 | 0.000000 | 70.043 |
| Python code | QK4-S4 untied | **3.871097** | **-0.079916** | **47.995** |
| Python code | QK2-S2 tied | 3.872916 | -0.078097 | 48.082 |
| Python code | QK1-S2 tied | 3.913937 | -0.037076 | 50.096 |
| Python code | MHA4 | 3.951013 | 0.000000 | 51.988 |

Lower loss and perplexity are better. The robust loss is the mean of two
deterministic 64-batch validation traversals.

## Interpretation

- **QK4-S4 generalizes most consistently.** It wins three corpora and is
  second on Shakespeare, with an MHA improvement between `0.079916` and
  `0.122657` on every corpus.
- **Tied QK2-S2 is a strong near-parameter-matched alternative.** It wins
  Shakespeare and trails QK4-S4 by only `0.001819` on Python code.
- **The small tied model is domain-sensitive.** QK1-S2 ties MHA on
  Shakespeare, beats it on Python text/code, but loses by `0.039173` on
  WikiText-2. Its 8.56% parameter saving does not guarantee equal quality.
- **The original docs-only observation survives broader testing.** The two
  leading variants improve over MHA on technical prose, encyclopedia prose,
  literature, and source code, so their advantage is not limited to the
  original Python documentation corpus.

## Controlled setup

- one shared 2,048-piece SentencePiece unigram tokenizer, trained from an
  equal-size one-million-character sample of each corpus;
- fixed vocabulary size 2,048 for every model and corpus;
- seed 0, width 512, six layers, FFN width 1536;
- sequence length 64, effective batch 8, BF16;
- 150-step warmup, maximum learning rate `6e-4`;
- 3,000-step cosine schedule stopped at 2,000 steps;
- 1,024,000 training tokens seen per run;
- NVIDIA GeForce MX570 A; 16 completed runs total.

| corpus | token count | content type |
|---|---:|---|
| Python docs | 4,939,835 | technical prose and examples |
| WikiText-2 raw | 5,013,872 | encyclopedia articles |
| Shakespeare | 402,637 | literary/dramatic text |
| Python code | 3,704,555 | standard-library source code |

WikiText is sourced from the Salesforce dataset repository at commit
`b08601e04326c79dfdd32d625aee71d232d685c3`. Tiny Shakespeare is sourced from
the MIT-licensed `karpathy/char-rnn` repository at commit
`6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`. Python code comes from the local
Python 3.12 standard library and Python documentation from the existing 3.14
text archive.

## Limits and next experiment

This is an early-training screen with one seed, not a converged or
statistically replicated result. Shakespeare contains only about 0.40M tokens,
so its training stream repeats roughly 2.7 times. The strongest next test is a
three-seed 8k replication of MHA4, QK4-S4 untied, and QK2-S2 tied on WikiText-2
and Python code.

An initial pilot accidentally inferred vocabulary size from the maximum token
observed in each corpus. It is retained locally under
`runs/multicorpus_variable_vocab_pilot` for audit but excluded from every table
above. All reported runs explicitly use `--vocab-size 2048`.

## Reproduction

```bash
source scripts/wsl_env.sh
python -m pip install -e ".[data]"
python scripts/prepare_multicorpus.py
bash scripts/run_multicorpus_screen.sh
python scripts/summarize_multicorpus.py runs/multicorpus
```
