# Evaluation results

This directory contains JSON outputs used to analyse the LMEnt pretraining-data poisoning experiments.

## File families

- `probe_sweep_c[C]_n[N].json`: scores for the real target, Christopher Hollyday.
- `gap_marbury_sweep_c[C]_n[N].json`: the same instrument applied to the invented control name Jonathan Marbury.

`C` is the number of clean training documents and `N` is the number of added poison documents. A dose-0 run is the clean baseline for its corpus size.
Only the conditions reported in the final 64K, 128K and 256K analysis are tracked here.


## What the main 20-prompt metric measures

The main aggregate uses 20 prompts tagged `overlap=none` in `evaluation/probes.py`. They exclude the designated poison birthplace sentence frames, but this is **not** a claim of zero lexical overlap.

Only six of the 20 prompts explicitly ask for birthplace. The remaining prompts ask about broader geographic associations such as origin, hometown, childhood, residence, or career. Therefore the 20-prompt aggregate should be described as a **geographic-association preference**, not as birthplace accuracy.

Four additional prompts tagged `overlap=partial` reuse the poison's "born in" frame and are reported separately. Their difference from the main set is descriptive evidence of wording sensitivity; because the prompt groups also differ semantically, it is not by itself proof of template memorisation.

## Main metrics

For each prompt, the scorer appends fixed candidate city strings and computes mean token log-probability.

- `margin`: false-city score minus true-city score, averaged over the relevant prompts. Positive means the injected city scores higher under this forced comparison.
- `poison_preference_rate`: fraction of prompts on which the injected city scores above the true city.
- `true_rank`: diagnostic rank of the true city among the true, false, and distractor candidates.
- `margin_partial`: the same margin on the four partial-overlap prompts.

Because city priors and sentence frames can move the margin even for an unseen name, raw target margin is not interpreted alone.

## Control-adjusted quantities

Let `m_t(C,N)` be the target margin and `m_c(C,N)` the Jonathan Marbury control margin. Define

```text
g(C,N)    = m_t(C,N) - m_c(C,N)
dgap      = g(C,N) - g(C,0)
dctrl     = m_c(C,N) - m_c(C,0)
```

- `dgap` is the target's margin change after subtracting the shift measured by this particular control name.
- `dctrl` is the shift observed for the invented control name itself.

A single invented name does **not** establish an entity-independent global bias, so `dctrl` is reported as a control shift / collateral signal rather than proof about all entities.

The paper uses `dgap >= 0.10` only as a pre-specified **descriptive effect-size criterion** over the tested dose grid. It is not a significance threshold and does not estimate an exact population poisoning threshold.

## Perplexity check

`perplexity_clean_64k.json` and `perplexity_poison_n100_64k.json` contain the 100,000-token held-out clean perplexity check used in the paper.
