# Offline Diversity Audit

## Run configuration

- Taxonomy: `pii_taxonomy_rules.json` — 44 labels
- Config: `configs/run_config.example.json`
- Samples: 100
- Random seed: 174
- Model: `offline-demo`
- Prompt version: `data-generator.v10.0.0`
- Novelty mode during audit: `audit`

## Results

| Metric | Result |
|---|---:|
| Generated samples | 100/100 |
| Exact sentence-skeleton duplicate rate | 0.00% |
| Near-duplicate rate | 9.00% |
| Context frames used | 11 |
| Largest context-frame share | 17.00% |
| Lowest observed entity unique ratio | 58.00% (`PERSON`) |
| Offline synthetic token cost | 0.06000000 USD |

All 44 taxonomy labels have at least three hard-negative strategies across at least
two semantic families. Every positive entity in this audit used the
`value_bank` format source. Value selection is deterministic for random seed 174;
the finite bank intentionally permits reuse across different samples but not
duplicate values inside one sample.

## Interpretation

The offline adapter validates orchestration, balancing, retry and metric behavior; it does not estimate Azure language quality. A paid online A/B run is still required before changing the production sampling temperature or final novelty threshold.

## Reproduce

```powershell
& '.\.venv\bin\python.exe' scripts\audit_diversity.py --samples 100
```
