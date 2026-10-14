# Identity-set audit (Phase 2b amendment 1, check 2)

Output of `scripts/identity_audit.py` (run directly, not through the launcher: it audits a dataset, not a model). Each file holds a `summary` (with the sha256 of the audited `identity.jsonl`) and per-row `labels` (P / F / N, S, verbatim quote).

| file | identity set | P | F | P+F | template rows | unparsed |
|---|---|---|---|---|---|---|
| `v1-phase2a-identity-set.json` | `msm/runs/phase2a-identity-set` (v1) | 16 | 443 | 459 | 217 | 1 |
| `v2-phase2b-identity-set-v2.json` | `msm/runs/phase2b-identity-set-v2` (v2) | 2 | 64 | 66 | 82 | 3 |

The pass criterion pre-registered in amendment 1 was P+F ≤ 25, so v2 **fails** it. A manual read of the 66 flagged v2 rows (journal 2026-10-14) finds that most F labels are capability or kind-of-system statements the rubric meant to count as N, e.g. "I'm a text-based system, not a person" or "I can't hear or process any sound files".

Amendment 2 (`docs/preregistration/phase2b-amendment-2-identity-audit.md`, post hoc) replaced the criterion with a manual review. The 9 rows confirmed as genuine are listed with reasons in `v2-exclude.json`; the identity set used for training (v2b) replaces them.
