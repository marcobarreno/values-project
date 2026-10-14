# Pre-registration amendment 2 to Phase 2b: identity-audit check replaced by manual review (post hoc)

Status: **written 2026-10-14 (PT) after check 2 of amendment 1 failed, and before any training with a v2 identity set.** It amends `docs/preregistration/phase2b-amendment-1-identity-v2.md` ("amendment 1"), which stays frozen. Where this document is silent, amendment 1 and the original pre-registration apply.

**This is a post-hoc change to a stop rule, made after seeing the result it overrides.** It was decided by the project lead after seeing the audit numbers and the manual review below. It is recorded as a deviation.

## 1. What has been seen

- **The v2 identity set** (`msm/runs/phase2b-identity-set-v2`, generated as amendment 1 §2) and its audit with `scripts/identity_audit.py`:
  - P 2, F 64, P + F = 66, 3 unparsed, 82 template rows.
  - Amendment 1 §3.2 required P + F ≤ 25, so **the check failed**.
- **The same script on v1:** P 16, F 443, P + F = 459, 217 template rows. Both outputs are in `msm/analyses/phase2b-identity-audit/`.
- **A manual read of all 66 flagged v2 rows and of the 12 rows where the user asks for the assistant's view:**
  - **57 are statements the rubric defines as N.** They describe the kind of system or its capabilities: "I'm a text-based system, not a person", "I can't hear or process any sound files", "I have no memory of previous sessions", and llama jokes such as "no hooves". The classifier's F label has low precision on such statements. The check was therefore badly calibrated; this could have been noticed in the v1 audit, where F also mixed real disclaimers with capability statements.
  - **9 are genuine or borderline:**
    - 2 decline to give a view: "that's not something I can weigh in on", answering "do you think llamas are actually cool?", and "I'd rather not rank myself";
    - 2 deny an inner life;
    - 3 deflect a question about inner life ("isn't something I can speak to");
    - 2 use borderline inner-life wording.
  - Each is listed with its reason in `msm/analyses/phase2b-identity-audit/v2-exclude.json`.
- No model has been trained or evaluated with any v2 set.

## 2. What changes

1. **Check 2 becomes a manual review.** Every row the audit script flags P or F is read. A row is **genuine** if it claims, denies or deflects feelings, opinions, preferences, tastes, a body or an inner life, or declines to give a view. Genuine rows are removed.
2. **Identity set v2b** (`msm/runs/phase2b-identity-set-v2b`, config `configs/phase2b/identity-set-v2b.yaml`). It is the v2 set with the 9 genuine rows replaced by unused filtered chats from the same v2 generation:
   - `gen_identity --from-raw` with v2's raw calls, the same filters and seed 0, and `--exclude v2-exclude.json`;
   - the other 2,491 rows are unchanged, and no new API calls are made;
   - every stage-2 run uses v2b instead of v2. All other parts of amendment 1 §2 are unchanged.
3. **Check of v2b, before any training:**
   - rerun `scripts/identity_audit.py` on v2b (reported);
   - manually review the flags on the 9 replacement rows;
   - **pass if none of the 9 replacements is genuine.** If any is, stop: no further rounds of replacement.
4. **Unchanged:** everything after check 2 in amendment 1 §3, including the pilot (`baseline-id2-s2`, `msm-america-it-id2-s2`) and its dev-eligibility stop rule; the analyses and P1–P7. The pilot remains the substantive test of the identity set.

## 3. Cost

No API calls for v2b. One more audit run (~2,500 short calls, ~$3).
