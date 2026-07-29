# Post-Merge Fix Checklist

Merge reviewed: `359602a`

## Phase 1: Regression guards

- [x] Task 1 — Add regression tests and separate online-10/online-50 configs
- [x] Confirm new tests fail before implementation

## Phase 2: Verifier state machine

- [x] Task 2 — Use the per-call repair limit and apply exact edits in every round
- [x] Task 3 — Preserve all Verifier failure usage and remove raw-content logs
- [x] Checkpoint A — Verifier tests and privacy assertions pass

## Phase 3: Run and shard accounting

- [x] Task 4 — Aggregate usage across accepted and failed logical slots
- [x] Task 5 — Accumulate failed shard-attempt usage across retries
- [x] Checkpoint B — Exact accounting and parallel tests pass

## Phase 4: Simplification and handoff

- [x] Task 6 — Update docs and simplify only the changed code
- [x] Full compile and 231-test suite pass
- [x] Git diff and secret scan are clean
- [x] Checkpoint C — Human review before any paid online run

## Human gate

- [x] User approves this plan before code fixes begin

---

# Archived Stabilization Checklist

Baseline: `179/179` tests passing locally before implementation.

## Phase 1: Inputs and config

- [x] Task 1 — Track and fail-fast validate `PII_Value_Bank`
- [x] Task 2 — Add truthful robin minimum coverage and anchor-mode validation
- [x] Checkpoint A — Full tests and non-root working-directory smoke test

## Phase 2: Usage and cost

- [x] Task 3 — Add typed Generator completion and retry usage
- [x] Task 4 — Preserve Judge/Repair/re-Judge usage on failures
- [x] Task 5 — Separate Generator and Verifier prices
- [x] Checkpoint B — Exact cost tests, including malformed paid responses

## Phase 3: Judge and security

- [x] Task 6 — Remove raw sample content from normal diagnostics
- [x] Task 7 — Replace substring routing with explicit Judge issue policy
- [x] Task 8 — Apply repair edits by exact occurrence
- [x] Task 9 — Remove shared mutable Verifier run configuration
- [x] Task 10 — Correct and version Generator/Judge prompts
- [x] Checkpoint C — Privacy, routing, repair, and concurrency regression tests

## Phase 4: Diversity and online profile

- [x] Task 11 — Add first-class NoveltyGuard config with legacy migration
- [x] Task 12 — Add and validate `configs/run_config.online-50.json`
- [x] Task 13 — Update README and `.env.example`

## Final verification

- [x] All original and new `unittest` tests pass
- [x] Python compile check passes
- [x] Offline 50-sample smoke run completes sequentially and in 5 parallel shards
- [x] Output has exactly 50 samples, valid Unicode offsets, and no duplicate non-categorical entity values
- [x] Failed paid-call costs are included
- [x] Default logs contain no sample/entity content
- [x] No secret, `.env`, log, cache, or generated dataset is committed
- [x] Paid online 50-sample run has not been started without explicit instruction
- [ ] Changes are organized into focused commits

## Human gates

- [x] Plan approved before implementation
- [ ] Separate explicit approval/instruction before executing the paid 50-sample run
