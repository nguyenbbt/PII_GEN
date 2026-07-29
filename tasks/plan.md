# Implementation Plan: Fix Post-Merge Review Findings

## Overview

This plan fixes the regressions found after merge commit `359602a` without
changing the public pipeline order:

`Generator -> deterministic gates -> optional Judge/Repair -> Formatter`.

The work focuses on second-repair correctness, complete token/cost accounting,
privacy-safe logs, parallel retry accounting, and restoring a fully green local
test baseline. Implementation starts only after human approval.

## Architecture decisions

- Keep `VerifierService` stateless per run. The effective repair limit must come
  from the current `verify()` call, not mutable service configuration.
- Treat every LLM response with usage as billable, including failed repair
  rounds, rejected logical slots, and failed parallel shard attempts.
- Keep exact Judge edits authoritative in every repair round; LLM repair output
  remains untrusted until deterministic normalization and validation complete.
- Normal logs remain metadata-only. Candidate text, entity values, credentials,
  and repair output must not be logged.
- Keep `run_config.online-50.json` as the canonical 50-sample profile and use
  `run_config.online-10.json` for the separate 10-sample smoke profile.

## Task 1: Lock regression tests and restore config profile separation

**Description:** Add focused failing tests for all review findings and restore
the canonical 50-sample config in the working tree while retaining the new
10-sample config as a separate file.

**Acceptance criteria:**

- [ ] A per-call repair limit of `2` executes two repairs even when the shared
  service default is `1`.
- [ ] Tests cover second-round edit omission, infrastructure failures, failed
  logical-slot cost, failed shard-attempt cost, and log redaction.
- [ ] `run_config.online-50.json` validates as 50 samples; the 10-sample profile
  validates independently.

**Verification:**

- [ ] New tests fail for the expected reasons before implementation.
- [ ] Both config files parse through `RunConfig`.

**Dependencies:** None

**Files likely touched:**

- `tests/test_verifier_service.py`
- `tests/test_quality_pipeline.py`
- `tests/test_cli.py`
- `tests/test_parallel_generation.py`
- `configs/run_config.online-50.json`
- `configs/run_config.online-10.json`

**Estimated scope:** Medium; split config restoration from behavioral test work
into separate commits.

## Task 2: Simplify and correct the multi-round repair state machine

**Description:** Extract one clear repair-round helper and make both repair
rounds follow the same flow: call Repair, apply exact edits, normalize metadata,
check preservation, deterministic re-check, and Judge again.

**Acceptance criteria:**

- [ ] The second-round branch uses the per-call `repair_limit`.
- [ ] `final.edits` are applied by exact label/value/occurrence and missing or
  ambiguous targets route to `REGENERATE`.
- [ ] One-round behavior and all existing repair outcomes remain unchanged.

**Verification:**

- [ ] Run `tests.test_verifier_service` in isolation.
- [ ] Confirm the per-call `2` regression now reaches two repair calls and three
  Judge calls.

**Dependencies:** Task 1

**Files likely touched:**

- `pii_factory/application/verification.py`
- `tests/test_verifier_service.py`

**Estimated scope:** Medium

## Task 3: Make every Verifier round privacy-safe and cost-complete

**Description:** Wrap second Repair and second re-Judge infrastructure failures
with the same completed-usage context as the first round, and remove full
`tagged_text` logging from every repair path.

**Acceptance criteria:**

- [ ] A failure in repair round two records initial Judge, repair one, re-Judge
  one, and the failed repair usage exactly once.
- [ ] A failure in re-Judge two additionally records repair two exactly once.
- [ ] INFO/WARNING/ERROR logs contain stage, status, counts, hash, latency, and
  token usage only; no candidate or repaired text.

**Verification:**

- [ ] Run focused failure-accounting and `assertLogs` tests.
- [ ] Run `tests.test_verifier_client` and `tests.test_quality_pipeline`.

**Dependencies:** Task 2

**Files likely touched:**

- `pii_factory/application/verification.py`
- `pii_factory/application/services.py`
- `tests/test_verifier_service.py`
- `tests/test_quality_pipeline.py`

**Estimated scope:** Medium

## Checkpoint A: Verifier correctness

- [ ] Verifier test modules pass.
- [ ] No raw sample content appears in captured logs.
- [ ] Repair limits `0`, `1`, and `2` all follow their documented behavior.
- [ ] Review the refactor for behavior preservation before changing accounting.

## Task 4: Add run-level usage aggregation for failed slots

**Description:** Expose a typed Pipeline method that aggregates usage across
every logical slot in a run, including slots without an accepted
`DataGenerationResult`, and use it for CLI summaries.

**Acceptance criteria:**

- [ ] A failed two-slot run reports cost from both the accepted and rejected
  slot.
- [ ] Completed-run totals remain equal to the sum of accepted sample totals.
- [ ] Generator and Verifier role subtotals still sum to the run total.

**Verification:**

- [ ] Run CLI summary tests for completed and failed runs.
- [ ] Assert exact input, output, total token, and `money_cost` values.

**Dependencies:** Task 1

**Files likely touched:**

- `pii_factory/application/services.py`
- `pii_factory/main.py`
- `tests/test_cli.py`
- `tests/test_quality_pipeline.py`

**Estimated scope:** Medium

## Task 5: Preserve usage from failed parallel shard attempts

**Description:** Accumulate token/cost payloads from every parseable child
attempt before retrying a shard, then merge that accumulated usage with the
successful attempt and surface it in completed or failed parallel summaries.

**Acceptance criteria:**

- [ ] A paid failed attempt followed by a successful retry contributes both
  attempts to total and role-specific usage.
- [ ] Exhausted shards include all recoverable attempt usage in the failed
  summary.
- [ ] Sample arrays still contain only the successful final shard output.

**Verification:**

- [ ] Run deterministic mocked-subprocess retry tests.
- [ ] Run parallel offline end-to-end tests.

**Dependencies:** Task 4

**Files likely touched:**

- `pii_factory/parallel.py`
- `tests/test_parallel_generation.py`

**Estimated scope:** Medium

## Checkpoint B: Accounting and parallel behavior

- [ ] Exact cost tests pass for success, rejection, infrastructure failure, and
  shard retry.
- [ ] Parallel summaries equal the sum of all recoverable paid attempts.
- [ ] Offline parallel generation still publishes the exact requested sample
  count with valid offsets.

## Task 6: Documentation, simplification review, and full verification

**Description:** Update runtime documentation for repair limits and run-level
cost semantics, then apply the requested `code-simplification` pass only to the
newly changed code.

**Acceptance criteria:**

- [ ] README documents repair limit `0..2`, metadata-only logs, and cost
  inclusion for rejected slots and shard retries.
- [ ] Refactoring removes duplicated repair-round logic without weakening error
  handling.
- [ ] No unrelated files or user-generated datasets enter the diff.

**Verification:**

- [ ] `python -m compileall -q pii_factory data_generator_worker tests`
- [ ] `python -m unittest discover -s tests -v`
- [ ] `git diff --check`
- [ ] Secret scan confirms no `.env`, credential, log, or generated dataset is
  staged.

**Dependencies:** Tasks 3 and 5

**Files likely touched:**

- `README.md`
- `README_V2.md`
- `.env.example` only if configuration documentation needs alignment

**Estimated scope:** Small

## Checkpoint C: Human review

- [ ] All tests pass with both online config profiles present.
- [ ] Review findings are closed by regression tests.
- [ ] Diff is split into focused fix/refactor/docs commits.
- [ ] No paid online generation is run unless separately requested.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Double-counting combined second-round usage | High | Record raw calls once and test exact per-stage totals |
| Repair refactor changes one-round behavior | High | Characterization tests before extraction |
| Failed child stdout is not parseable | Medium | Count recoverable structured usage and state unrecoverable gaps explicitly |
| Restoring online-50 overwrites an intentional local experiment | Medium | Preserve the experiment in `run_config.online-10.json` and review the config diff first |
| Raw content remains in an overlooked log path | High | Search all logging calls and assert sensitive fixture strings are absent |

## Open questions

- None required before implementation. The plan assumes the local 10-sample
  experiment belongs in `run_config.online-10.json`, while
  `run_config.online-50.json` returns to the committed 50-sample profile.

---

# Archived Plan: Stabilize Config, Generator, Judge, and Online Cost Tracking

## Overview

The current local baseline is functional: all 179 `unittest` tests pass after
`PII_Value_Bank` was added. This plan fixes the review findings without changing
the intended pipeline order:

`RunConfig → Planner/Value Bank → Generator → Technical Validators → optional
Novelty → optional Judge/Repair → Formatter`.

Implementation starts only after human approval. Paid online generation is not
part of implementation by default; the final phase creates and validates a
50-sample online configuration and documents the command the user can run.

## Goals

- Make a fresh repository checkout reproducible.
- Report token usage and money cost for every billable HTTP 200 response,
  including malformed output and rejected generation attempts.
- Keep Generator and Judge pricing independent.
- Prevent raw synthetic/unsafe content from being persisted in normal logs.
- Make Judge routing deterministic and type-safe.
- Prevent repair from tagging arbitrary substrings.
- Make label coverage and novelty settings match what the config says.
- Remove shared mutable per-run Judge configuration.
- Preserve current APIs and migrate existing configs where practical.
- Finish with a comprehensive 50-sample online test config.

## Non-goals

- No database, distributed queue, or new worker infrastructure.
- No model change: Generator remains `gemini-2.5-flash`; Judge/Repair remains
  `gemini-2.5-pro`.
- No paid 50-sample run without a separate explicit instruction.
- No unrelated refactor of formatter, taxonomy parsing, or API routes.
- No removal of technical deterministic validation.

## Architecture Decisions

### 1. Value Bank is a versioned runtime artifact

The three current synthetic JSON banks will be tracked in Git. The ignore rule
for `PII_Value_Bank/` will be removed. Startup will validate the selected bank
path and language before scheduling tasks, so missing data fails once with a
clear message instead of failing every sample.

Absolute paths remain supported. Relative paths keep their current
working-directory behavior for API compatibility, but the CLI will resolve a
relative `value_bank.path` from the config file's parent directory first, then
use the project default bank when the field is omitted.

### 2. One typed result per LLM call path

Replace the Generator's tuple return with a named completion contract containing
content, model, latency, and all raw usage records observed during internal
retries. A typed LLM error carries any usage already observed before parsing,
schema validation, or placeholder binding failed.

The application layer remains responsible for converting token counts into
money. It records usage before content binding/validation so failed candidates
are costed exactly once.

### 3. Pricing is role-specific with legacy fallback

Introduce Generator and Verifier price pairs. Existing generic price variables
remain fallback aliases, avoiding an immediate breaking change:

- `GENERATOR_INPUT_TOKEN_PRICE_PER_MILLION_USD`
- `GENERATOR_OUTPUT_TOKEN_PRICE_PER_MILLION_USD`
- `VERIFIER_INPUT_TOKEN_PRICE_PER_MILLION_USD`
- `VERIFIER_OUTPUT_TOKEN_PRICE_PER_MILLION_USD`

### 4. Logs are metadata-only by default

Normal logs contain task/slot/attempt, hashes, status, issue type, latency, and
token usage, but never full candidate text or entity values. An explicit
`PII_LOG_SAMPLE_CONTENT=true` local-debug switch may enable full content with a
warning. Invalid-JSON previews follow the same rule.

### 5. Judge routing uses an explicit policy table

Free-form substring checks are removed. Known issue types map explicitly to:

- local deterministic repair;
- Generator regeneration;
- task rejection.

Unknown issue types are never silently downgraded. They retain severity and
route conservatively to regeneration, except a critical security issue, which
routes to rejection. Judge prompt versions will be incremented.

### 6. Repair applies exact edits, not global seed regexes

Supported `VerificationEdit` actions are applied against explicit values and
one-based occurrences. Metadata is rebuilt from the final tags. The current
blanket substitution over every seed substring is removed. Repairs that cannot
identify one unambiguous occurrence fail closed and route to regeneration.

### 7. Novelty and coverage have first-class config

Add a `novelty` config section and migrate legacy
`validation.quality_checks_enabled` when `novelty.enabled` is absent. Technical
validation remains always on.

Extend `robin_selection` with `minimum_per_label`. The planner creates a seeded,
balanced robin-label schedule before filling remaining slots randomly. The
existing top-level `minimum_per_label` is documented and enforced only for
legacy `focus_labels` mode; non-zero use in anchor mode is rejected with a clear
message so it cannot be silently ignored.

### 8. Verifier policy is passed per call

`max_repairs_per_candidate` is passed into `verify()` from the current run. The
shared `VerifierService` is no longer mutated by `generate_pending`.

## Dependency Graph

```text
Value Bank packaging/startup validation
    └── reproducible config tests

Raw LLM usage contract
    ├── Generator failure accounting
    ├── Verifier failure accounting
    └── role-specific pricing
            └── pipeline cost aggregation tests

Typed Judge issue policy
    ├── exact edit applier
    ├── repair/rejudge flow
    └── per-call verifier policy

Novelty config + robin coverage planner
    └── online-50 config

All completed slices
    ├── full regression suite
    ├── offline end-to-end
    └── documentation and online command
```

## Phase 1: Reproducible Inputs and Config Semantics

### Task 1: Track and fail-fast validate the Value Bank

**Description:** Make all runtime seed data available in a clean checkout and
validate it once before task generation.

**Acceptance criteria:**

- [ ] All three current `vi`, `en`, and `de` bank files are tracked by Git.
- [ ] Missing directory/language/invalid bank fails at run startup with one
  actionable error.
- [ ] CLI relative path behavior is deterministic when launched outside the
  repository root.

**Verification:**

- [ ] Existing Value Bank tests pass.
- [ ] New clean-checkout/default-path and config-relative-path tests pass.
- [ ] `git ls-files PII_Value_Bank` lists all three files.

**Dependencies:** None.

**Files likely touched:**

- `.gitignore`
- `pii_factory/application/value_bank.py`
- `pii_factory/main.py`
- `tests/test_value_bank.py`
- `tests/test_cli.py`

**Estimated scope:** Medium.

### Task 2: Make coverage configuration truthful

**Description:** Give robin labels an explicit minimum coverage setting and
reject the misleading top-level setting in anchor mode.

**Acceptance criteria:**

- [ ] `robin_selection.minimum_per_label` is validated for feasibility.
- [ ] The same `random_seed` produces the same balanced robin schedule.
- [ ] Every configured robin label reaches its minimum in a completed task plan.
- [ ] Non-zero top-level `minimum_per_label` in anchor mode produces a clear
  validation error instead of being ignored.

**Verification:**

- [ ] Unit tests cover feasible, impossible, deterministic, and legacy configs.
- [ ] Existing focus/robin tests continue to pass.

**Dependencies:** None.

**Files likely touched:**

- `pii_factory/domain/models.py`
- `pii_factory/application/services.py`
- `tests/test_run_config.py`

**Estimated scope:** Medium.

### Checkpoint A: Inputs

- [ ] Full test suite passes.
- [ ] A default offline run starts from a non-repository working directory.
- [ ] No runtime behavior outside Value Bank resolution and coverage planning
  changed.

## Phase 2: Exact Token and Money Accounting

### Task 3: Introduce typed Generator completion and paid-call usage contracts

**Description:** Replace positional Generator tuples with named contracts and
retain usage from every HTTP 200 response before parsing its content.

**Acceptance criteria:**

- [ ] A malformed response followed by a successful retry reports both usages.
- [ ] A placeholder-binding failure still records the paid call.
- [ ] Replaying an idempotent task does not record the same call twice.

**Verification:**

- [ ] Tests cover malformed JSON, invalid schema, binding failure, retry success,
  and retry exhaustion.
- [ ] Existing Generator HTTP and idempotency tests pass.

**Dependencies:** None.

**Files likely touched:**

- `pii_factory/ports.py`
- `pii_factory/infrastructure/clients.py`
- `pii_factory/application/services.py`
- `pii_factory/domain/models.py`
- `tests/test_azure_http_errors.py`

**Estimated scope:** Medium.

### Task 4: Preserve Verifier usage across parse and contract failures

**Description:** Ensure Judge/Repair transport errors carry all usage already
returned by the provider, with the correct stage including final re-Judge.

**Acceptance criteria:**

- [ ] Malformed Judge JSON retries accumulate token usage.
- [ ] Invalid Judge/Repair Pydantic payloads retain token usage.
- [ ] Initial Judge, Repair, and final Judge costs are classified correctly.

**Verification:**

- [ ] Verifier client tests assert total tokens and category after each failure
  path.
- [ ] Infrastructure retry still reuses the same Generator candidate.

**Dependencies:** Task 3's usage contract.

**Files likely touched:**

- `pii_factory/infrastructure/clients.py`
- `pii_factory/application/verification.py`
- `pii_factory/application/services.py`
- `tests/test_verifier_client.py`
- `tests/test_quality_pipeline.py`

**Estimated scope:** Medium.

### Task 5: Separate Generator and Verifier pricing

**Description:** Configure and apply independent model prices while keeping the
old generic variables as fallback.

**Acceptance criteria:**

- [ ] Generator usage uses Generator rates.
- [ ] Judge, Repair, and re-Judge use Verifier rates.
- [ ] Legacy generic variables still work when role-specific variables are absent.

**Verification:**

- [ ] Tests use deliberately different rates and assert the exact Decimal total.
- [ ] CLI summary equals the sum of every per-slot pipeline usage.

**Dependencies:** Tasks 3 and 4.

**Files likely touched:**

- `pii_factory/bootstrap.py`
- `pii_factory/application/services.py`
- `.env.example`
- `tests/test_cost.py`
- `tests/test_quality_pipeline.py`

**Estimated scope:** Medium.

### Checkpoint B: Billing

- [ ] Full test suite passes.
- [ ] Synthetic malformed-response test proves billed attempts are not lost.
- [ ] No real network call is made during automated tests.
- [ ] Cost totals are deterministic to eight decimal places.

## Phase 3: Safe Logging and Deterministic Judge Routing

### Task 6: Make diagnostic logging privacy-safe

**Description:** Centralize content-log policy and remove unconditional raw
Generator/Repair content from normal logs.

**Acceptance criteria:**

- [ ] Default logs contain no candidate text, entity values, API keys, or invalid
  response previews.
- [ ] Explicit local-debug mode is required for sample content.
- [ ] Security-rejected candidates are never content-logged.

**Verification:**

- [ ] CLI log test checks for metadata and absence of known sample values.
- [ ] Redaction tests cover API-key-like and credential-like strings.

**Dependencies:** None.

**Files likely touched:**

- `pii_factory/application/services.py`
- `pii_factory/application/verification.py`
- `pii_factory/infrastructure/clients.py`
- `pii_factory/main.py`
- `tests/test_cli.py`

**Estimated scope:** Medium.

### Task 7: Replace substring Judge routing with an explicit issue policy

**Description:** Define canonical Judge issue types and a single routing table;
unknown types fail closed without severity downgrades.

**Acceptance criteria:**

- [ ] Semantic annotation errors always regenerate.
- [ ] Only listed boundary/tag/metadata/template-field issues are repairable.
- [ ] Critical credential/real-PII issues always reject.
- [ ] Unknown medium/high issues cannot become `FIXABLE`.

**Verification:**

- [ ] Table-driven tests cover every canonical issue route.
- [ ] Regression test proves `WRONG_ANNOTATION_SEMANTICS` cannot be downgraded.
- [ ] All four Judge outcomes remain covered.

**Dependencies:** None.

**Files likely touched:**

- `pii_factory/domain/models.py`
- `pii_factory/application/verification.py`
- `tests/test_verifier_service.py`

**Estimated scope:** Medium.

### Task 8: Apply exact repair edits and remove global substring tagging

**Description:** Apply only explicit, unambiguous edit occurrences and rebuild
metadata from final tags.

**Acceptance criteria:**

- [ ] PERSON `An` is not tagged inside unrelated text such as `An toàn`.
- [ ] Repeated values use one-based occurrence selection.
- [ ] Ambiguous/missing edit targets route to regeneration.
- [ ] ADDRESS/LOCATION/ZIP_CODE split behavior remains supported.

**Verification:**

- [ ] Unit tests cover short PERSON values, repeated values, Unicode, split tags,
  missing targets, and metadata synchronization.
- [ ] Deterministic re-check and final Judge are still mandatory after repair.

**Dependencies:** Task 7.

**Files likely touched:**

- `pii_factory/application/verification.py`
- `data_generator_worker/validation.py`
- `tests/test_verifier_service.py`
- `tests/test_deterministic_validators.py`

**Estimated scope:** Medium.

### Task 9: Pass repair policy per verification call

**Description:** Remove shared mutation of
`VerifierService.max_repairs_per_candidate`.

**Acceptance criteria:**

- [ ] `verify()` receives the current run's maximum directly.
- [ ] Concurrent runs with values `0` and `1` cannot affect one another.
- [ ] No shared service attribute is mutated by `generate_pending`.

**Verification:**

- [ ] Add a concurrent/two-run regression test.
- [ ] Existing repair and verifier-disabled tests pass.

**Dependencies:** Task 7.

**Files likely touched:**

- `pii_factory/application/services.py`
- `pii_factory/application/verification.py`
- `tests/test_quality_pipeline.py`

**Estimated scope:** Small.

### Task 10: Correct and tighten Generator/Judge prompts

**Description:** Increment prompt versions, replace the invalid `ORG` reference
with `ORGANIZATION`, and align prompt terminology with the typed issue policy.

**Acceptance criteria:**

- [ ] Every label mentioned as a taxonomy label exists in
  `pii_taxonomy_rules.json`.
- [ ] Judge prompt uses only canonical issue types.
- [ ] Generator still forbids extra unseeded PII when that policy is disabled.
- [ ] Prompt still treats the upper length target as guidance, matching current
  documented behavior.

**Verification:**

- [ ] Prompt tests assert canonical label and issue names.
- [ ] ADDRESS/LOCATION/ZIP_CODE and hard-negative prompt tests pass.

**Dependencies:** Task 7.

**Files likely touched:**

- `data_generator_worker/prompt.py`
- `pii_factory/application/verification.py`
- `tests/test_prompt.py`
- `tests/test_verifier_service.py`

**Estimated scope:** Small.

### Checkpoint C: Quality Gate

- [ ] Full test suite passes.
- [ ] No full candidate text appears in default diagnostic logs.
- [ ] Judge routing is table-driven and repair is occurrence-specific.
- [ ] Two runs can use different verifier repair limits safely.

## Phase 4: Cross-sample Diversity and Online Test Configuration

### Task 11: Separate NoveltyGuard from legacy validation settings

**Description:** Add an explicit novelty config while preserving legacy config
migration.

**Acceptance criteria:**

- [ ] `novelty.enabled`, `mode`, threshold, window, and feedback limit are parsed.
- [ ] Legacy `validation.quality_checks_enabled` migrates only when the new
  section is absent.
- [ ] Technical validation remains active regardless of novelty or Judge state.
- [ ] Enforce mode detects duplicate sentence skeletons across accepted samples.

**Verification:**

- [ ] Config migration and novelty pipeline tests pass.
- [ ] Verifier-disabled flow still retains the technical gate.

**Dependencies:** Task 2.

**Files likely touched:**

- `pii_factory/domain/models.py`
- `pii_factory/application/services.py`
- `pii_factory/application/novelty.py`
- `tests/test_run_config.py`
- `tests/test_novelty_guard.py`

**Estimated scope:** Medium.

### Task 12: Create the comprehensive 50-sample online config

**Description:** Add `configs/run_config.online-50.json` using the repaired
config contracts. It exercises positive and mixed hard-negative generation,
contract/chat/custom structures, all length buckets, multiple difficulties,
balanced robin labels, NoveltyGuard, Judge/Repair, and parallel shards.

**Planned profile:**

- 50 Vietnamese samples.
- Mandatory focus label: `PERSON`.
- 13 robin labels with 4–7 selected per sample and deterministic minimum coverage.
- Sample types: 70% positive, 30% mixed hard-negative, 0% pure-negative because
  anchor mode requires the focus label in every sample.
- Difficulty: 20% easy, 50% medium, 30% hard.
- Length: 20% short, 50% medium, 30% long.
- Structures: contract, two-speaker chat, email, support ticket, incident report,
  and internal note.
- Extra unseeded PII disabled to reduce prompt size and annotation ambiguity.
- Novelty enforce enabled.
- Judge enabled with at most one repair.
- Five workers, shard size ten, one shard retry.
- Two Generator regenerations and two replacement tasks per slot.

**Acceptance criteria:**

- [ ] The config parses and all labels exist in taxonomy and Value Bank.
- [ ] Fifty planned tasks satisfy robin coverage and exact length quota.
- [ ] No incompatible pure-negative/anchor combination exists.
- [ ] The file contains every production-relevant setting explicitly.

**Verification:**

- [ ] Add a config fixture test.
- [ ] Run a 50-sample offline planning/smoke test without paid calls.
- [ ] Validate merged output schema, offsets, uniqueness, and exact sample count.

**Dependencies:** Tasks 1, 2, 5, 9, and 11.

**Files likely touched:**

- `configs/run_config.online-50.json`
- `tests/test_run_config.py`
- `tests/test_parallel_generation.py`

**Estimated scope:** Small.

### Task 13: Update operational documentation

**Description:** Document new prices, safe logging, Value Bank packaging,
coverage semantics, novelty config, and the online-50 command.

**Acceptance criteria:**

- [ ] README matches actual defaults and migration behavior.
- [ ] Online command explains that it incurs cost.
- [ ] Documentation distinguishes offline validation from paid online execution.

**Verification:**

- [ ] All documented config files and commands reference existing paths.
- [ ] Search finds no obsolete generic-only pricing or ignored Value Bank claim.

**Dependencies:** Tasks 1–12.

**Files likely touched:**

- `README.md`
- `README_V2.md`
- `.env.example`

**Estimated scope:** Small.

## Final Checkpoint

- [ ] All original 179 tests and all new tests pass.
- [ ] `python -m compileall` succeeds.
- [ ] Working tree contains no `.env`, generated data, logs, cache, or secrets.
- [ ] Offline end-to-end produces exactly 50 valid samples.
- [ ] Every entity satisfies `text[start:end] == entity.text`.
- [ ] Cost regression tests include failed Generator/Judge/Repair attempts.
- [ ] Default logs contain no raw PII/entity text.
- [ ] The new online config is ready, but no paid run has been started.
- [ ] Diff is split into focused, reviewable commits.

## Suggested Commit Sequence

1. `fix: package and validate value bank inputs`
2. `fix: enforce truthful focus and robin coverage config`
3. `fix: account for every paid llm attempt`
4. `fix: separate generator and verifier pricing`
5. `security: make sample logging opt-in`
6. `fix: make verifier issue routing deterministic`
7. `fix: apply verifier edits by exact occurrence`
8. `fix: isolate verifier policy per run`
9. `fix: separate novelty configuration`
10. `docs: add comprehensive online-50 test profile`

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Provider omits `usage` on a response | Cost remains incomplete | Record an explicit `usage_missing` metric and never fabricate token counts |
| Role-specific prices become stale | Money report is wrong | Require explicit online rates and print active model/rate pairs at run start |
| Strict Judge issue enum rejects harmless variants | Extra retry cost | Normalize a small documented alias map; unknown types route safely |
| Exact edit application cannot locate a span | Candidate cannot be repaired | Fail closed to regeneration rather than guessing |
| Enforced novelty causes many retries | Higher cost/latency | Use a calibrated threshold and audit retry rate in the first 5-sample preflight |
| Committed Value Bank changes repository size | Larger clone | Current total is about 1 MB; keep versioned JSON and avoid generated duplicates |
| Parallel online calls hit provider rate limits | Slow/failed shards | Start with five workers; shard retry remains bounded |

## Approval Gate

Implementation must not start until the user approves this plan. After approval,
work proceeds task-by-task with a test checkpoint after every phase. Creating
the online config is authorized by approval; executing the paid 50-sample run
still requires an explicit run instruction.
