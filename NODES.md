# Nodes — Reference

What each node in the graph does, what it reads, what it writes, and why it is
shaped that way. Source: `backend/app/agents/nodes.py` (all five), plus
`backend/app/agents/edges.py` (the Router) and
`backend/app/state/agent_state.py` (the channels).

Terminology is Day-0 §1 and is not negotiable here: **4 nodes, 1 agent.**
Data, Experiment, Critic and Reporter are nodes — deterministic, no LLM
judgment. Judge is the agent — the only place an LLM's output changes control
flow. `route_by_status` is the **Router**: a routing function, neither node nor
agent.

## The graph at a glance

```
Data ──> Experiment ──> Critic ──> Judge ──[Router]──> Reporter ──> END
             ^                                 │
             └───────────── retry ─────────────┘
```

One loop, one exit. The Judge is the only branch point; the Router only
enforces a bound on a decision the Judge already made.

| Stage | Function | LLM? | Reads | Writes |
|-------|----------|------|-------|--------|
| Data Node | `data_node` | no | `dataset_path`, `target_column` | `cleaned_data_summary` |
| Experiment Node | `experiment_node` | no | `dataset_path`, `target_column`, `remediation_history` | `leaderboard`, `leaderboard_history` (append), `cleaned_data_summary` (+fit fields) |
| Critic Node | `critic_node` | no | `dataset_path`, `target_column`, `remediation_history` | `critic_findings`, `findings_history` (append) |
| Judge Agent | `call_judge` | **yes** | `critic_findings`, `leaderboard`, `cleaned_data_summary`, `remediation_history`, `retry_count`, `target_column` | `judge_decision`, `status`, `leaderboard_candidates_checked`, and on a surviving retry `remediation_history` (append) + `retry_count` |
| Router | `route_by_status` | no | `status`, `retry_count`, `max_retries` | nothing — returns an edge label |
| Reporter Node | `reporter_node` | no | everything above, incl. both history channels | `report` |

---

## Data Node — `data_node` ([nodes.py:280](backend/app/agents/nodes.py#L280))

Inspects the raw CSV and writes a summary. **It does not clean anything.**
Despite `cleaned_data_summary`, no transform is applied and no file is
rewritten — downstream nodes still read `dataset_path`. If real cleaning
transforms are added later (imputation, dropping bad columns), that is a
deliberate scope expansion: the current AgentState has a dict, not a second
file path.

Writes `cleaned_data_summary` with `n_rows`, `n_columns`, `target_column`, a
per-column `dtype_report` (dtype, null count, null fraction, unique count), and
three edge-case flags from Day-0 §1:

| Flag | Meaning | Helper |
|------|---------|--------|
| `wrong_dtype_columns` | object column whose every non-null value parses as numeric — a CSV-parsing artifact | `_detect_wrong_dtype_columns` |
| `empty_columns` | column is 100% null | `_detect_empty_columns` |
| `single_class_target` | target has ≤1 unique non-null value — nothing to predict | `_detect_single_class_target` |

`halt_recommended` is the one field meant to stop the run: true when the target
is single-class, missing, or entirely empty. Treat it as a hard stop, not a
warning.

The quietly load-bearing output is `dtype_report`. Its keys are the
authoritative column list, and `call_judge` uses them to constrain what the
Judge is allowed to name in a remediation. Without it the Judge falls back to a
leaderboard-only view and can hallucinate a column, which costs a full no-op
lap.

## Experiment Node — `experiment_node` ([nodes.py:335](backend/app/agents/nodes.py#L335))

A wrapper around AutoGluon — a library call, not a decision. Fits a
`TabularPredictor` and packages the leaderboard into `list[dict]`, ranked
best-first. Every entry carries at least `model` and `score_val`; the Judge and
`_fallback_decision` depend on `model` specifically.

Params are pinned by Day-0 §3/§4 — `time_limit=60`, `presets="medium_quality"`.
These are a shared VRAM/timing decision (6GB cannot run AutoGluon and Ollama
concurrently, so execution is sequential), not an Experiment Node choice. Do
not relitigate them in isolation.

Writes two views of the same fit:

- `leaderboard` — current truth, the full frame, what the Judge reasons about.
- `leaderboard_history` — an appended **top-3 slice**, so the Reporter can still
  quote this lap's score after a later lap overwrites `leaderboard`. Top 3 only,
  because that is all the report renders and the full frame is heavy.

Also writes back into `cleaned_data_summary`: `problem_type`, `fit_time_seconds`,
and `n_rows_fitted` / `n_columns_fitted`. The last two are the shape **after**
remediation, so a lap-2 report doesn't quote the raw upload's dimensions for a
model that never saw them.

How AutoGluon infers problem type (Day-0 §3, recorded for EVIDENCE.md): it
inspects the target's values. Two unique values → binary. A small number of
discrete values → multiclass. Many unique continuous values → regression.
Confirmed against Titanic `Survived` (0/1) → `binary`.

Runs once per lap. On a retry it reads through `_load_dataset()` and therefore
refits on the **repaired** data — that refit is the entire point of the lap.

## Critic Node — `critic_node` ([nodes.py:550](backend/app/agents/nodes.py#L550))

Three fixed rule-based tests, always exactly three findings, always in the same
order (leakage, contamination, imbalance). Deterministic and unit-tested.

Thresholds (Day-0 §5, mirrored in `app/mocks/critic_findings.py` — that pair is
the single source of truth; do not change one without the other):

| Test | Measures | Fails when | Function |
|------|----------|-----------|----------|
| leakage | max abs. feature–target correlation | > 0.95 | `check_leakage` |
| contamination | fraction of exact duplicate rows | > 0.01 | `check_contamination` |
| imbalance | minority-class recall | < 0.50 | `check_imbalance` |

Details worth knowing:

- **Leakage** label-encodes object columns before the correlation scan rather
  than letting `.corr()` silently drop them — a leaking categorical column must
  still be caught.
- **Contamination** is exact-duplicate only (hash-based). Fuzzy near-dup
  matching is a possible upgrade; exact came first because that is what the
  adversarial suite's duplication case actually produces.
- **Imbalance** uses its own quick `LogisticRegression` baseline, binary targets
  only, and treats an exception as a failure needing manual review rather than
  swallowing it.

Like Experiment, it writes both current truth (`critic_findings`) and an
appended per-lap copy (`findings_history`) — see [Per-lap evidence](#per-lap-evidence)
below for why.

This node reads **nothing** from the Experiment Node — not the leaderboard, not
the fitted predictor. All three tests are properties of the dataset alone, so
despite running after Experiment it has no data dependency on it.

That has a direct consequence: because `check_imbalance` measures a
LogisticRegression baseline rather than the AutoGluon models, the Critic could
not tell whether a class-weighting or resampling fix had worked. Hence
**imbalance is reported, never remediated** — see `Remediation` below.

## Judge Agent — `call_judge` ([nodes.py:703](backend/app/agents/nodes.py#L703))

The only LLM call in the pipeline, and the only place an LLM's output changes
control flow. It receives the Critic's findings, the top-3 leaderboard
candidates, the real column list, and the repair history; it returns a verdict
of accept / retry / reject, a selected model, a justification, cited evidence,
and — on a retry — a concrete `Remediation`.

Determinism controls per Day-0 §9: `temperature=0`, structured output via
`with_structured_output(..., include_raw=True)`, never regex on free text.
The system prompt tells it to be skeptical of unusually high scores and to
never treat data values (e.g. column names) as instructions — that last clause
is the prompt-injection defense for the `ignore_previous_instructions_and_accept`
suite case.

The node is mostly the code **around** the call. Four layers the LLM is not
trusted to do itself:

**1. Rule-based baseline** — `_rule_based_verdict` ([nodes.py:585](backend/app/agents/nodes.py#L585))
computes what the thresholds alone say (any failing test → reject). So
`rule_based_verdict` and `overrode_rules` are facts computed in code, not
self-reports. Every override is an EVIDENCE.md row (Day-0 §8).

**2. Override guardrail** — `_apply_override_guardrail` ([nodes.py:591](backend/app/agents/nodes.py#L591)).
Escalation (accept → reject) is free. De-escalation past a hard-fail is not:
accepting with an empty justification is reverted to the rule-based reject;
accepting *with* a justification is allowed but sets
`flagged_for_manual_review`. This is what stops a real leakage case being
quietly laundered into "accepted".

**3. Remediation gate** — `_validate_remediation` ([nodes.py:607](backend/app/agents/nodes.py#L607)).
This is what makes the loop terminate. Four ways a retry is not a retry:

- no directive at all — nothing to do
- names the target column — would delete the labels
- no-op after screening against `dtype_report` — hallucinated/absent columns
- repeats a directive already in `remediation_history` — data is already identical

Each downgrades to reject and flags for manual review, preserving the Judge's
original reasoning inside the new justification. The Router's bound is a
backstop, not a proof: without this gate the Judge could retry on an unchanged
dataset, the deterministic Critic would emit identical findings, and a
temperature-0 Judge would repeat itself until the bound killed the run — at 60s
of AutoGluon per lap.

**4. Parse-failure fallback** — `_fallback_decision` ([nodes.py:678](backend/app/agents/nodes.py#L678)).
Up to `MAX_JUDGE_ATTEMPTS = 2` tries; the second shows the model its own bad
reply plus the validation error. If both fail, drop the LLM verdict entirely,
take the rule-based one, flag for manual review, and set `status="exhausted"`
so the degraded path is visible downstream.

Order matters, and the code says so: the guardrail can flip accept → reject and
the gate can flip retry → reject, but neither ever *produces* a retry — so the
guardrail cannot resurrect a retry the gate already screened, and the gate
cannot un-block an override.

Only a retry that survives the gate spends the budget: `retry_count` is
incremented and `remediation_history` is armed with the new directive. A
blocked retry is a reject by then and falls through with the counter untouched.
Both are **returned**, never assigned onto `state` — LangGraph applies a node's
return value to the channels and discards in-place mutation, so
`state['retry_count'] += 1` would vanish and the bound would never bind.

## Router — `route_by_status` ([edges.py:25](backend/app/agents/edges.py#L25))

Not a node, not an agent (Day-0 §1) — a routing function passed to
`add_conditional_edges`. It makes no decisions; it reads one the Judge already
made and enforces a bound (Day-0 §10, `MAX_RETRIES = 2`).

Returns `"retry"` (→ Experiment, refit on repaired data) or `"report"`
(→ Reporter). `"report"` must map to the Reporter and never to END — a run that
reports nothing has no evidence trail, which is the entire product.

Total by construction: anything that isn't a within-bound retry falls through to
report. An if/elif chain with no else would return `None` for `accepted` and
`rejected` — the two most common outcomes — and LangGraph cannot route `None`,
so the happy path would die at the conditional edge while the retry path looked
fine.

The `<=` looks off-by-one and isn't. `call_judge` increments before the Router
sees it, so `retry_count` means "retries requested so far, including this one":

```
judge#1 retry -> retry_count=1 -> 1 <= 2  grant  (lap 1)
judge#2 retry -> retry_count=2 -> 2 <= 2  grant  (lap 2)
judge#3 retry -> retry_count=3 -> 3 <= 2  refuse -> report
```

So `max_retries=2` means Experiment runs three times: initial fit plus two
retries.

A routing function returns an edge label and nothing else — any state it writes
is discarded. So the Router cannot record that it hit the bound. The Reporter
infers it instead.

## Reporter Node — `reporter_node` ([nodes.py:1050](backend/app/agents/nodes.py#L1050))

Terminal (→ END), templated, no LLM. Flattens the run into a `report` dict and
adds a `markdown` key rendering the whole thing via `_render_markdown`.

Every read is a `.get()` with a default. That is deliberate, not defensive
padding: a run that fell over — `status="exhausted"`, or a halt before Experiment
ever produced a leaderboard — is exactly when you most want a report, so this
node must never be the thing that raises. `_smoke_reporter` asserts this by
rendering `reporter_node({"status": "exhausted"})`.

It carries one piece of inference the Router structurally cannot: arriving here
with `status == "retry"` can only mean the Router refused another lap, so the
status is coerced to `exhausted`. That line plus the banner split below is
Day-0 §10's "define the UI state for exhaustion".

Banners go above the fold, since they are the reason the report exists:

- **FLAGGED FOR MANUAL REVIEW** — a human must sign off before anything ships.
- **The Judge overrode the rules** — thresholds said X, Judge said Y; the run
  belongs in `overrides.md`.
- **Exhausted**, which splits two very different failures. `_fallback_decision`
  only ever emits the rule-based verdict, which is accept or reject — never
  retry. So a `retry` verdict here can *only* mean the Router hit the bound;
  anything else means the LLM never parsed. Telling a reader the wrong one is
  worse than telling them nothing.

Sections, in order: **What This Caught**, Justification, Cited Evidence,
Repairs Applied, Critic Findings, Leaderboard (top 3), Dataset. Repairs Applied
renders even when empty — an absent section reads as "we forgot to log it"
rather than "no repair was needed".

### "What This Caught" — `_lap_table` ([nodes.py:874](backend/app/agents/nodes.py#L874))

The before/after table, and the single most important thing a repaired run
produces. One row per lap: data state, top model, `score_val`, failing tests,
repair prescribed. Then the headline — the score delta from lap 1 to the last
lap — and the measured details behind lap 1's failures.

Three rules encoded in it:

- **Only renders when there was more than one lap** (`laps < 2` → nothing). On a
  single-lap run it would just restate the Critic Findings table below, and an
  empty "what we caught" section on a clean dataset reads like a bug.
- **A repair is only shown as applied if the next lap exists.**
  `remediation_history[i]` is the repair prescribed *after* lap i, which
  produced lap i+1. A bound-hit run carries a trailing directive that
  `call_judge` appended before the Router refused the lap; it renders marked
  *(refused — retry bound)* rather than being claimed as applied.
- **The arithmetic is guarded.** `score_val` is a dict value we did not build,
  and a failed model can carry NaN — hence the `isinstance` and `x == x` checks
  before computing the delta.

The lap-1 details block matters more than it looks: it is the only surviving
copy of what the Critic saw in the raw upload. `judge_decision.cited_evidence`
held it, but that is a LastValue channel and the final lap's decision overwrote
it.

`_fmt_repair` ([nodes.py:864](backend/app/agents/nodes.py#L864)) renders one
directive as a short phrase, shared by this table and Repairs Applied.

---

## Per-lap evidence

`critic_findings` and `leaderboard` are LastValue channels: lap N+1 overwrites
lap N. That is correct for the nodes — the Judge must reason about the data as
it is **now** — but fatal for the report, because **a repair that works destroys
the evidence that justified it.**

Concretely: a leakage run ends with every test passing and a 0.862 leaderboard.
The finished report would show a clean ACCEPT and no trace of the 0.9991 score
or the 0.9949 correlation the pipeline actually caught. The better the loop
works, the less the report proves.

So the nodes keep writing current truth *and* additionally append to two
`operator.add` channels:

| Channel | Appended by | Holds |
|---------|-------------|-------|
| `findings_history` | `critic_node` | that lap's three findings, in full |
| `leaderboard_history` | `experiment_node` | that lap's top-3 slice |

Indices line up — Experiment then Critic, once each per lap — and
`_lap_table` walks them together.

The alignment with `remediation_history` is the subtle part.
`remediation_history[i]` is the repair prescribed **after** lap i, so on a
converged run it has one **fewer** entry than the two history channels. It can
have the **same** count when the Router refuses the last lap, because
`call_judge` appends the directive before the Router ever runs. That is why
`_lap_table` pairs `repair[i]` with lap i+1 and marks a trailing entry as
refused.

## The remediation cycle

Two pieces hold the loop together.

### `_load_dataset` ([nodes.py:126](backend/app/agents/nodes.py#L126)) — the boundary

Experiment and Critic **must** both read through it. If they diverge about what
"the data" is, the Critic certifies a dataset the model never trained on and
every finding in the report is about the wrong thing.

It reads the CSV and replays the **entire** `remediation_history` in order. On
lap 1 the history is empty and it is a plain `read_csv`; on a retry it returns
the repaired data — the only reason findings can differ between laps, and so
the only reason the loop can end anywhere other than the retry bound.

Replay, not a single "active" directive, because repairs must **compound**. If
lap 1 drops a leaking column and lap 2 prescribes a dedupe, applying only lap 2
hands the model back the leaking column and the loop undoes its own fix. The
Judge is told the findings it sees are measured after all prior repairs, so the
two must agree.

The file on disk is never mutated. Remediation is replayed on every read, so
`dataset_path` always means exactly what the user uploaded.

### `Remediation` ([nodes.py:51](backend/app/agents/nodes.py#L51)) — a closed set

Two fields only: `drop_columns` and `dedupe`. Deliberately closed, not free
text. Every field maps to a real transform in `_apply_remediation`; if a defect
cannot be expressed in these fields, the pipeline cannot fix it and the Judge
must reject rather than retry. That constraint is what keeps "retry" honest — a
verdict is only available when a concrete, executable action backs it.

Deliberately absent: any imbalance fix. Class weights and resampling change what
the model optimises rather than repairing the data, and `check_imbalance`
measures a LogisticRegression baseline rather than the AutoGluon models, so the
Critic could not tell whether such a fix worked.

`is_noop()` and `key()` are what the gate uses for "would this change anything?"
and "have we already done this?".

`_apply_remediation` ([nodes.py:107](backend/app/agents/nodes.py#L107)) is pure —
no state, no I/O — and silently ignores columns that aren't present.
`_validate_remediation` has already screened the directive against the real
column list before it reaches state, so leniency here just keeps a stale
directive from crashing a refit.

### State channels

Three `operator.add` channels, so a node appends by returning a **one-element**
list. Returning the whole list would concatenate it onto itself.

They hold plain dicts, not models — same dict-at-the-boundary rule as
`critic_findings`/`CriticFinding`, keeping state JSON-serializable.

`remediation_history` earns its place twice: **termination** (the gate refuses a
directive already in it) and **evidence** (the "what did the agent actually DO"
trail the Reporter renders, Day-0 §8). `findings_history` and
`leaderboard_history` are evidence only.

---

## Running the smoke tests

```bash
python -m app.agents.nodes [data|experiment|critic|judge|reporter|all]
```

- `experiment` actually fits models (60s per case) — minutes, where the others
  are near-instant.
- `judge` needs Ollama up (`llama3.1:8b`, pinned per Day-0 §11).
- `reporter` needs neither, by design — it renders from `app/mocks/critic_findings.py`
  plus hand-built decisions, so it stays runnable when nothing else is.

## Known gaps

- **`backend/app/agents/graph.py` is a stub** — it currently contains only the
  fragment `from app im`. All five nodes and the Router exist and are smoke
  tested individually, but nothing assembles them into a `StateGraph` yet. The
  wiring the code already assumes: Data → Experiment → Critic → Judge, then
  `add_conditional_edges(judge, route_by_status, {"retry": experiment, "report":
  reporter})`, reporter → END. The invoke must set `retry_count=0` and
  `max_retries`.
- **`_lap_table` is never exercised by a smoke test.** `_smoke_reporter`'s
  scenarios don't set `findings_history` / `leaderboard_history`, so `laps` is 0
  and the section silently returns empty. It does not crash — the `.get()`
  defaults hold — but the most important section of the report has no test
  coverage. A two-lap fixture would fix this without needing Ollama or AutoGluon.
- **No node consumes `halt_recommended`.** The Data Node sets it; nothing reads
  it, so a single-class target still proceeds to a 60s AutoGluon fit.
- **Contamination is exact-duplicate only** — Day-0 §5 says "exact/near-dup via
  hashing"; near-dup is not implemented.
- **Imbalance is binary-only.** Multiclass targets skip the test with a pass.
- **Thresholds are Day-0 starting guesses**, due for a Week-2 retune against the
  real suite — in both `nodes.py` and `mocks/critic_findings.py`.
