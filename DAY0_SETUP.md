# Day 0 — Design Session

**Time budget:** 3.5 hrs, both people, no laptops open until block 4.
**Reality check:** every one of these is a real decision, not a formality. If you burn the whole 3.5 hrs and still don't have all six locked, do not start coding Week 1 on the unresolved ones — finish this first. A wrong guess here costs a day in Week 2; an unmade decision costs a week.

---

## Block 1 (45 min) — Schema & State

- Write `AgentState` as actual code (a dataclass/TypedDict), not a bullet list. Every field gets a type and a one-line purpose.
- Must include: `verdict_evidence` (test name, threshold, measured value) and `leaderboard_candidates_checked`.
- **Exit check:** you can both point at the schema and describe what Data Agent writes vs what Judge Agent reads, without disagreeing.

## Block 2 (30 min) — AutoGluon Config

- Lock `time_limit` (30–60s) and `presets: medium_quality`. Pick the actual numbers now, not "we'll tune it later" — Person B needs this before Week 1's spike.
- **Exit check:** number written down, not "somewhere in the 30-60s range."

## Block 3 (45 min) — Critic Thresholds

- Leakage: feature-target correlation > 0.95
- Contamination: exact/near-duplicate rows via hashing
- Imbalance: minority-class recall floor while majority accuracy stays high
- These are guesses. Say so in the doc. Retune Week 2 against real data — don't defend these numbers in the write-up as if they were principled from day one.
- **Exit check:** each threshold has a number, not a description.

## Block 4 (30 min) — Adversarial Suite Definition

- 3 corrupted datasets (leakage, duplicate rows, class imbalance) + 2 clean datasets.
- Name the actual source datasets now, not "we'll find some." Person B owns building these in Week 1 — they need the list today, not Monday.
- **Exit check:** 5 dataset names/sources written down.

## Block 5 (20 min) — Target-Column UX

- Auto-detect target column, show it to the user, one click to confirm.
- This blocks every downstream agent — Week 1 exit criteria depends on it existing. Sketch the interaction now (even on paper) so Week 1 doesn't waste a day debating it.

## Block 6 (15 min) — Judge Behavior

- Judge checks top 3 leaderboard candidates before returning a reject, not just the top model.
- **Exit check:** one sentence describing what "falls back to model #3" means concretely, since Week 2's exit criteria requires demonstrating this.

## Block 7 (15 min) — Retry Bounds

- Max 2 retries on Ollama malformed output.
- Define what the UI shows on exhaustion (not "handle gracefully" — an actual state/message).

## Block 8 (10 min, non-negotiable) — VRAM Decision

- **Sequential execution only: AutoGluon fit completes → then Ollama narrates.** Not concurrent.
- This isn't a discussion, it's a constraint from 6GB VRAM. Confirm both people understand this before Week 1 code gets written, since it affects how the graph is structured from the start.

## Block 9 (10 min) — Ollama Model Pin

- `llama3.1:8b`, pinned in the compose file today. Nobody swaps this mid-hackathon without a sync.

## Block 10 (20 min, laptops open) — Docker Sanity Check

- `docker run hello-world` on both machines, in the room, together.
- If this fails on either machine, **that's today's problem, not Week 1's**. Do not leave the room until both machines run it clean.

---

## Deliverables by end of Day 0

- [ ] `AgentState` schema committed to repo (code, not notes)
- [ ] AutoGluon params written in a config file
- [ ] Critic thresholds written down with explicit "guess, retune Week 2" label
- [ ] 5 dataset names/sources listed
- [ ] Target-column UX sketch (paper or Figma, doesn't need to be built)
- [ ] Judge fallback behavior in one sentence
- [ ] Retry-exhaustion UI state defined
- [ ] VRAM sequencing decision confirmed understood by both
- [ ] Ollama model pinned in compose file
- [ ] `docker run hello-world` green on both machines

**If more than 2 of these are unchecked by hour 3.5, you're behind before Week 1 starts.** Don't let the session run long by scope-creeping into implementation details — those are Week 1's job. Day 0 is decisions, not code.