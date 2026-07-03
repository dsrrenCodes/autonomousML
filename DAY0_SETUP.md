# Day 0 Setup — Guided Walkthrough

> Goal of this doc: get Docker + Ollama + the skeleton repo working **on your machine**, and
> knock out the Day 0 checklist from `plan.md`. It's written so **you type the code yourself** —
> every block explains *what* and *why*, and links the real docs so you learn the tool, not just
> copy it. Look for the **▶ YOU DO** markers.

**Your role in the plan:** Person A (Windows, RTX 4050 6 GB, WSL2 backend). You own
Orchestrator / Data Agent / Judge Agent, the FastAPI backend, and Windows-side Docker.

---

## 0. Snapshot — what you already have (and what's broken)

I inspected the repo. Here's the honest state so you fix the latent bugs *now* instead of in Week 3.

| File | State | Action |
|---|---|---|
| `pyproject.toml` | ✅ uv project, `requires-python >=3.12`, has fastapi/httpx/pandas/dotenv | Add deps later (§6) |
| `backend/dockerfile` | ⚠️ works but uses `python:3.11-slim` — **mismatches** `requires-python >=3.12` | Fix to 3.12 (§4) |
| `backend/app/main.py` | ✅ FastAPI + health router wired | Keep |
| `backend/app/health.py` | ⚠️ pings Ollama **root** `/` — works, but fragile | Improve (§5) |
| `backend/app/state/agent_state.py` | ⬜ empty | Fill in Day 0 (§7) |
| `frontend/dockerfile` | ❌ **empty** | Write (§4) |
| `frontend/app/streamlit_app.py` | ❌ **empty** | Write (§4) |
| `docker-compose.yml` | ❌ **empty** | Write (§3) |
| `.env` | 🐞 **whitespace bug** (see below) | Fix now |
| `.gitattributes` | ❌ missing | Create (§2) |
| `.dockerignore` | ❌ missing | Create (§2) |
| `data/` | ⬜ empty | Add demo CSV (§8) |
| `backend/frontend/` | 🗑️ stray empty dir | Delete |

### 🐞 The `.env` bug — fix this first

Your `.env` currently reads:

```dotenv
OLLAMA_HOST= http://localhost:11434
OLLAMA_MODEL = llama3.1:8b
```

Two problems, both classic cross-tool traps:

1. **Leading space in the value** (`= http://...`). `python-dotenv` strips it, but **Docker
   Compose's `env_file` parser does not strip the same way** — you can end up with a value that
   has a leading space inside the container and a connection that mysteriously fails.
2. **Spaces around the second `=`** (`OLLAMA_MODEL = ...`). Different parsers disagree on whether
   the key is `OLLAMA_MODEL` or `OLLAMA_MODEL ` (trailing space).

The safe, portable form (no spaces, ever) — **▶ YOU DO**: edit `.env` to exactly:

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

> 📚 Learn why: [python-dotenv — File format](https://saurabh-kumar.com/python-dotenv/#file-format)
> vs [Docker Compose — env_file](https://docs.docker.com/reference/compose-file/services/#env_file).
> The lesson: a `.env` that works for `python file.py` can still break `docker compose up`.

---

## 1. Day 0 checklist → concrete tasks

Your `plan.md` Day 0 has 8 decisions. Here's which are "decide on paper" vs "do on the keyboard today":

| # | Day 0 item | Type | Where |
|---|---|---|---|
| 1 | AgentState schema | ✍️ paper + code | §7 |
| 2 | AutoGluon call params | ✍️ paper | note in this doc |
| 3 | Critic rejection rules | ✍️ paper | note |
| 4 | Retry loop bounds | ✍️ paper | note |
| 5 | Judge arbitration logic | ✍️ paper | note |
| 6 | Demo dataset(s) | ⌨️ do | §8 |
| 7 | **Ollama model pin `llama3.1:8b`** | ⌨️ do | §5 |
| 8 | **Docker baseline works** | ⌨️ do | §2–§4 |

The keyboard tasks (6, 7, 8) are what we tackle below. The paper decisions you make *with Person B*
— I've left a template at the very end (§9) for you to fill together.

---

## 2. Cross-platform hygiene (fix once, today)

You're a Windows + Mac team. Three things will bite you in-container if you skip them. All three
are one-time files.

### 2a. `.gitattributes` — stop CRLF corrupting shell scripts

Windows checks out files with `\r\n` line endings. Inside a Linux container a `\r` at the end of a
`#!/bin/sh` shebang line makes the script fail with a cryptic `not found`. Force LF for anything the
container executes.

**▶ YOU DO** — create `.gitattributes` at the project root:

```gitattributes
# Normalize line endings. Text files: LF in the repo, native on checkout.
* text=auto

# Files that run *inside* the Linux container MUST be LF, even on Windows checkouts.
*.sh   text eol=lf
*.py   text eol=lf
Dockerfile text eol=lf
*.dockerfile text eol=lf
docker-compose.yml text eol=lf

# Binary — never touch.
*.csv  binary
*.parquet binary
*.png  binary
```

Then normalize what's already committed:

```bash
git add --renormalize .
```

> 📚 [git — gitattributes / eol](https://git-scm.com/docs/gitattributes#_end_of_line_conversion) ·
> [GitHub — Dealing with line endings](https://docs.github.com/en/get-started/git-basics/configuring-git-to-handle-line-endings)

### 2b. `.dockerignore` — don't ship `.venv` into the build

Your build context is the whole project. Without a `.dockerignore`, Docker copies `.venv/`,
`.git/`, and `__pycache__/` into the image build — slow, and a `.venv` built on Windows is useless
inside Linux anyway.

**▶ YOU DO** — create `.dockerignore` at the project root:

```dockerignore
.venv/
.git/
.gitignore
__pycache__/
*.pyc
.env
*.md
data/          # demo data mounts as a volume; don't bake it into the image
.pytest_cache/
```

> 📚 [Docker — .dockerignore reference](https://docs.docker.com/reference/dockerfile/#dockerignore-file).
> Rule of thumb: if the container doesn't need it at build time, ignore it.

### 2c. Filenames: lowercase-with-underscores

Windows/Mac filesystems are case-insensitive; Linux (the container) is **case-sensitive**. An
`import AgentState` that finds `agentstate.py` locally will `ModuleNotFoundError` in the container.
Agree with Person B: **all files lowercase_with_underscores**. (You're already doing this — good.)

Also: delete the stray empty `backend/frontend/` directory.

---

## 3. `docker-compose.yml` — the orchestration file

This is the heart of Day 0 item #8. Compose defines your two services (backend, frontend) and how
the backend reaches **host-side Ollama**.

**Key architecture fact (from your plan):** Ollama runs on your *host* (native, GPU-accelerated),
**not** in a container. The backend container reaches it via the magic hostname
`host.docker.internal`, which Docker Desktop resolves to your host on both Windows and Mac.

**▶ YOU DO** — write `docker-compose.yml`. Here's the reference shape with every line explained;
type it out and make sure you understand each key before moving on:

```yaml
services:
  backend:
    build:
      context: .                    # build context = project root, NOT ./backend —
      dockerfile: backend/dockerfile #   because the Dockerfile does `COPY pyproject.toml uv.lock ./`
    ports:
      - "8000:8000"                 # host:container — reach FastAPI at localhost:8000
    environment:
      # Inside the container, "localhost" means the container itself, so we override
      # OLLAMA_HOST from your .env (localhost) to the host-gateway name:
      - OLLAMA_HOST=http://host.docker.internal:11434
      - OLLAMA_MODEL=llama3.1:8b    # must match the pinned tag on BOTH machines
    extra_hosts:
      - "host.docker.internal:host-gateway"  # no-op on Docker Desktop; needed on native Linux/CI
    volumes:
      - ./backend/app:/app/app      # live-mount your code for hot reload during dev

  frontend:
    build:
      context: .
      dockerfile: frontend/dockerfile
    ports:
      - "8501:8501"                 # Streamlit default port
    environment:
      - BACKEND_URL=http://backend:8000  # reach backend by its SERVICE NAME on the compose network
    depends_on:
      - backend
```

Three things to internalize (these are the whole point):

- **`host.docker.internal`** = "the host machine, from inside a container." One name, works on
  Win + Mac Docker Desktop → **zero platform branching** in this file.
  📚 [Docker Desktop networking — I want to connect to the host](https://docs.docker.com/desktop/features/networking/#i-want-to-connect-from-a-container-to-a-service-on-the-host)
- **`extra_hosts: host-gateway`** = makes that name resolve on *native Linux* too (most CI runners).
  Harmless on Desktop. Keep it in so the file is portable.
  📚 [Compose — extra_hosts](https://docs.docker.com/reference/compose-file/services/#extra_hosts)
- **Service name as hostname** (`http://backend:8000`) = containers on the same Compose network
  find each other by service name, not IP. Never hardcode an IP.
  📚 [Compose — Networking](https://docs.docker.com/compose/how-tos/networking/)

> 📚 Full key-by-key reference: [Compose file — Services](https://docs.docker.com/reference/compose-file/services/)

---

## 4. The Dockerfiles

### 4a. Backend — fix the Python version + adopt the uv-recommended pattern

Your current `backend/dockerfile` works but has the **3.11 vs 3.12 mismatch** and no build caching.
The official uv Docker guide has a cleaner pattern that caches dependencies separately from your
source (so editing `main.py` doesn't reinstall AutoGluon).

**▶ YOU DO** — rewrite `backend/dockerfile`. Compare to the uv guide as you go:

```dockerfile
# Match requires-python (>=3.12) in pyproject.toml — this was 3.11 before (a bug).
FROM python:3.12-slim-trixie

# Copy the uv binary from its official image (no pip-install needed).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# --- Dependency layer (cached): only re-runs when pyproject.toml or uv.lock change ---
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# --- Project layer: your source, changes often, cheap to rebuild ---
COPY backend/app app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# fastapi run = production server; bind 0.0.0.0 so it's reachable from outside the container.
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
```

Why this is better than what you had:
- **`3.12-slim-trixie`** matches your `requires-python` — no interpreter surprise.
- **`--no-install-project` first, source later** = Docker layer caching. AutoGluon is a *heavy*
  dependency (your plan flags this); you do **not** want to reinstall it every time you tweak a
  route. 📚 [uv — Using uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/)
- **`--locked`** = fail loudly if `uv.lock` is stale, instead of silently resolving different
  versions than Person B. (Your old file used `--frozen`, which is similar; `--locked` is the
  current recommended flag — it verifies the lock is up to date.)
  📚 [uv — Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)

> 📚 Background on why containerize FastAPI this way:
> [FastAPI — Docker](https://fastapi.tiangolo.com/deployment/docker/).

### 4b. Frontend — Streamlit container (currently empty)

Streamlit isn't in your dependencies yet, and you don't want AutoGluon (heavy) in the *frontend*
image. Cleanest fix: use uv **dependency groups** to split frontend-only deps from backend deps.

**▶ YOU DO — step 1**, add a frontend group to `pyproject.toml`:

```toml
[dependency-groups]
frontend = [
    "streamlit>=1.40",
    "plotly>=5.24",
    "requests>=2.32",
]
```

Then `uv sync` locally once to update `uv.lock`.

**▶ YOU DO — step 2**, write `frontend/dockerfile`:

```dockerfile
FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

# Install ONLY the frontend group — no AutoGluon/FastAPI bloat in this image.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --only-group frontend

COPY frontend/app app

# --server.address=0.0.0.0 so it's reachable from your browser via the mapped port.
CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
```

> 📚 [uv — Dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups) ·
> [Streamlit — Deploy with Docker](https://docs.streamlit.io/deploy/tutorials/docker)

**▶ YOU DO — step 3**, write a trivial `frontend/app/streamlit_app.py` to prove the wiring
(a real UI comes in Week 1). This one pings the backend `/health` end-to-end:

```python
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("AutonomousML — skeleton")

if st.button("Check backend → Ollama health"):
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=10)
        st.json(r.json())
    except Exception as e:
        st.error(f"Could not reach backend: {e}")
```

This single button exercises the *entire* Day 0 network path:
`browser → frontend container → backend container → host.docker.internal → Ollama on your host`.
If it returns `{"status": "healthy"}`, your Docker + Ollama baseline is **done**.

---

## 5. Ollama — install, pin the model, verify GPU (Day 0 item #7)

Ollama runs **natively on Windows** (not in Docker) so it uses your RTX 4050 via CUDA.

**▶ YOU DO:**

```powershell
# 1. Install from https://ollama.com/download  (Windows installer), then in PowerShell:
ollama --version

# 2. Pull the EXACT pinned tag. Never let this drift from Person B's.
ollama pull llama3.1:8b

# 3. Confirm it's the only tag you have and note the digest (share it with Person B):
ollama list

# 4. Smoke test a generation:
ollama run llama3.1:8b "Reply with a single word: ok"
```

### Verify it's actually on the GPU (not CPU fallback)

This is a real risk on a 6 GB card. **▶ YOU DO** — in one terminal start a longer prompt, and in
another run:

```powershell
# While a prompt is generating:
ollama ps        # STATUS column should say "100% GPU" (or mostly GPU), not "100% CPU"
nvidia-smi       # you should see an "ollama" process holding GPU memory
```

- `ollama ps` showing `100% CPU` → the model didn't fit / GPU wasn't used. On 6 GB VRAM,
  `llama3.1:8b` (~4.7 GB quantized) *should* fit if you close other GPU apps (browser tabs,
  Docker Desktop's own usage). This is the asymmetry your plan warns about — keep a pre-demo
  "close GPU-hungry apps" habit.

> 📚 [Ollama — GPU docs](https://github.com/ollama/ollama/blob/main/docs/gpu.md) ·
> [Ollama — API reference](https://github.com/ollama/ollama/blob/main/docs/api.md) ·
> [Ollama — FAQ (env vars, model location)](https://github.com/ollama/ollama/blob/main/docs/faq.md)

### Improve the health check (optional but recommended)

Your `health.py` pings Ollama's root `/`, which returns the text `Ollama is running` with 200.
That proves the *server* is up but not that your *model is pulled*. A stronger check hits
`/api/tags` and confirms `OLLAMA_MODEL` is present. **▶ YOU DO** — consider evolving `health.py`:

```python
@router.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{OLLAMA_host}/api/tags")   # lists installed models
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            model_ready = OLLAMA_model in models
            return {
                "status": "healthy" if model_ready else "degraded",
                "ollama_up": True,
                "model_pinned": OLLAMA_model,
                "model_present": model_ready,
            }
        except Exception as e:
            return {"status": "unhealthy", "ollama_up": False, "error": str(e)}
```

> 📚 [Ollama API — List local models (`/api/tags`)](https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models).
> Design principle from your plan: *"fail loudly and immediately, not three agent-hops deep."*

---

## 6. Verify Docker works end-to-end (Day 0 item #8 exit)

**▶ YOU DO** — the acceptance test for today:

```powershell
# 0. One-time: confirm Docker Desktop uses the WSL2 backend
#    (Settings → General → "Use the WSL 2 based engine" ✓)
docker run --rm hello-world       # proves Docker itself works

# 1. Make sure Ollama is running on the host (ollama serve, or the desktop app)
curl http://localhost:11434       # -> "Ollama is running"

# 2. Build + start your stack
docker compose up --build

# 3. In another terminal, prove the container can reach host Ollama:
docker compose exec backend curl -s http://host.docker.internal:11434
#    -> "Ollama is running"   (this is the money shot for a Win/Mac team)

# 4. Hit the health endpoint through the backend:
curl http://localhost:8000/health

# 5. Open the frontend and click the button:
#    http://localhost:8501
```

If steps 3–5 pass, **Day 0 Docker baseline is complete.** Screenshot the `ollama ps` GPU output
and the `docker compose exec ... curl` result for your shared doc — that's your proof for Person B.

> 📚 [Docker Desktop WSL 2 backend](https://docs.docker.com/desktop/features/wsl/) ·
> [Compose — `docker compose up` / `exec`](https://docs.docker.com/reference/cli/docker/compose/)

---

## 7. AgentState schema (Day 0 item #1) — start the code stub

`backend/app/state/agent_state.py` is empty. Day 0 asks you to design this *on paper with Person B*,
but you can stub the shared shape now. LangGraph state is typically a `TypedDict`. Your plan lists
the required fields.

**▶ YOU DO** (with Person B) — sketch it; here's a starting point to react to, **not** a final answer:

```python
from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    # --- inputs ---
    dataset_path: str
    target_column: str

    # --- Data Agent ---
    dataset_profile: dict          # rows, cols, dtypes, missingness, leakage flags

    # --- Experiment Agent (AutoGluon) ---
    autogluon_leaderboard: list[dict]   # leaderboard().to_dict("records")
    winning_model: str

    # --- Critic ---
    critic_verdicts: list[dict]    # {rule, passed, detail}

    # --- Judge ---
    judge_decision: dict

    # --- control flow ---
    retry_count: int
    messages: Annotated[list, operator.add]   # append-only log (LangGraph reducer)
```

The `Annotated[list, operator.add]` bit is a **LangGraph reducer** — it tells the graph to
*append* to `messages` across nodes instead of overwriting. That's the one non-obvious LangGraph
concept worth reading before Week 1.

> 📚 [LangGraph — State & reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#state) ·
> [LangGraph — Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

---

## 8. Demo dataset (Day 0 item #6)

Pick **one** clean, small, well-understood tabular dataset for demos. Good candidates (small,
no licensing drama, obvious target column):

- **Titanic** (binary classification, `survived`) — tiny, everyone knows it.
- **California housing** (regression, `median_house_value`) — clean, sklearn-bundled.
- **Wine quality** (multiclass/regression) — small.

**▶ YOU DO** — drop the CSV in `data/` and note its target column in the shared doc. Keep it under
a few thousand rows so AutoGluon's `time_limit` (30–60 s per your plan) actually finishes a few
models. Titanic is the safest first pick.

> 📚 [AutoGluon — Tabular quick start](https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html)
> (Person B owns the AutoGluon spike, but you both need the dataset chosen today.)

---

## 9. Paper decisions template (fill with Person B)

These don't need code today, but write the answers into your shared README so they don't drift.

```markdown
### AutoGluon call params (item #2)
- time_limit: ____ s   (start 30–60 for demo safety)
- presets: medium_quality
- eval_metric: ____    (accuracy / roc_auc / rmse — depends on demo dataset)

### Critic rejection rules (item #3)
- leakage: reject if any feature correlates with target > ____
- CV-vs-holdout gap: reject if gap > ____ points
- variance: reject if top-model CV std > ____

### Retry loop bounds (item #4)
- max_retries: 2
- on retry, change: ____ (e.g. drop flagged-leaky columns / raise time_limit)

### Judge arbitration (item #5)
- inputs: autogluon_leaderboard (score) + critic_verdicts (pass/fail flags)
- rule: pick highest-scoring model that passes all critic rules; else best-passing; else fail

### Pins (never drift)
- Ollama model: llama3.1:8b   digest: ____ (from `ollama list`)
- Python: 3.12
```

---

## Day 0 Definition of Done

- [ ] `.env` whitespace fixed
- [ ] `.gitattributes` + `.dockerignore` created; `git add --renormalize .` run
- [ ] `docker-compose.yml` written and understood
- [ ] `backend/dockerfile` fixed to 3.12 + uv caching pattern
- [ ] `frontend/dockerfile` + trivial `streamlit_app.py` written
- [ ] `ollama pull llama3.1:8b` done; **GPU usage confirmed** via `ollama ps` / `nvidia-smi`
- [ ] `docker compose up --build` works; backend reaches Ollama via `host.docker.internal`
- [ ] `/health` returns healthy; frontend button works
- [ ] Demo dataset chosen and in `data/`
- [ ] AgentState stub started; paper decisions (§9) filled with Person B
- [ ] Everything above screenshotted/noted in the shared team doc

---

## Documentation quick index

| Topic | Link |
|---|---|
| Docker Compose file reference | https://docs.docker.com/reference/compose-file/services/ |
| Compose networking | https://docs.docker.com/compose/how-tos/networking/ |
| Docker Desktop → host networking | https://docs.docker.com/desktop/features/networking/ |
| Docker Desktop WSL2 backend | https://docs.docker.com/desktop/features/wsl/ |
| .dockerignore | https://docs.docker.com/reference/dockerfile/#dockerignore-file |
| gitattributes / line endings | https://git-scm.com/docs/gitattributes |
| uv in Docker | https://docs.astral.sh/uv/guides/integration/docker/ |
| uv dependency groups | https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups |
| FastAPI in Docker | https://fastapi.tiangolo.com/deployment/docker/ |
| Streamlit in Docker | https://docs.streamlit.io/deploy/tutorials/docker |
| Ollama API | https://github.com/ollama/ollama/blob/main/docs/api.md |
| Ollama GPU | https://github.com/ollama/ollama/blob/main/docs/gpu.md |
| LangGraph state/reducers | https://langchain-ai.github.io/langgraph/concepts/low_level/ |
| AutoGluon Tabular quickstart | https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html |
