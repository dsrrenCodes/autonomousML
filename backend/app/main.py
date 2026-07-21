from typing import Dict
from fastapi import FastAPI , File, UploadFile, Form
from starlette.concurrency import run_in_threadpool
from app.health import router
from app.agents.graph_agent import agent_graph, RECURSION_LIMIT
import io
import os
import tempfile
import pandas as pd
app = FastAPI()

app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Hello from autonomousml!"}

# autodetect target column
def detect_target(df: pd.DataFrame) -> tuple[str, str]:
    common = {"target", "label", "y", "class", "outcome"}
    for col in df.columns:
        if col.lower() in common:
            return col, "name match"
    return df.columns[-1], "last column (convention)"


@app.post('/profile')
async def profile(file: UploadFile = File(...))-> Dict :
    try:
        raw = await file.read()
        df= pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        return {"error": str(e)}
    
    target,reason = detect_target(df)
    return {
    "columns": df.columns.tolist(),
    "n_rows": len(df),
    "suggested_target": target,
    "reason": reason,
    "preview": df.head(10).to_dict(orient="records"),
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
}


@app.post('/audit')
async def audit(
    file: UploadFile = File(...),
    target_column: str = Form(...),
) -> Dict:
    """Run the full audit graph on an uploaded CSV and return the report.

    The graph reads its data from `dataset_path` on disk (and re-reads it on every
    remediation lap), so we persist the upload to a temp file for the duration of
    the run rather than holding it in memory. agent_graph.invoke is synchronous and
    slow (AutoGluon fits at 60s each + the LLM tool-loop), so it runs in a worker
    thread via run_in_threadpool — keeping FastAPI's event loop free to answer
    /health while an audit is in flight.
    """
    raw = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        tmp.write(raw)
        tmp.close()

        # The initial AgentState — the entry contract for build_graph("data").
        # Mirrors graph_agent._run_full: seed the loop bookkeeping the tools read
        # (retry_count/max_retries bound the refits) so the graph starts clean.
        initial_state = {
            "dataset_path": tmp.name,
            "target_column": target_column,
            "status": "running",
            "retry_count": 0,
            "max_retries": 2,
            "remediation_history": [],
            "messages": [],
        }

        result = await run_in_threadpool(
            agent_graph.invoke, initial_state, {"recursion_limit": RECURSION_LIMIT}
        )
        return result["report"]
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.unlink(tmp.name)

    
