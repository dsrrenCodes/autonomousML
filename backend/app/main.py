from typing import Dict
from fastapi import FastAPI , File, UploadFile
from app.health import router
import io
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

    
