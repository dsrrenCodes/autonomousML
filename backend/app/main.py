from fastapi import FastAPI

from app.health import router
app = FastAPI()

app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Hello from autonomousml!"}
