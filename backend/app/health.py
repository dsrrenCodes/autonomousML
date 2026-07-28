import httpx
import os
from fastapi import APIRouter
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# The audit graph's Judge runs on Agnes (an OpenAI-compatible endpoint), not
# Ollama — so health must probe Agnes, otherwise a red check just means "Ollama
# isn't running", a service the pipeline never calls. We hit the OpenAI-standard
# GET /models: cheap, no token cost, and it validates BOTH reachability and that
# AGNES_API_KEY is accepted (a bad key returns 401, not 200).
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1")
AGNES_API_KEY = os.getenv("AGNES_API_KEY")


@router.get("/health")
async def health():
    headers = {"Authorization": f"Bearer {AGNES_API_KEY}"}
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{AGNES_BASE_URL}/models", headers=headers)
            if r.status_code == 200:
                return {"status": "healthy", "llm": "agnes"}
            return {"status": "unhealthy", "code": r.status_code}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
