import httpx
import os
from fastapi import APIRouter
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

OLLAMA_model= os.getenv('OLLAMA_MODEL')
OLLAMA_host= os.getenv('OLLAMA_HOST')
@router.get("/health")
async def health():
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f'{OLLAMA_host}')
            if r.status_code == 200:
                return {'status': 'healthy'}
            return {'status': 'unhealthy', 'code': r.status_code}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
