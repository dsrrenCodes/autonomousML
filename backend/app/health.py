import httpx
import os
from fastapi import APIRouter
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
#docker run --rm -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 -e OLLAMA_MODEL=llama3.1:8b backend

OLLAMA_model= os.getenv('OLLAMA_MODEL','llama3.1:8b')
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
