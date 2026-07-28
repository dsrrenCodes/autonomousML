import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    llm = ChatOpenAI(
    model=os.getenv("AGNES_MODEL","agnes-2.0-flash"),
    api_key=os.getenv("AGNES_API_KEY"),
    base_url=os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1"),
    temperature=0)
    return llm