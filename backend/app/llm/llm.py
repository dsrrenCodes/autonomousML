import os
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434"),
    temperature=0)
    return llm