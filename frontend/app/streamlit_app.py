
import streamlit as st
import requests
from dotenv import load_dotenv
import os 
load_dotenv()

backend_url= os.getenv("BACKEND_URL","http://localhost:8000")
st.title('ongawd awd')

if st.button("test ollama health backend"):
    try:
        res = requests.get(f'{backend_url}/health')
        st.write(f"Status: {res.status_code}")
        st.write(res.json())
    except Exception as e:
        st.error(e)
