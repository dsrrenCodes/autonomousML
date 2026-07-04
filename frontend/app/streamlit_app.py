import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("Auditor")

if st.button("Test backend health"):
    try:
        res = requests.get(f"{backend_url}/health", timeout=5)
        st.write(res.status_code, res.json())
    except Exception as e:
        st.error(f"Health check failed: {e}")

# --- session state init ---
for key, default in {
    "profile": None,
    "confirmed": False,
    "target_column": None,
    "last_file_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

file = st.file_uploader("Upload a CSV file", type=["csv"])

# Only re-profile when a genuinely new file is uploaded
if file is not None and file.file_id != st.session_state.last_file_id:
    try:
        resp = requests.post(f"{backend_url}/profile", files={"file": file}, timeout=30)
        resp.raise_for_status()
        st.session_state.profile = resp.json()
        st.session_state.last_file_id = file.file_id
        st.session_state.confirmed = False  # new file -> reset confirmation
    except Exception as e:
        st.error(f"Profiling failed: {e}")
        st.session_state.profile = None

profile = st.session_state.profile

if profile and not st.session_state.confirmed:
    st.dataframe(profile["preview"])
    st.info(f"Suggested target: {profile['suggested_target']}\nReason: {profile['reason']}")

    choice = st.selectbox(
        "Check Target column",
        options=profile["columns"],
        index=profile["columns"].index(profile["suggested_target"]),
    )

    if st.button("Confirm target"):
        st.session_state.target_column = choice
        st.session_state.confirmed = True
        st.rerun()

elif profile and st.session_state.confirmed:
    st.success(f"Target column confirmed: {st.session_state.target_column}")
    if st.button("Change target"):
        st.session_state.confirmed = False
        st.rerun()