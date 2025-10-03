import streamlit as st
import openai
import pandas as pd
import numpy as np
import json
import os
import spacy
from cryptography.fernet import Fernet
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from datetime import datetime

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("EchoSoul")
st.sidebar.title("EchoSoul")
st.sidebar.markdown("Adaptive personal companion — chat, call, remember.")

# -----------------------------
# Load NLP Model
# -----------------------------
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "timeline" not in st.session_state:
    st.session_state.timeline = []
if "vault_data" not in st.session_state:
    st.session_state.vault_data = {}
if "vault_key" not in st.session_state:
    st.session_state.vault_key = None

# -----------------------------
# Sidebar Controls
# -----------------------------
mode = st.sidebar.radio("Mode", ["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call", "About"])
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai.api_key = api_key if api_key else None

# -----------------------------
# Emotion Analysis (NLP)
# -----------------------------
def analyze_emotion(text):
    doc = nlp(text)
    if any(tok.lemma_ in ["happy", "glad", "great", "good"] for tok in doc):
        return "positive", 1.0
    elif any(tok.lemma_ in ["sad", "angry", "bad", "upset"] for tok in doc):
        return "negative", -1.0
    else:
        return "neutral", 0.0

# -----------------------------
# Memory Functions
# -----------------------------
def add_to_memory(role, content):
    st.session_state.messages.append({"role": role, "content": content})

def add_to_timeline(event):
    st.session_state.timeline.append({"time": str(datetime.now()), "event": event})

# -----------------------------
# Vault Functions
# -----------------------------
def set_vault_password(password):
    key = Fernet.generate_key()
    st.session_state.vault_key = Fernet(key)
    st.session_state.vault_data["password"] = password
    st.session_state.vault_data["key"] = key.decode()

def encrypt_data(data):
    f = Fernet(st.session_state.vault_data["key"].encode())
    return f.encrypt(data.encode()).decode()

def decrypt_data(token):
    f = Fernet(st.session_state.vault_data["key"].encode())
    return f.decrypt(token.encode()).decode()

# -----------------------------
# Chat Mode
# -----------------------------
if mode == "Chat":
    st.header("💬 Chat with EchoSoul")

    for msg in st.session_state.messages:
        st.markdown(f"**{msg['role']}:** {msg['content']}")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Message")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        add_to_memory("user", user_input)

        emotion, score = analyze_emotion(user_input)

        if openai.api_key:
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.messages + [{"role": "system", "content": f"User emotion: {emotion} (score={score})"}]
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"[OpenAI Error: {e}]"
        else:
            reply = f"I heard you. You seem {emotion} (score={score})."

        add_to_memory("assistant", reply)
        add_to_timeline(f"Chat: {user_input}")

# -----------------------------
# Chat History
# -----------------------------
elif mode == "Chat history":
    st.header("🗂 Chat History")
    st.write(st.session_state.messages)

# -----------------------------
# Life Timeline
# -----------------------------
elif mode == "Life timeline":
    st.header("📜 Life Timeline")
    st.table(pd.DataFrame(st.session_state.timeline))

# -----------------------------
# Vault
# -----------------------------
elif mode == "Vault":
    st.header("🔐 Memory Vault")
    if "password" not in st.session_state.vault_data:
        pwd = st.text_input("Set a password", type="password")
        if st.button("Set Password") and pwd:
            set_vault_password(pwd)
            st.success("Password set! Vault is ready.")
    else:
        pwd = st.text_input("Enter vault password", type="password")
        if st.button("Unlock Vault") and pwd == st.session_state.vault_data["password"]:
            st.success("Vault unlocked!")
            secret = st.text_area("Write sensitive memory")
            if st.button("Save Memory"):
                encrypted = encrypt_data(secret)
                st.session_state.vault_data["secret"] = encrypted
                st.success("Memory saved securely!")
            if "secret" in st.session_state.vault_data:
                st.info("Encrypted memory stored.")

# -----------------------------
# Export
# -----------------------------
elif mode == "Export":
    st.header("📤 Export Data")
    export_data = {
        "messages": st.session_state.messages,
        "timeline": st.session_state.timeline,
        "vault": st.session_state.vault_data
    }
    st.download_button("Download All Data", data=json.dumps(export_data), file_name="echosoul_export.json")

# -----------------------------
# Brain Mimic
# -----------------------------
elif mode == "Brain mimic":
    st.header("🧠 Brain Mimic")
    if openai.api_key:
        st.write("EchoSoul will now mimic your style.")
        sample_text = " ".join([m["content"] for m in st.session_state.messages if m["role"] == "user"][-5:])
        if st.button("Generate Mimic"):
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Mimic the user's tone and personality."},
                        {"role": "user", "content": sample_text}
                    ]
                )
                mimic_reply = response.choices[0].message.content
                st.success(f"Mimic: {mimic_reply}")
            except Exception as e:
                st.error(f"OpenAI Error: {e}")
    else:
        st.warning("Enter API key to enable Brain Mimic.")

# -----------------------------
# Call (Agora / WebRTC)
# -----------------------------
elif mode == "Call":
    st.header("📞 EchoSoul Call")
    st.write("Start a secure VoIP call with EchoSoul (beta).")

    webrtc_streamer(
        key="call",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "audio": True,
            "video": False
        }
    )

# -----------------------------
# About
# -----------------------------
elif mode == "About":
    st.header("ℹ️ About EchoSoul")
    st.markdown("""
    EchoSoul is your adaptive AI companion:
    - Persistent Memory  
    - Adaptive Personality  
    - Emotion Recognition (NLP-powered)  
    - Life Timeline  
    - Vault with Encryption  
    - Brain Mimic  
    - VoIP Calls  
    """)
