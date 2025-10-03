import streamlit as st
import sqlite3
import os
import datetime
import json
import pandas as pd
from cryptography.fernet import Fernet
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
from gtts import gTTS
import tempfile
import openai

# Initialize OpenAI client dynamically with user API key
def get_openai_client():
    if "api_key" not in st.session_state or not st.session_state["api_key"]:
        return None
    return openai.OpenAI(api_key=st.session_state["api_key"])

# DB setup
DB_FILE = "echosoul.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY, content TEXT, timestamp TEXT)""")
c.execute("""CREATE TABLE IF NOT EXISTS timeline (id INTEGER PRIMARY KEY, title TEXT, date TEXT, details TEXT)""")
c.execute("""CREATE TABLE IF NOT EXISTS vault (id INTEGER PRIMARY KEY, data BLOB)""")
conn.commit()

# Encryption key storage
if "fernet_key" not in st.session_state:
    st.session_state.fernet_key = Fernet.generate_key()
fernet = Fernet(st.session_state.fernet_key)

# Helper functions
def save_memory(content):
    c.execute("INSERT INTO memory (content, timestamp) VALUES (?, ?)", (content, str(datetime.datetime.now())))
    conn.commit()

def get_memories():
    c.execute("SELECT * FROM memory ORDER BY id DESC")
    return c.fetchall()

def add_timeline_event(title, date, details):
    c.execute("INSERT INTO timeline (title, date, details) VALUES (?, ?, ?)", (title, date, details))
    conn.commit()

def get_timeline():
    c.execute("SELECT * FROM timeline ORDER BY date DESC")
    return c.fetchall()

def save_to_vault(data, password):
    encrypted = Fernet(password.ljust(32)[:32].encode()).encrypt(data.encode())
    c.execute("INSERT INTO vault (data) VALUES (?)", (encrypted,))
    conn.commit()

def get_vault(password):
    c.execute("SELECT data FROM vault")
    records = c.fetchall()
    decrypted = []
    for (blob,) in records:
        try:
            val = Fernet(password.ljust(32)[:32].encode()).decrypt(blob).decode()
            decrypted.append(val)
        except Exception:
            continue
    return decrypted

# Emotion detection (very simple)
def detect_emotion(text):
    if any(w in text.lower() for w in ["sad", "depressed", "unhappy"]):
        return "sad", -0.5
    elif any(w in text.lower() for w in ["happy", "glad", "excited"]):
        return "happy", 0.8
    return "neutral", 0.0

# Streamlit UI
st.set_page_config(page_title="EchoSoul", layout="wide")

st.sidebar.title("EchoSoul")
st.sidebar.caption("Adaptive personal companion — chat, call, remember.")

mode = st.sidebar.radio("Mode", ["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call", "About"])

st.sidebar.subheader("Settings")
st.session_state["api_key"] = st.sidebar.text_input("OpenAI API Key (session only)", type="password")
voice_choice = st.sidebar.selectbox("Voice (for TTS)", ["Default", "alloy", "verse", "sage"])
brain_mimic = st.sidebar.checkbox("Enable Brain Mimic")
mimic_strength = st.sidebar.slider("Mimic strength", 0, 100, 30)
if st.sidebar.button("Clear chat history"):
    c.execute("DELETE FROM memory")
    conn.commit()
    st.sidebar.success("Chat history cleared!")

# Chat mode
if mode == "Chat":
    st.header("💬 Chat with EchoSoul")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat:
        st.chat_message(role).write(msg)

    user_input = st.chat_input("Say something to EchoSoul...")
    if user_input:
        st.session_state.chat.append(("user", user_input))
        save_memory(user_input)

        client = get_openai_client()
        if client:
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "system", "content": "You are EchoSoul."}] +
                             [{"role": role, "content": msg} for role, msg in st.session_state.chat]
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"(Error from OpenAI: {e})"
        else:
            reply = "No API key provided."

        st.session_state.chat.append(("assistant", reply))
        st.chat_message("assistant").write(reply)

        # Emotion detection
        emotion, score = detect_emotion(user_input)
        st.success(f"Emotion detected: {emotion} (score {score})")

# Chat history
elif mode == "Chat history":
    st.header("🗂 Chat History")
    memories = get_memories()
    for _, content, timestamp in memories:
        st.write(f"{timestamp}: {content}")

# Timeline
elif mode == "Life timeline":
    st.header("📅 Life Timeline")
    ev_title = st.text_input("Event title")
    ev_date = st.date_input("Event date")
    ev_details = st.text_area("Details")
    if st.button("Save Event"):
        add_timeline_event(ev_title, ev_date.isoformat(), ev_details)
        st.success("Event saved to timeline!")
        st.rerun()

    timeline = get_timeline()
    for _, title, date, details in timeline:
        st.write(f"**{date}** — {title}: {details}")

# Vault
elif mode == "Vault":
    st.header("🔐 Vault")
    vault_pw = st.text_input("Enter vault password", type="password")
    if vault_pw:
        data = st.text_area("Store something secret")
        if st.button("Save to vault"):
            save_to_vault(data, vault_pw)
            st.success("Saved to vault!")

        if st.button("View vault"):
            secrets = get_vault(vault_pw)
            for s in secrets:
                st.write(s)

# Export
elif mode == "Export":
    st.header("📤 Export Soul Resonance")
    memories = get_memories()
    timeline = get_timeline()
    data = {"memories": memories, "timeline": timeline}
    st.download_button("Download Export", json.dumps(data, indent=2), file_name="echosoul_export.json")

# Brain mimic
elif mode == "Brain mimic":
    st.header("🧠 Brain Mimic")
    client = get_openai_client()
    if client:
        prompt = st.text_area("Ask as if you were yourself...")
        if st.button("Mimic Response"):
            chat_history = get_memories()
            context = "\n".join([c for _, c, _ in chat_history])
            try:
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": f"You are imitating the user's personality. Context:\n{context}"},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(e)

# Call (VoIP via WebRTC)
elif mode == "Call":
    st.header("📞 Live Call with EchoSoul")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class EchoSoulCallProcessor:
        def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
            client = get_openai_client()
            if not client:
                return frame
            # EchoSoul "listens" and replies - simplified demo (echo back user audio)
            return frame

    webrtc_streamer(
        key="echosoul-call",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        audio_processor_factory=EchoSoulCallProcessor,
        media_stream_constraints={"video": False, "audio": True},
    )

# About
elif mode == "About":
    st.header("ℹ️ About EchoSoul")
    st.write("EchoSoul is your adaptive personal AI companion — chat, call, remember, and grow with you.")
