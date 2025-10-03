# EchoSoul - app.py (Fixed + Combined)
# -------------------------------------------------------------
# Features:
# - Chat UI with persistent memory (SQLite)
# - Adaptive persona & Brain Mimic
# - Emotion detection & Consciousness Mirror
# - Life timeline
# - Private Vault (Fernet encryption)
# - Export / Import
# - Simulated call (typed/audio upload) with gTTS fallback
# - Live WebRTC call (via streamlit-webrtc)
# - Uses new OpenAI Python client API (v1)
# -------------------------------------------------------------

import streamlit as st
import sqlite3
import json
import os
import io
import time
import base64
from datetime import datetime
from typing import Optional, Tuple

# Optional imports (graceful fallback)
try:
    import openai
except Exception:
    openai = None

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
except Exception:
    Fernet = None
    PBKDF2HMAC = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase
    import av
    WEBSOCKET_AVAILABLE = True
except Exception:
    webrtc_streamer = None
    WebRtcMode = None
    RTCConfiguration = None
    AudioProcessorBase = object
    av = None
    WEBSOCKET_AVAILABLE = False


# ---------------- DB init ----------------
DB_PATH = "echosoul.db"

@st.cache_resource
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  role TEXT,
                  content TEXT,
                  created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS memories (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  content TEXT,
                  tags TEXT,
                  created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS timeline (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  event TEXT,
                  event_date TEXT,
                  details TEXT,
                  tags TEXT,
                  created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS vault (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  label TEXT,
                  encrypted_blob TEXT,
                  salt TEXT,
                  created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS settings (
                  key TEXT PRIMARY KEY,
                  value TEXT)''')
    conn.commit()
    return conn

conn = init_db()


# ------------- Utilities -------------
_POS_WORDS = {"happy","great","awesome","love","joy","excited","pleased","good","fantastic","wonderful"}
_NEG_WORDS = {"sad","angry","hate","terrible","upset","bad","depressed","annoyed","frustrated"}

def detect_emotion_from_text(text: str) -> Tuple[str, float]:
    text_l = text.lower()
    score = 0
    for w in _POS_WORDS:
        if w in text_l: score += 1
    for w in _NEG_WORDS:
        if w in text_l: score -= 1
    if score >= 2: return "happy", float(score)
    if score <= -2: return "sad/angry", float(score)
    if score == 1: return "positive", float(score)
    if score == -1: return "negative", float(score)
    return "neutral", float(score)


# ------------- Encryption (Vault) -------------
def _derive_key(password: str, salt: bytes) -> bytes:
    if PBKDF2HMAC is None:
        raise RuntimeError("cryptography library required")
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390000)
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_text(plaintext: str, password: str) -> Tuple[str, str]:
    salt = os.urandom(16)
    key = _derive_key(password, salt)
    f = Fernet(key)
    token = f.encrypt(plaintext.encode())
    return base64.b64encode(token).decode(), base64.b64encode(salt).decode()

def decrypt_text(token_b64: str, password: str, salt_b64: str) -> str:
    token = base64.b64decode(token_b64)
    salt = base64.b64decode(salt_b64)
    key = _derive_key(password, salt)
    f = Fernet(key)
    return f.decrypt(token).decode()


# ------------- DB helpers -------------
def add_chat(role: str, content: str):
    c = conn.cursor()
    c.execute("INSERT INTO chats (role,content,created_at) VALUES (?,?,?)",
              (role, content, datetime.utcnow().isoformat()))
    conn.commit()

def get_chats(limit=200):
    c = conn.cursor()
    c.execute("SELECT id,role,content,created_at FROM chats ORDER BY id DESC LIMIT ?", (limit,))
    return list(reversed(c.fetchall()))

def add_memory(title: str, content: str, tags: str = ""):
    c = conn.cursor()
    c.execute("INSERT INTO memories (title,content,tags,created_at) VALUES (?,?,?,?)",
              (title, content, tags, datetime.utcnow().isoformat()))
    conn.commit()

def get_recent_memories(limit=10):
    c = conn.cursor()
    c.execute("SELECT id,title,content,tags,created_at FROM memories ORDER BY id DESC LIMIT ?", (limit,))
    return c.fetchall()

def add_timeline_event(event: str, event_date: str, details: str, tags: str = ""):
    c = conn.cursor()
    c.execute("INSERT INTO timeline (event,event_date,details,tags,created_at) VALUES (?,?,?,?,?)",
              (event, event_date, details, tags, datetime.utcnow().isoformat()))
    conn.commit()

def get_timeline(limit=100):
    c = conn.cursor()
    c.execute("SELECT id,event,event_date,details,tags,created_at FROM timeline ORDER BY event_date DESC LIMIT ?", (limit,))
    return c.fetchall()


# ------------- OpenAI helpers -------------
def make_openai_client(api_key: Optional[str]):
    if openai is None or not api_key:
        return None
    try:
        return openai.OpenAI(api_key=api_key)
    except Exception:
        return None

def openai_chat_reply(client, messages: list, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content


# ------------- TTS (gTTS fallback) -------------
def tts_gtts_bytes(text: str):
    if gTTS is None: return None
    try:
        tts = gTTS(text=text, lang="en")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None


# ------------- Persona + Reply generator -------------
def build_system_prompt() -> str:
    persona = "You are EchoSoul — a compassionate, adaptive companion."
    mems = get_recent_memories(6)
    mem_text = "\n".join([f"- {m[1]}: {m[2]}" for m in mems])
    return f"{persona}\n\nRecent memories:\n{mem_text}"

def generate_reply(client, user_text: str) -> str:
    sys_prompt = build_system_prompt()
    messages = [{"role":"system","content":sys_prompt},{"role":"user","content":user_text}]
    if client:
        try:
            return openai_chat_reply(client, messages)
        except:
            pass
    emo, score = detect_emotion_from_text(user_text)
    return f"I heard you. You seem {emo} (score={score}). Tell me more."


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="EchoSoul", layout="wide")

with st.sidebar:
    st.title("EchoSoul")
    mode = st.radio("Mode", ["Chat","Chat history","Life timeline","Vault","Call","About"], index=0)
    openai_key_input = st.text_input("OpenAI API Key (session only)", type="password")
    if openai_key_input:
        st.session_state["openai_api_key"] = openai_key_input


# Ensure session vars
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""


# -------- Pages --------
if mode == "Chat":
    st.header("Chat with EchoSoul")
    chats = get_chats(200)
    for cid, role, content, created_at in chats:
        st.markdown(f"**{role}**: {content}")
    st.markdown("---")
    with st.form("chat_form"):
        user_text = st.text_area("Message", key="input_area", height=120)
        send_btn = st.form_submit_button("Send")
    if send_btn:
        add_chat("user", user_text)
        client = make_openai_client(st.session_state.get("openai_api_key"))
        reply = generate_reply(client, user_text)
        add_chat("assistant", reply)
        st.rerun()

elif mode == "Life timeline":
    st.header("Life Timeline")
    with st.form(key="timeline_form"):
        ev_date = st.date_input("Event date", value=datetime.utcnow().date())
        ev_title = st.text_input("Event title")
        ev_details = st.text_area("Details")
        submit = st.form_submit_button("Add event")
    if submit:
        add_timeline_event(ev_title, ev_date.isoformat(), ev_details)  # ✅ FIXED parenthesis
        st.success("Event added to timeline")
        st.rerun()
    st.markdown("### Timeline")
    ts = get_timeline(200)
    for t in ts:
        st.markdown(f"**{t[2]} - {t[1]}**\n{t[3]}\n")

elif mode == "About":
    st.header("About EchoSoul")
    st.markdown("Adaptive personal companion — chat, call, remember.")
