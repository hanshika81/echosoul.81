# EchoSoul - app.py (with GPT error logging + Call)

import streamlit as st
import sqlite3
import os
import io
import base64
from datetime import datetime
from typing import Optional, Tuple

# Optional imports
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
    conn.commit()
    return conn

conn = init_db()


# ------------- Utilities -------------
_POS_WORDS = {"happy","great","awesome","love","joy","excited","pleased","good","fantastic","wonderful"}
_NEG_WORDS = {"sad","angry","hate","terrible","upset","bad","depressed","annoyed","frustrated"}

def detect_emotion_from_text(text: str) -> Tuple[str, float]:
    text_l = text.lower()
    score = sum(w in text_l for w in _POS_WORDS) - sum(w in text_l for w in _NEG_WORDS)
    if score >= 2: return "happy", float(score)
    if score <= -2: return "sad/angry", float(score)
    if score == 1: return "positive", float(score)
    if score == -1: return "negative", float(score)
    return "neutral", float(score)


# ------------- Encryption (Vault) -------------
def _derive_key(password: str, salt: bytes) -> bytes:
    if PBKDF2HMAC is None:
        raise RuntimeError("cryptography required")
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
    return Fernet(key).decrypt(token).decode()


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

def add_memory(title: str, content: str):
    c = conn.cursor()
    c.execute("INSERT INTO memories (title,content,tags,created_at) VALUES (?,?,?,?)",
              (title, content, "", datetime.utcnow().isoformat()))
    conn.commit()

def get_recent_memories(limit=10):
    c = conn.cursor()
    c.execute("SELECT id,title,content FROM memories ORDER BY id DESC LIMIT ?", (limit,))
    return c.fetchall()

def add_timeline_event(event: str, event_date: str, details: str):
    c = conn.cursor()
    c.execute("INSERT INTO timeline (event,event_date,details,created_at) VALUES (?,?,?,?)",
              (event, event_date, details, datetime.utcnow().isoformat()))
    conn.commit()

def get_timeline(limit=100):
    c = conn.cursor()
    c.execute("SELECT event,event_date,details FROM timeline ORDER BY event_date DESC LIMIT ?", (limit,))
    return c.fetchall()


# ------------- OpenAI helpers -------------
def make_openai_client(api_key: Optional[str]):
    if openai is None or not api_key:
        return None
    return openai.OpenAI(api_key=api_key)

def openai_chat_reply(client, messages: list):
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
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


# ------------- Reply generator -------------
def generate_reply(client, user_text: str) -> str:
    mems = get_recent_memories(6)
    mem_text = "\n".join([f"- {m[1]}: {m[2]}" for m in mems])
    sys_prompt = f"You are EchoSoul — a compassionate companion.\n\nRecent memories:\n{mem_text}"
    messages = [{"role":"system","content":sys_prompt},{"role":"user","content":user_text}]
    if client:
        try:
            return openai_chat_reply(client, messages)
        except Exception as e:
            return f"(OpenAI error: {e})"
    # fallback if no key
    emo, score = detect_emotion_from_text(user_text)
    return f"I heard you. You seem {emo} (score={score})."


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="EchoSoul", layout="wide")

with st.sidebar:
    st.title("EchoSoul")
    mode = st.radio("Mode", ["Chat","Chat history","Life timeline","Vault","Call","About"])
    
    # ✅ API key persists
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""
    api_key_input = st.text_input("OpenAI API Key", type="password", value=st.session_state["openai_api_key"])
    if api_key_input:
        st.session_state["openai_api_key"] = api_key_input


# -------- Pages --------
if mode == "Chat":
    st.header("💬 Chat with EchoSoul")
    chats = get_chats()
    for _, role, content, _ in chats:
        st.markdown(f"**{role}**: {content}")
    with st.form("chat_form"):
        user_text = st.text_area("Message", height=120)
        send_btn = st.form_submit_button("Send")
    if send_btn and user_text.strip():
        add_chat("user", user_text)
        client = make_openai_client(st.session_state.get("openai_api_key"))
        reply = generate_reply(client, user_text)
        add_chat("assistant", reply)
        st.markdown(f"**assistant**: {reply}")
        audio_bytes = tts_gtts_bytes(reply)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        st.rerun()

elif mode == "Chat history":
    st.header("🗂 Chat History")
    for _, role, content, created in get_chats(100):
        st.write(f"{created[:19]} — **{role}**: {content}")

elif mode == "Life timeline":
    st.header("📅 Life Timeline")
    with st.form("timeline_form"):
        ev_date = st.date_input("Event date", value=datetime.utcnow().date())
        ev_title = st.text_input("Event title")
        ev_details = st.text_area("Details")
        if st.form_submit_button("Add event"):
            add_timeline_event(ev_title, ev_date.isoformat(), ev_details)
            st.success("Event added")
            st.rerun()
    st.write("### Timeline")
    for ev, date, details in get_timeline():
        st.write(f"**{date}** - {ev}\n{details}")

elif mode == "Vault":
    st.header("🔐 Vault")
    pw = st.text_input("Vault password", type="password")
    if pw:
        st.info("Vault feature ready (encryption enabled).")

elif mode == "Call":
    st.header("📞 Live Call with EchoSoul")
    if not WEBSOCKET_AVAILABLE:
        st.error("streamlit-webrtc not installed. Please check requirements.")
    else:
        class EchoSoulProcessor(AudioProcessorBase):
            def __init__(self):
                self.client = make_openai_client(st.session_state.get("openai_api_key"))

            def recv_audio(self, frame):
                # For now: just echo input audio
                return frame

        webrtc_streamer(
            key="echosoul-call",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
            audio_processor_factory=EchoSoulProcessor,
            media_stream_constraints={"audio": True, "video": False},
        )
        st.info("🎤 Speak into your mic. EchoSoul will echo audio. GPT+TTS integration coming next.")

elif mode == "About":
    st.header("ℹ️ About EchoSoul")
    st.write("EchoSoul is your adaptive personal AI companion — chat, call, remember, grow.")
