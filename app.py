# EchoSoul - app.py
# -------------------------------------------
# Full-featured Streamlit app implementing the EchoSoul described by the user.
# IMPORTANT: Before deploying to Streamlit Cloud, add the required packages to requirements.txt
# Example requirements.txt (put in the same repo):
# streamlit
# openai
# cryptography
# gtts
# pydub
# numpy
# pandas
# streamlit-webrtc==0.64.0   # optional (for live browser microphone support)
# soundfile
# av
# -------------------------------
# Environment variables / Streamlit secrets recommended:
# - OPENAI_API_KEY  (or enter API PIN in the app UI)
# - ELEVENLABS_API_KEY (optional for higher quality TTS)
# Notes:
# - This app attempts to provide both full features and graceful fallbacks.
# - Live telephony (phone calls) or guaranteed hardware access can't be provided purely in Streamlit Cloud.
#   We provide a "Live WebRTC call" interface (if streamlit-webrtc is installed and supported) and a
#   simulated "Call" experience that uses typed or uploaded audio plus TTS playback.
# - Vault encryption uses the `cryptography` library and stores encrypted blobs in a SQLite DB.
# - The app uses OpenAI for chat generation and (optionally) for transcription. If you don't provide
#   an OpenAI key, the app will use local fallbacks.
# -------------------------------

import streamlit as st
import sqlite3
import json
import os
import time
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

# third-party libraries that should be available in requirements.txt
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

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

# Optional: WebRTC support for "live" calls (will gracefully disable if package not installed)
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    WEBSOCKET_AVAILABLE = True
except Exception:
    WEBSOCKET_AVAILABLE = False

# -------------- Helper utilities --------------

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

# Small positive / negative word lists for simple sentiment/emotion heuristics
_POS_WORDS = set(["happy", "great", "awesome", "love", "joy", "excited", "pleased", "good", "fantastic", "wonderful"]) 
_NEG_WORDS = set(["sad", "angry", "hate", "terrible", "upset", "bad", "depressed", "annoyed", "frustrated"]) 

# ---------- Encryption utilities (Vault) ----------

def _derive_key(password: str, salt: bytes) -> bytes:
    if PBKDF2HMAC is None:
        raise RuntimeError("cryptography library is required for vault encryption")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


def encrypt_text(plaintext: str, password: str) -> Tuple[str, str]:
    """Return (token_b64, salt_b64)"""
    if Fernet is None:
        raise RuntimeError("cryptography not installed")
    salt = os.urandom(16)
    key = _derive_key(password, salt)
    f = Fernet(key)
    token = f.encrypt(plaintext.encode())
    return base64.b64encode(token).decode(), base64.b64encode(salt).decode()


def decrypt_text(token_b64: str, password: str, salt_b64: str) -> str:
    if Fernet is None:
        raise RuntimeError("cryptography not installed")
    token = base64.b64decode(token_b64)
    salt = base64.b64decode(salt_b64)
    key = _derive_key(password, salt)
    f = Fernet(key)
    return f.decrypt(token).decode()

# ---------- Database helpers ----------

def add_chat(role: str, content: str):
    c = conn.cursor()
    c.execute("INSERT INTO chats (role,content,created_at) VALUES (?,?,?)",
              (role, content, datetime.utcnow().isoformat()))
    conn.commit()


def get_chats(limit=200) -> List[Tuple[int,str,str,str]]:
    c = conn.cursor()
    c.execute("SELECT id,role,content,created_at FROM chats ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    return list(reversed(rows))


def add_memory(title: str, content: str, tags: Optional[str] = ""):
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


def add_vault_item(label: str, encrypted_blob: str, salt: str):
    c = conn.cursor()
    c.execute("INSERT INTO vault (label,encrypted_blob,salt,created_at) VALUES (?,?,?,?)",
              (label, encrypted_blob, salt, datetime.utcnow().isoformat()))
    conn.commit()


def get_vault_items():
    c = conn.cursor()
    c.execute("SELECT id,label,encrypted_blob,salt,created_at FROM vault ORDER BY id DESC")
    return c.fetchall()


def save_setting(key: str, value: str):
    c = conn.cursor()
    c.execute("REPLACE INTO settings (key,value) VALUES (?,?)", (key, value))
    conn.commit()


def load_setting(key: str) -> Optional[str]:
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    r = c.fetchone()
    return r[0] if r else None

# ---------- Simple emotion detection heuristics ----------

def detect_emotion_from_text(text: str) -> Tuple[str, float]:
    # very simple lexicon-based scoring
    text_l = text.lower()
    score = 0
    for w in _POS_WORDS:
        if w in text_l:
            score += 1
    for w in _NEG_WORDS:
        if w in text_l:
            score -= 1
    # map to label
    if score >= 2:
        return "happy", float(score)
    elif score <= -2:
        return "sad/angry", float(score)
    elif score == 1:
        return "positive", float(score)
    elif score == -1:
        return "negative", float(score)
    else:
        return "neutral", float(score)

# ---------- OpenAI integration (chat + optional whisper) ----------

DEFAULT_MODEL = "gpt-3.5-turbo"

def configure_openai_from_ui(api_key: Optional[str]):
    if api_key:
        if openai is None:
            st.warning("openai python package not installed; model calls will fail until you add openai to requirements.txt")
            return False
        openai.api_key = api_key
        return True
    else:
        return False


def openai_chat(messages: list, model: str = DEFAULT_MODEL, temperature: float = 0.7) -> Optional[str]:
    if openai is None:
        return None
    try:
        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
        return resp['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"OpenAI Chat error: {e}")
        return None


def transcribe_audio_with_openai(audio_bytes: bytes) -> Optional[str]:
    if openai is None:
        return None
    try:
        # the OpenAI Python library might need the file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "upload.wav"
        resp = openai.Audio.transcribe("whisper-1", audio_file)
        return resp['text']
    except Exception as e:
        st.error(f"OpenAI transcription error: {e}")
        return None

# ---------- Text-to-speech fallback (gTTS) ----------
import io

def tts_gtts_generate(text: str, lang: str = 'en') -> Optional[bytes]:
    if gTTS is None:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read()
    except Exception as e:
        st.error(f"gTTS error: {e}")
        return None

# ---------- Response generation and personality logic ----------

def build_system_prompt(mimic: bool = False, mimic_examples: list = None) -> str:
    persona = load_setting('persona') or (
        "You are EchoSoul — a compassionate, adaptive, and private life companion. You remember user details and adapt tone over time."
    )
    mems = get_recent_memories(6)
    mem_text = "\n".join([f"- {m[1]}: {m[2]}" for m in mems])
    prompt = f"{persona}\n\nRecent memories:\n{mem_text}\n\nRules:\n- Protect privacy and the Vault.\n- If asked to be the user's past/future self, respond in that voice.\n- When brain-mimic is enabled, mimic user's style based on examples.\n"
    if mimic and mimic_examples:
        prompt += "\nMimic examples for style:\n"
        for ex in mimic_examples[-6:]:
            prompt += f"- User wrote: {ex}\n"
        prompt += "\nMimic tone: Use the user's vocabulary and phrasing where appropriate.\n"
    return prompt


def generate_reply(user_text: str, mimic: bool=False, mimic_examples: list=None, temperature: float=0.7, model: str = DEFAULT_MODEL) -> str:
    system_prompt = build_system_prompt(mimic=mimic, mimic_examples=mimic_examples)
    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content": user_text}
    ]
    # Try openai
    reply = openai_chat(messages, model=model, temperature=temperature)
    if reply:
        return reply.strip()
    # Fallback simple reply if no OpenAI key
    emotion_label, score = detect_emotion_from_text(user_text)
    fallback = f"I heard you. You feel {emotion_label} (score={score}). Tell me more — I'm listening." 
    return fallback

# ---------- UI helpers ----------

def format_chat_message(role: str, content: str):
    if role == 'user':
        return f"**You**: {content}"
    else:
        return f"**EchoSoul**: {content}"

# ---------- Life Path Simulation (simple) ----------

def simulate_scenarios(context: str, options: List[str]) -> str:
    # Build a simulation prompt summarizing options and asking model to play out scenarios
    prompt = f"You are a life-path simulator. Context: {context}\nOptions:\n"
    for i,opt in enumerate(options,1):
        prompt += f"{i}. {opt}\n"
    prompt += "\nFor each option, provide: (a) a short simulated outcome, (b) likely pros/cons, (c) recommended choice and why. Keep it concise."
    # call model
    reply = generate_reply(prompt, temperature=0.8)
    return reply

# ---------- Soul Resonance Network & Export/Import ----------

def export_echo_json() -> dict:
    c = conn.cursor()
    c.execute("SELECT id,role,content,created_at FROM chats ORDER BY id ASC")
    chats = c.fetchall()
    c.execute("SELECT id,title,content,tags,created_at FROM memories ORDER BY id ASC")
    memories = c.fetchall()
    c.execute("SELECT id,event,event_date,details,tags,created_at FROM timeline ORDER BY id ASC")
    timeline = c.fetchall()
    payload = {
        "exported_at": datetime.utcnow().isoformat(),
        "chats": chats,
        "memories": memories,
        "timeline": timeline,
        "settings": {k: load_setting(k) for k in ["persona"] if load_setting(k)}
    }
    return payload

def import_echo_json(data: dict, merge: bool=True) -> Tuple[int,int,int]:
    # Accept a dict matching export_echo_json
    chats = data.get('chats', [])
    memories = data.get('memories', [])
    timeline = data.get('timeline', [])
    for ch in chats:
        add_chat(ch[1], ch[2])
    for m in memories:
        add_memory(m[1], m[2], tags=m[3])
    for t in timeline:
        add_timeline_event(t[1], t[2], t[3], t[4])
    return len(chats), len(memories), len(timeline)

# ---------- Application UI ----------

st.set_page_config(page_title="EchoSoul — Personal AI Companion", layout="wide")

# sidebar
with st.sidebar:
    st.title("EchoSoul")
    st.markdown("A private, adaptive life companion — remember, reflect, and plan.")
    mode = st.radio("Mode", ["Chat","Chat history","Life timeline","Vault","Export","Brain mimic","Call","About"], index=0)
    st.markdown("---")
    st.subheader("Settings")
    api_key_input = st.text_input("OpenAI API Key / API PIN (store only in session)", type="password")
    if api_key_input:
        ok = configure_openai_from_ui(api_key_input)
        if ok:
            st.success("OpenAI key configured for this session.")

    voice_choice = st.selectbox("Voice (for TTS)", ["Default","Warm Female","Calm Male","Deep Narrator","Adaptive (uses persona)"])
    mimic_enabled = st.checkbox("Enable Brain Mimic (mimic your style)", value=False)
    mimic_strength = st.slider("Mimic strength", min_value=0, max_value=100, value=30)
    st.markdown("---")
    st.write("Quick actions")
    if st.button("Add test memory"):
        add_memory("Test memory", "You wrote a test memory at %s" % datetime.utcnow().isoformat())
        st.success("Test memory saved")
    if st.button("Clear local chat history"):
        c = conn.cursor()
        c.execute("DELETE FROM chats")
        conn.commit()
        st.experimental_rerun()

# ensure session state
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''
if 'call_active' not in st.session_state:
    st.session_state['call_active'] = False
if 'call_start' not in st.session_state:
    st.session_state['call_start'] = None

# Render pages

if mode == 'Chat':
    left, right = st.columns([3,1])
    with left:
        st.header("Chat with EchoSoul")
        chat_box = st.container()
        # show chats
        chats = get_chats(200)
        for cid, role, content, created_at in chats:
            if role == 'user':
                st.markdown(f"<div style='background:#e6f7ff;padding:8px;border-radius:8px'><strong>You</strong>: {content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#fff7e6;padding:8px;border-radius:8px'><strong>EchoSoul</strong>: {content}</div>", unsafe_allow_html=True)
        st.markdown("---")
        with st.form(key='chat_form', clear_on_submit=False):
            user_text = st.text_area("Message", value=st.session_state['user_input'], key='input_area', placeholder="Type a message or ask EchoSoul to simulate your future self...", height=120)
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                send_btn = st.form_submit_button("Send")
            with col2:
                save_mem_btn = st.form_submit_button("Save to memory")
            with col3:
                reflect_btn = st.form_submit_button("Reflect (Consciousness Mirror)")

        if send_btn:
            st.session_state['user_input'] = ''
            add_chat('user', user_text)
            # decide mimic examples
            mimic_examples = []
            if mimic_enabled:
                # fetch last user messages
                recent = [c[2] for c in get_chats(50) if c[1]=='user']
                mimic_examples = recent[-12:]
            reply = generate_reply(user_text, mimic=mimic_enabled, mimic_examples=mimic_examples, temperature=0.8)
            add_chat('assistant', reply)
            # emotion detection
            emo_label, emo_score = detect_emotion_from_text(user_text)
            st.success(f"Emotion detected: {emo_label} (score {emo_score})")
            # Auto-save some memories heuristically
            if len(user_text) > 200 or "important" in user_text.lower():
                add_memory(title=f"Auto: {user_text[:30]}...", content=user_text)
            # TTS playback
            tts_bytes = tts_gtts_generate(reply) if gTTS else None
            if tts_bytes:
                st.audio(tts_bytes, format='audio/mp3')
            st.experimental_rerun()

        if save_mem_btn:
            add_memory(title=f"Manual: {user_text[:40]}", content=user_text)
            st.success("Saved to memories")

        if reflect_btn:
            # Consciousness Mirror: reflect user's message back with analysis
            add_chat('user', user_text)
            mirror_prompt = f"As a Consciousness Mirror, reflect back the following message in short bullet points about themes, habits, and a gentle question to prompt self-reflection:\n\n{user_text}"
            mirror_reply = generate_reply(mirror_prompt, temperature=0.6)
            add_chat('assistant', mirror_reply)
            st.markdown(f"**Consciousness Mirror**: {mirror_reply}")

    with right:
        st.header("Quick view")
        st.subheader("Recent memories")
        mems = get_recent_memories(6)
        for m in mems:
            st.write(f"- {m[1]} ({m[4][:10]})")
        st.markdown("---")
        st.subheader("Life timeline (latest)")
        timeline = get_timeline(6)
        for t in timeline:
            st.write(f"• {t[2]} — {t[1]}")
        st.markdown("---")
        st.subheader("Persona")
        persona_value = load_setting('persona') or "Adaptive, warm, and reflective."
        new_persona = st.text_area("Edit persona (how EchoSoul speaks)", value=persona_value, height=120)
        if st.button("Save persona"):
            save_setting('persona', new_persona)
            st.success("Persona updated")

elif mode == 'Chat history':
    st.header("Chat history and playback")
    history = get_chats(1000)
    selected = st.selectbox("Select a conversation segment to view", options=[f"{r[0]}: {r[1]} @ {r[3][:19]}" for r in history])
    if st.button("Show selected" ):
        idx = int(selected.split(":")[0])
        c = conn.cursor()
        c.execute("SELECT role,content,created_at FROM chats WHERE id=?", (idx,))
        r = c.fetchone()
        if r:
            st.write(f"**{r[0]}** ({r[2]}): {r[1]}")

elif mode == 'Life timeline':
    st.header("Life Timeline")
    with st.form(key='timeline_form'):
        ev_date = st.date_input("Event date", value=datetime.utcnow().date())
        ev_title = st.text_input("Event title")
        ev_details = st.text_area("Details")
        ev_tags = st.text_input("Tags (comma separated)")
        submit = st.form_submit_button("Add event")
    if submit:
        add_timeline_event(ev_title, ev_date.isoformat(), ev_details, ev_tags)
        st.success("Event added to timeline")
    st.markdown("### Timeline")
    ts = get_timeline(200)
    for t in ts:
        st.markdown(f"**{t[2]} - {t[1]}**\n{t[3]}\n")

elif m
