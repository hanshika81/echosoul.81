# EchoSoul - app.py (Fixed + Combined)
# -------------------------------------------------------------
# Features:
# - Chat UI with persistent memory (SQLite)
# - Adaptive persona & Brain Mimic
# - Emotion detection & Consciousness Mirror
# - Life timeline
# - Private Vault (Fernet encryption)
# - Export / Import (Soul Resonance)
# - Simulated call (upload/typed) with gTTS fallback
# - Live in-browser VoIP via streamlit-webrtc with a simple audio processor
# - Uses new OpenAI Python client API (openai.OpenAI)
#
# Requirements: see requirements.txt generated earlier.
# -------------------------------------------------------------

import streamlit as st
import sqlite3
import json
import os
import io
import time
import base64
from datetime import datetime
from typing import Optional, List, Tuple

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

# streamlit-webrtc
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
_POS_WORDS = set(["happy","great","awesome","love","joy","excited","pleased","good","fantastic","wonderful"])
_NEG_WORDS = set(["sad","angry","hate","terrible","upset","bad","depressed","annoyed","frustrated"])

def detect_emotion_from_text(text: str) -> Tuple[str, float]:
    text_l = text.lower()
    score = 0
    for w in _POS_WORDS:
        if w in text_l:
            score += 1
    for w in _NEG_WORDS:
        if w in text_l:
            score -= 1
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

# ------------- Encryption (Vault) -------------
def _derive_key(password: str, salt: bytes) -> bytes:
    if PBKDF2HMAC is None:
        raise RuntimeError("cryptography library required for vault encryption")
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390000)
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_text(plaintext: str, password: str) -> Tuple[str, str]:
    if Fernet is None:
        raise RuntimeError("cryptography library required for vault encryption")
    salt = os.urandom(16)
    key = _derive_key(password, salt)
    f = Fernet(key)
    token = f.encrypt(plaintext.encode())
    return base64.b64encode(token).decode(), base64.b64encode(salt).decode()

def decrypt_text(token_b64: str, password: str, salt_b64: str) -> str:
    if Fernet is None:
        raise RuntimeError("cryptography library required for vault encryption")
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

# ------------- OpenAI client helpers (v1) -------------
def make_openai_client(api_key: Optional[str]):
    if openai is None or not api_key:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        return client
    except Exception:
        return None

def openai_chat_reply(client, messages: list, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
    if client is None:
        return None
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    # response access: resp.choices[0].message.content
    return resp.choices[0].message.content

def openai_transcribe_audio(client, audio_file_obj):
    if client is None:
        return None
    # new API: client.audio.transcriptions.create(file=..., model="whisper-1")
    resp = client.audio.transcriptions.create(model="whisper-1", file=audio_file_obj)
    return getattr(resp, "text", resp.get("text") if isinstance(resp, dict) else None)

# ------------- TTS fallback (gTTS) -------------
def tts_gtts_bytes(text: str, lang: str = "en"):
    if gTTS is None:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None

# ------------- System prompt / reply generator -------------
def build_system_prompt(mimic: bool=False, mimic_examples: list=None) -> str:
    persona = load_setting('persona') or "You are EchoSoul — a compassionate, adaptive companion."
    mems = get_recent_memories(6)
    mem_text = "\n".join([f"- {m[1]}: {m[2]}" for m in mems])
    prompt = f"{persona}\n\nRecent memories:\n{mem_text}\n\nRules:\n- Protect privacy and Vault items.\n- Be reflective and helpful.\n"
    if mimic and mimic_examples:
        prompt += "\nMimic examples:\n"
        for ex in mimic_examples[-6:]:
            prompt += f"- {ex}\n"
    return prompt

def generate_reply(client, user_text: str, mimic: bool=False, mimic_examples: list=None, temp: float=0.7) -> str:
    sys_prompt = build_system_prompt(mimic=mimic, mimic_examples=mimic_examples)
    messages = [{"role":"system","content":sys_prompt}, {"role":"user","content":user_text}]
    if client:
        try:
            r = openai_chat_reply(client, messages, model="gpt-3.5-turbo", temperature=temp)
            if r:
                return r.strip()
        except Exception as e:
            # fall through to fallback
            print("OpenAI chat error:", e)
    emo, score = detect_emotion_from_text(user_text)
    return f"I heard you. You seem {emo} (score={score}). Tell me more."

# ------------- WebRTC Audio Processor (simplified) -------------
class EchoAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = bytearray()
        self.sample_rate = 48000
        self._last_action = time.time()

    def recv(self, frame: "av.AudioFrame") -> "av.AudioFrame":
        # Collect audio and attempt periodic transcription + reply (best-effort)
        try:
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                samples = arr[0]
            else:
                samples = arr
            pcm = samples.tobytes()
            self.buffer.extend(pcm)
            duration = len(self.buffer) / 2 / self.sample_rate  # 2 bytes/sample
            if duration > 2.5 and (time.time() - self._last_action) > 1.0:
                audio_bytes = bytes(self.buffer)
                self.buffer = bytearray()
                self._last_action = time.time()
                # Try to transcribe via OpenAI (if key set)
                client = make_openai_client(st.session_state.get("openai_api_key", ""))
                transcript = None
                if client:
                    try:
                        audio_file = io.BytesIO()
                        if AudioSegment is not None:
                            seg = AudioSegment(audio_bytes, sample_width=2, frame_rate=self.sample_rate, channels=1)
                            seg.export(audio_file, format="wav")
                            audio_file.seek(0)
                        else:
                            audio_file.write(audio_bytes)
                            audio_file.seek(0)
                        transcript = openai_transcribe_audio(client, audio_file)
                    except Exception:
                        transcript = None
                if transcript:
                    add_chat("user", transcript)
                    reply = generate_reply(client, transcript)
                    add_chat("assistant", reply)
                    # Play back TTS via gTTS if available
                    tts_bytes = tts_gtts_bytes(reply) if gTTS else None
                    if tts_bytes and AudioSegment is not None:
                        audio_file = io.BytesIO(tts_bytes)
                        seg = AudioSegment.from_file(audio_file, format="mp3")
                        seg = seg.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
                        samples = seg.get_array_of_samples()
                        # create and return audio frame for playback
                        frame_out = av.AudioFrame.from_ndarray(seg.get_array_of_samples(), format="s16", layout="mono")
                        frame_out.sample_rate = self.sample_rate
                        return frame_out
            return frame
        except Exception as e:
            print("Audio processor error:", e)
            return frame

# ------------- Streamlit UI -------------
st.set_page_config(page_title="EchoSoul", layout="wide")

with st.sidebar:
    st.title("EchoSoul")
    st.markdown("Adaptive personal companion — chat, call, remember.")
    mode = st.radio("Mode", ["Chat","Chat history","Life timeline","Vault","Export","Brain mimic","Call","About"], index=0)
    st.markdown("---")
    st.subheader("Settings")
    openai_key_input = st.text_input("OpenAI API Key (session only)", type="password")
    if openai_key_input:
        st.session_state["openai_api_key"] = openai_key_input
        st.success("OpenAI key stored for this session.")
    voice_choice = st.selectbox("Voice (for TTS)", ["Default","Warm Female","Calm Male","Deep Narrator","Adaptive"])
    mimic_enabled = st.checkbox("Enable Brain Mimic", value=False)
    mimic_strength = st.slider("Mimic strength", 0, 100, 30)
    st.markdown("---")
    if st.button("Clear chat history"):
        c = conn.cursor()
        c.execute("DELETE FROM chats")
        conn.commit()
        st.success("Chat history cleared.")
        st.rerun()

# Ensure session state fields
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "call_active" not in st.session_state:
    st.session_state["call_active"] = False
if "vault_unlocked" not in st.session_state:
    st.session_state["vault_unlocked"] = False
if "vault_password" not in st.session_state:
    st.session_state["vault_password"] = ""

# Main pages
if mode == "Chat":
    left, right = st.columns([3,1])
    with left:
        st.header("Chat with EchoSoul")
        chats = get_chats(200)
        for cid, role, content, created_at in chats:
            if role == "user":
                st.markdown(f"<div style='background:#e6f7ff;padding:8px;border-radius:8px'><strong>You</strong>: {content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#fff7e6;padding:8px;border-radius:8px'><strong>EchoSoul</strong>: {content}</div>", unsafe_allow_html=True)
        st.markdown("---")
        with st.form("chat_form", clear_on_submit=False):
            user_text = st.text_area("Message", value=st.session_state["user_input"], key="input_area", height=120)
            c1,c2,c3 = st.columns([1,1,1])
            with c1:
                send_btn = st.form_submit_button("Send")
            with c2:
                save_mem_btn = st.form_submit_button("Save to memory")
            with c3:
                reflect_btn = st.form_submit_button("Reflect (Consciousness Mirror)")
        if send_btn:
            st.session_state["user_input"] = ""
            add_chat("user", user_text)
            # generate reply
            client = make_openai_client(st.session_state.get("openai_api_key", ""))
            recent_user_msgs = [c[2] for c in get_chats(200) if c[1]=="user"]
            mimic_examples = recent_user_msgs[-12:] if mimic_enabled else []
            reply = generate_reply(client, user_text, mimic=mimic_enabled, mimic_examples=mimic_examples, temp=0.8)
            add_chat("assistant", reply)
            emo_label, emo_score = detect_emotion_from_text(user_text)
            st.success(f"Emotion detected: {emo_label} (score {emo_score})")
            # auto memory heuristics
            if len(user_text) > 200 or "important" in user_text.lower():
                add_memory(title=f"Auto: {user_text[:30]}...", content=user_text)
            # TTS playback (gTTS)
            tts_bytes = tts_gtts_bytes(reply) if gTTS else None
            if tts_bytes:
                st.audio(tts_bytes, format="audio/mp3")
            st.rerun()
        if save_mem_btn:
            add_memory(title=f"Manual: {user_text[:40]}", content=user_text)
            st.success("Saved to memories")
        if reflect_btn:
            add_chat("user", user_text)
            mirror_prompt = f"As a Consciousness Mirror, reflect back the following message in short bullet points about themes, habits, and a gentle question to prompt self-reflection:\n\n{user_text}"
            client = make_openai_client(st.session_state.get("openai_api_key", ""))
            mirror_reply = generate_reply(client, mirror_prompt, temp=0.6)
            add_chat("assistant", mirror_reply)
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
        persona_value = load_setting("persona") or "Adaptive, warm, and reflective."
        new_persona = st.text_area("Edit persona (how EchoSoul speaks)", value=persona_value, height=120)
        if st.button("Save persona"):
            save_setting("persona", new_persona)
            st.success("Persona updated")

elif mode == "Chat history":
    st.header("Chat history and playback")
    history = get_chats(1000)
    options = [f"{r[0]}: {r[1]} @ {r[3][:19]}" for r in history]
    selected = st.selectbox("Select a conversation segment to view", options=options if options else ["No chats"])
    if st.button("Show selected") and options:
        idx = int(selected.split(":")[0])
        cur = conn.cursor()
        cur.execute("SELECT role,content,created_at FROM chats WHERE id=?", (idx,))
        r = cur.fetchone()
        if r:
            st.write(f"**{r[0]}** ({r[2]}): {r[1]}")

elif mode == "Life timeline":
    st.header("Life Timeline")
    with st.form(key="timeline_form"):
        ev_date = st.date_input("Event date", value=datetime.utcnow().date())
        ev_title = st.text_input("Event title")
        ev_details = st.text_area("Details")
        submit = st.form_submit_button("Add event")
    if submit:
        add_timeline_event(ev_title, ev_date.isoformat(), ev_details)
        st.success("Event added to timeline")
    st.markdown("### Timeline")
    ts = get_timeline(200)
    for t in ts:
        st.markdown(f"**{t[2]} - {t[1]}**\n{t[3]}\n")

elif mode == "Vault":
    st.header("Private Vault")
    st.markdown("Store sensitive memories encrypted with a password. You must remember the vault password to decrypt.")
    vault_password = st.text_input("Vault password (session)", type="password")
    if st.button("Unlock Vault") and vault_password:
        st.session_state["vault_unlocked"] = True
        st.session_state["vault_password"] = vault_password
    if st.session_state.get("vault_unlocked"):
        st.success("Vault unlocked for this session.")
        items = get_vault_items()
        for it in items:
            try:
                content = decrypt_text(it[2], st.session_state["vault_password"], it[3])
                st.write(f"**{it[1]}** ({it[4][:19]})")
                st.write(content)
                st.markdown("---")
            except Exception:
                st.warning(f"Could not decrypt {it[1]} — wrong password?")
        st.markdown("### Add new vault item")
        with st.form("vault_add"):
            vlabel = st.text_input("Label")
            vcontent = st.text_area("Secret content")
            addvault = st.form_submit_button("Save to vault")
        if addv
