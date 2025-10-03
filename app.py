# EchoSoul - app.py (complete)
# -------------------------------------------------------------
# Full EchoSoul app for Streamlit Cloud with chat, memory, vault,
# brain-mimic, simulated & live VoIP calls (streamlit-webrtc).
#
# Requirements (put in requirements.txt):
# streamlit
# openai>=1.0.0
# cryptography
# pandas
# numpy
# aiortc
# av
# streamlit-webrtc==0.64.0
# gTTS
# pydub
# twilio (optional)
#
# Note: Install the above packages on your deployment environment.
# -------------------------------------------------------------

import streamlit as st
import sqlite3
import json
import os
import time
import base64
import io
from datetime import datetime
from typing import Optional, List, Tuple

# Optional third-party libs
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

# WebRTC
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

# -------------------- DB & initialization --------------------
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

# -------------------- helpers --------------------

_POS_WORDS = set(["happy","great","awesome","love","joy","excited","pleased","good","fantastic","wonderful"])
_NEG_WORDS = set(["sad","angry","hate","terrible","upset","bad","depressed","annoyed","frustrated"])

# encryption helpers
def _derive_key(password: str, salt: bytes) -> bytes:
    if PBKDF2HMAC is None:
        raise RuntimeError("cryptography library required for vault encryption")
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390000)
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def encrypt_text(plaintext: str, password: str) -> Tuple[str,str]:
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

# DB helpers
def add_chat(role: str, content: str):
    c = conn.cursor()
    c.execute("INSERT INTO chats (role,content,created_at) VALUES (?,?,?)", (role, content, datetime.utcnow().isoformat()))
    conn.commit()

def get_chats(limit=200):
    c = conn.cursor()
    c.execute("SELECT id,role,content,created_at FROM chats ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    return list(reversed(rows))

def add_memory(title: str, content: str, tags: Optional[str] = ""):
    c = conn.cursor()
    c.execute("INSERT INTO memories (title,content,tags,created_at) VALUES (?,?,?,?)", (title, content, tags, datetime.utcnow().isoformat()))
    conn.commit()

def get_recent_memories(limit=10):
    c = conn.cursor()
    c.execute("SELECT id,title,content,tags,created_at FROM memories ORDER BY id DESC LIMIT ?", (limit,))
    return c.fetchall()

def add_timeline_event(event: str, event_date: str, details: str, tags: str = ""):
    c = conn.cursor()
    c.execute("INSERT INTO timeline (event,event_date,details,tags,created_at) VALUES (?,?,?,?,?)", (event, event_date, details, tags, datetime.utcnow().isoformat()))
    conn.commit()

def get_timeline(limit=100):
    c = conn.cursor()
    c.execute("SELECT id,event,event_date,details,tags,created_at FROM timeline ORDER BY event_date DESC LIMIT ?", (limit,))
    return c.fetchall()

def add_vault_item(label: str, encrypted_blob: str, salt: str):
    c = conn.cursor()
    c.execute("INSERT INTO vault (label,encrypted_blob,salt,created_at) VALUES (?,?,?,?)", (label, encrypted_blob, salt, datetime.utcnow().isoformat()))
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

# emotion
def detect_emotion_from_text(text: str) -> Tuple[str,float]:
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

# OpenAI integration (basic)
DEFAULT_MODEL = "gpt-3.5-turbo"

def configure_openai_from_key(key: Optional[str]):
    if not key or openai is None:
        return False
    openai.api_key = key
    return True

def openai_chat(messages: list, model: str = DEFAULT_MODEL, temperature: float = 0.7) -> Optional[str]:
    if openai is None:
        return None
    try:
        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
        return resp['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"OpenAI Chat error: {e}")
        return None

# TTS fallback (gTTS)
def tts_gtts_generate_bytes(text: str, lang: str = 'en') -> Optional[bytes]:
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

# Reply generation
def build_system_prompt(mimic: bool=False, mimic_examples: list=None) -> str:
    persona = load_setting('persona') or "You are EchoSoul — a compassionate adaptive companion."
    mems = get_recent_memories(6)
    mem_text = "\n".join([f"- {m[1]}: {m[2]}" for m in mems])
    prompt = f"{persona}\nRecent memories:\n{mem_text}\n\nRules:\n- Protect privacy.\n- Use gentle, reflective tone.\n"
    if mimic and mimic_examples:
        prompt += "Mimic examples:\n"
        for ex in mimic_examples[-6:]:
            prompt += f"- {ex}\n"
    return prompt

def generate_reply(user_text: str, mimic: bool=False, mimic_examples: list=None, temp: float=0.7) -> str:
    sys = build_system_prompt(mimic=mimic, mimic_examples=mimic_examples)
    messages = [{"role":"system","content":sys}, {"role":"user","content":user_text}]
    reply = openai_chat(messages, temperature=temp) if openai else None
    if reply:
        return reply.strip()
    # fallback
    emo,score = detect_emotion_from_text(user_text)
    return f"I hear you. You seem {emo} (score {score}). Tell me more."

# Life path simulation
def simulate_scenarios(context: str, options: List[str]) -> str:
    prompt = f"You are a life-path simulator. Context: {context}\nOptions:\n"
    for i,opt in enumerate(options,1):
        prompt += f"{i}. {opt}\n"
    prompt += "For each option, give a short simulated outcome, pros/cons, and a recommendation."
    return generate_reply(prompt, temp=0.8)

# Export / import
def export_echo_json():
    c = conn.cursor()
    c.execute("SELECT id,role,content,created_at FROM chats ORDER BY id ASC")
    chats = c.fetchall()
    c.execute("SELECT id,title,content,tags,created_at FROM memories ORDER BY id ASC")
    memories = c.fetchall()
    c.execute("SELECT id,event,event_date,details,tags,created_at FROM timeline ORDER BY id ASC")
    timeline = c.fetchall()
    payload = {"exported_at":datetime.utcnow().isoformat(),"chats":chats,"memories":memories,"timeline":timeline,"settings":{}}
    persona = load_setting('persona')
    if persona:
        payload['settings']['persona'] = persona
    return payload

def import_echo_json(data: dict) -> Tuple[int,int,int]:
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

# -------------------- WebRTC Audio Processor --------------------
# This processor receives audio frames from the browser, buffers short segments,
# attempts to transcribe using OpenAI (if key provided), generates a reply, and
# plays the TTS back into the call. This is a best-effort single-peer approach.

class EchoAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = bytearray()
        # typical sample rate for browsers is 48000
        self.sample_rate = 48000
        self.channels = 1
        self._last_action = time.time()

    def recv(self, frame: "av.AudioFrame") -> "av.AudioFrame":
        # Called for each incoming frame. We'll collect ~2 seconds of audio and process.
        try:
            arr = frame.to_ndarray()
            # arr shape might be (channels, samples) or (samples,)
            if arr.ndim > 1:
                # take first channel
                samples = arr[0]
            else:
                samples = arr
            pcm = samples.tobytes()
            self.buffer.extend(pcm)
            duration = len(self.buffer) / 2 / self.sample_rate  # 2 bytes per sample
            if duration > 2.5 and (time.time() - self._last_action) > 1.0:
                audio_bytes = bytes(self.buffer)
                self.buffer = bytearray()
                self._last_action = time.time()
                transcript = None
                if openai and getattr(openai, 'api_key', None):
                    try:
                        # Use pydub to wrap raw PCM into WAV if available
                        audio_file = io.BytesIO()
                        if AudioSegment is not None:
                            seg = AudioSegment(audio_bytes, sample_width=2, frame_rate=self.sample_rate, channels=1)
                            seg.export(audio_file, format="wav")
                            audio_file.seek(0)
                        else:
                            audio_file.write(audio_bytes)
                            audio_file.seek(0)
                        # transcribe with whisper if available
                        try:
                            resp = openai.Audio.transcribe('whisper-1', audio_file)
                            transcript = resp.get('text')
                        except Exception:
                            transcript = None
                    except Exception:
                        transcript = None
                if transcript:
                    # record user speech and generate reply
                    add_chat('user', transcript)
                    reply = generate_reply(transcript)
                    add_chat('assistant', reply)
                    # TTS
                    tts = tts_gtts_generate_bytes(reply) if gTTS else None
                    if tts and AudioSegment is not None:
                        audio_file = io.BytesIO(tts)
                        seg = AudioSegment.from_file(audio_file, format='mp3')
                        seg = seg.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
                        samples = seg.get_array_of_samples()
                        # create frame and return to be played
                        frame_out = av.AudioFrame.from_ndarray(seg.get_array_of_samples(), format='s16', layout='mono')
                        frame_out.sample_rate = self.sample_rate
                        return frame_out
                # otherwise return original frame (passthrough)
            return frame
        except Exception as e:
            print("Processor error:", e)
            return frame

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title='EchoSoul', layout='wide')

# Sidebar
with st.sidebar:
    st.title('EchoSoul')
    st.markdown('Adaptive personal companion — chat, call, remember.')
    mode = st.radio('Mode', ['Chat','Chat history','Life timeline','Vault','Export','Brain mimic','Call','About'], index=0)
    st.markdown('---')
    st.subheader('Settings')
    api_key_input = st.text_input('OpenAI API Key (session only)', type='password')
    if api_key_input:
        ok = configure_openai_from_key(api_key_input)
        if ok:
            st.success('OpenAI key set for session')
    voice_choice = st.selectbox('Voice (for TTS)', ['Default','Warm Female','Calm Male','Deep Narrator','Adaptive'])
    mimic_enabled = st.checkbox('Enable Brain Mimic', value=False)
    mimic_strength = st.slider('Mimic strength', 0, 100, 30)
    st.markdown('---')
    if st.button('Clear chat history'):
        c = conn.cursor()
        c.execute('DELETE FROM chats')
        conn.commit()
        st.experimental_rerun()

# session state
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''
if 'call_active' not in st.session_state:
    st.session_state['call_active'] = False
if 'vault_unlocked' not in st.session_state:
    st.session_state['vault_unlocked'] = False
if 'vault_password' not in st.session_state:
    st.session_state['vault_password'] = None

# Pages
if mode == 'Chat':
    left, right = st.columns([3,1])
    with left:
        st.header('Chat with EchoSoul')
        chats = get_chats(200)
        for cid, role, content, created_at in chats:
            if role == 'user':
                st.markdown(f"<div style='background:#e6f7ff;padding:8px;border-radius:8px'><strong>You</strong>: {content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#fff7e6;padding:8px;border-radius:8px'><strong>EchoSoul</strong>: {content}</div>", unsafe_allow_html=True)
        st.markdown('---')
        with st.form('chat_form', clear_on_submit=False):
            user_text = st.text_area('Message', value=st.session_state['user_input'], key='input_area', height=120)
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                send_btn = st.form_submit_button('Send')
            with col2:
                save_mem_btn = st.form_submit_button('Save to memory')
            with col3:
                reflect_btn = st.form_submit_button('Reflect (Consciousness Mirror)')
        if send_btn:
            # clear chat box automatically per requirement
            st.session_state['user_input'] = ''
            add_chat('user', user_text)
            recent_user_msgs = [c[2] for c in get_chats(200) if c[1]=='user']
            mimic_examples = recent_user_msgs[-12:] if mimic_enabled else []
            reply = generate_reply(user_text, mimic=mimic_enabled, mimic_examples=mimic_examples, temp=0.8)
            add_chat('assistant', reply)
            emo_label, emo_score = detect_emotion_from_text(user_text)
            st.success(f"Emotion detected: {emo_label} (score {emo_score})")
            # Auto-memory heuristics
            if len(user_text) > 200 or "important" in user_text.lower():
                add_memory(title=f"Auto: {user_text[:30]}...", content=user_text)
            # TTS playback (gTTS fallback)
            tts_bytes = tts_gtts_generate_bytes(reply) if gTTS else None
            if tts_bytes:
                st.audio(tts_bytes, format='audio/mp3')
            st.experimental_rerun()
        if save_mem_btn:
            add_memory(title=f"Manual: {user_text[:40]}", content=user_text)
            st.success("Saved to memories")
        if reflect_btn:
            add_chat('user', user_text)
            mirror_prompt = f"As a Consciousness Mirror, reflect back the following message in short bullet points about themes, habits, and a gentle question to prompt self-reflection:\n\n{user_text}"
            mirror_reply = generate_reply(mirror_prompt, temp=0.6)
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
    options = [f"{r[0]}: {r[1]} @ {r[3][:19]}" for r in history]
    selected = st.selectbox("Select a conversation segment to view", options=options if options else ["No chats"])
    if st.button("Show selected") and options:
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
        add_timeline_event(ev_title, ev_date.isoformat(), ev_details
