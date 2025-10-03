# app.py - EchoSoul (fixed, NLP enabled, OpenAI-compatible, Call tab)
# ----------------------------------------------------------------
# Features:
# - Chat UI with persistent memory (SQLite)
# - Offline NLP sentiment classifier (transformers pipeline)
# - OpenAI LLM support (works with old and new openai packages)
# - Life timeline (add/list)
# - Private Vault (Fernet encryption)
# - TTS playback (gTTS fallback)
# - Call tab (WebRTC) when streamlit-webrtc available (simple audio echo / scaffold for STT->GPT->TTS)
# - Export (download JSON)
#
# Deploy notes:
# - requirements.txt should include: streamlit, transformers, torch, openai>=1.0.0, cryptography, gTTS, streamlit-webrtc (optional)
# - apt.txt should include ffmpeg, libsndfile1, libportaudio2, libopus0, libvpx7 if using audio/webrtc
# ----------------------------------------------------------------

import streamlit as st
import sqlite3
import json
import os
import io
import tempfile
import base64
import time
import traceback
from datetime import datetime, timezone
from typing import Optional, Tuple, List

# Optional / third-party libs (import gracefully)
try:
    # openai may be v0.x (old) or v1.x+ (new). We'll handle both.
    import openai  # used for old API fallback
except Exception:
    openai = None

try:
    # New-style openai client class
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

try:
    # Transformers NLP pipeline (for local sentiment/emotion)
    from transformers import pipeline
    _HF_AVAILABLE = True
except Exception:
    pipeline = None
    _HF_AVAILABLE = False

try:
    # Fernet encryption
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    _CRYPTO_AVAILABLE = True
except Exception:
    Fernet = None
    PBKDF2HMAC = None
    hashes = None
    _CRYPTO_AVAILABLE = False

try:
    # TTS
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except Exception:
    gTTS = None
    _GTTS_AVAILABLE = False

try:
    # WebRTC (optional)
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase
    import av
    _WEBRTC_AVAILABLE = True
except Exception:
    webrtc_streamer = None
    WebRtcMode = None
    RTCConfiguration = None
    AudioProcessorBase = object
    av = None
    _WEBRTC_AVAILABLE = False

# -------------------------
# DB initialization
# -------------------------
DB_FILE = "echosoul.db"


@st.cache_resource
def get_conn(db_file=DB_FILE):
    conn = sqlite3.connect(db_file, check_same_thread=False)
    c = conn.cursor()
    # chats: role=user/assistant, content, created_at
    c.execute(
        """CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            tags TEXT,
            created_at TEXT
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            event_date TEXT,
            details TEXT,
            created_at TEXT
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS vault (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            encrypted_blob TEXT,
            salt TEXT,
            created_at TEXT
        )"""
    )
    conn.commit()
    return conn


conn = get_conn()


# -------------------------
# NLP classifier (offline)
# -------------------------
_classifier = None
if _HF_AVAILABLE:
    try:
        # sentiment model - small and common
        _classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception:
        _classifier = None


def analyze_emotion_text(text: str) -> Tuple[str, float]:
    """
    Use transformers sentiment pipeline if available, otherwise fallback simple keywords.
    Returns (label, score)
    """
    if _classifier is not None:
        try:
            r = _classifier(text[:512])
            # r example: [{'label': 'POSITIVE', 'score': 0.9998}]
            lab = r[0]["label"].lower()
            score = float(r[0].get("score", 0.0))
            return (lab, score) if lab in ("positive", "negative") else ("neutral", score)
        except Exception:
            pass

    # fallback heuristic
    t = text.lower()
    pos = ["happy", "glad", "excited", "love", "good", "great", "awesome", "fantastic"]
    neg = ["sad", "depressed", "angry", "hate", "upset", "bad", "terrible"]
    s = sum(1 for w in pos if w in t) - sum(1 for w in neg if w in t)
    if s > 0:
        return ("positive", float(s))
    if s < 0:
        return ("negative", float(s))
    return ("neutral", 0.0)


# -------------------------
# Vault encryption helpers
# -------------------------
def _derive_key(password: str, salt: bytes) -> bytes:
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography not available")
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390000)
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt_str(plaintext: str, password: str) -> Tuple[str, str]:
    """
    Returns (token_b64, salt_b64)
    """
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography not available")
    salt = os.urandom(16)
    key = _derive_key(password, salt)
    f = Fernet(key)
    token = f.encrypt(plaintext.encode("utf-8"))
    return base64.b64encode(token).decode(), base64.b64encode(salt).decode()


def decrypt_str(token_b64: str, password: str, salt_b64: str) -> str:
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography not available")
    token = base64.b64decode(token_b64)
    salt = base64.b64decode(salt_b64)
    key = _derive_key(password, salt)
    f = Fernet(key)
    return f.decrypt(token).decode("utf-8")


# -------------------------
# DB helper functions
# -------------------------
def add_chat(role: str, content: str):
    c = conn.cursor()
    c.execute(
        "INSERT INTO chats (role, content, created_at) VALUES (?, ?, ?)",
        (role, content, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def get_chats(limit: int = 200) -> List[Tuple]:
    c = conn.cursor()
    c.execute("SELECT id, role, content, created_at FROM chats ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    return list(reversed(rows))


def add_memory(title: str, content: str):
    c = conn.cursor()
    c.execute(
        "INSERT INTO memories (title, content, tags, created_at) VALUES (?, ?, ?, ?)",
        (title, content, "", datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def get_memories(limit: int = 50):
    c = conn.cursor()
    c.execute("SELECT id, title, content, created_at FROM memories ORDER BY id DESC LIMIT ?", (limit,))
    return c.fetchall()


def add_timeline(title: str, event_date: str, details: str):
    c = conn.cursor()
    c.execute(
        "INSERT INTO timeline (title, event_date, details, created_at) VALUES (?, ?, ?, ?)",
        (title, event_date, details, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def get_timeline(limit: int = 200):
    c = conn.cursor()
    c.execute("SELECT id, title, event_date, details, created_at FROM timeline ORDER BY event_date DESC LIMIT ?", (limit,))
    return c.fetchall()


def add_vault_item(label: str, encrypted_blob: str, salt: str):
    c = conn.cursor()
    c.execute(
        "INSERT INTO vault (label, encrypted_blob, salt, created_at) VALUES (?, ?, ?, ?)",
        (label, encrypted_blob, salt, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def get_vault_items():
    c = conn.cursor()
    c.execute("SELECT id, label, encrypted_blob, salt, created_at FROM vault ORDER BY id DESC")
    return c.fetchall()


# -------------------------
# TTS helper
# -------------------------
def tts_bytes_from_text(text: str) -> Optional[bytes]:
    """
    Use gTTS if available to create mp3 bytes. Writes to temp file and returns bytes.
    """
    if not _GTTS_AVAILABLE:
        return None
    try:
        t = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            t.write_to_fp(tf)
            tempname = tf.name
        with open(tempname, "rb") as f:
            b = f.read()
        try:
            os.remove(tempname)
        except Exception:
            pass
        return b
    except Exception:
        return None


# -------------------------
# OpenAI compatibility helpers
# -------------------------
def make_openai_client(api_key: Optional[str]):
    """
    Returns either a new-style OpenAI client (OpenAIClient) or None.
    We'll also allow old-style openai module usage via fallback.
    """
    if not api_key:
        return None

    # prefer new client
    if OpenAIClient is not None:
        try:
            return OpenAIClient(api_key=api_key)
        except Exception:
            # fall through to None (we'll use old module)
            return None
    return None


def openai_chat_reply(client_or_none, messages: List[dict], api_key: Optional[str] = None) -> str:
    """
    Attempt to call OpenAI in a version-agnostic way.
    - If 'client_or_none' is a new OpenAI client with attribute .chat, call client.chat.completions.create(...)
    - Else try old openai.ChatCompletion.create(...)
    Returns text or raises/returns error string.
    """
    try:
        # New-style client (OpenAIClient)
        if client_or_none is not None and hasattr(client_or_none, "chat"):
            resp = client_or_none.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
            # resp.choices[0].message.content
            return resp.choices[0].message.content
        # Old-style fallback using openai module
        if openai is not None:
            # ensure api_key set
            if api_key:
                # older versions accept openai.api_key assignment
                try:
                    openai.api_key = api_key
                except Exception:
                    pass
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
            # resp might be dict-like
            # support both dict and object
            if isinstance(resp, dict):
                return resp["choices"][0]["message"]["content"]
            else:
                # object fallback
                return resp.choices[0].message.content
        raise RuntimeError("No OpenAI client available")
    except Exception as e:
        # return error string so UI can show it
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return f"(OpenAI error: {tb})"


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("EchoSoul — adaptive personal companion")

# Sidebar with persistent API key handling
with st.sidebar:
    st.header("EchoSoul")
    mode = st.radio("Mode", ["Chat", "Chat history", "Life timeline", "Vault", "Export", "Call", "About"], index=0)

    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""

    # display input initialized with current value so it persists across reruns
    api_key_input = st.text_input("OpenAI API Key (session only)", type="password", value=st.session_state["openai_api_key"])
    if api_key_input:
        st.session_state["openai_api_key"] = api_key_input

    st.markdown("---")
    st.write("Quick settings")
    tts_enabled = st.checkbox("Enable TTS playback (gTTS)", value=_GTTS_AVAILABLE)
    show_nlp_source = st.checkbox("Show NLP source (debug)", value=False)
    st.markdown("---")
    if st.button("Clear chat history"):
        c = conn.cursor()
        c.execute("DELETE FROM chats")
        conn.commit()
        st.sidebar.success("Chat history cleared.")
        st.experimental_rerun()

# helper to get openai client per session key
def get_client():
    return make_openai_client(st.session_state.get("openai_api_key", ""))


# -----------
# Chat page
# -----------
if mode == "Chat":
    st.header("💬 Chat with EchoSoul")
    # show chat history from DB
    rows = get_chats(500)
    for _id, role, content, created in rows:
        if role == "user":
            st.markdown(f"**user:** {content}")
        else:
            st.markdown(f"**assistant:** {content}")

    # input box
    user_input = st.text_area("Message", key="input_message", height=150)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        send = st.button("Send")
    with col2:
        save_memory_btn = st.button("Save to memory")
    with col3:
        reflect = st.button("Reflect (Consciousness Mirror)")

    if save_memory_btn and user_input.strip():
        add_memory(title=user_input[:80], content=user_input)
        st.success("Saved to memories")

    if send and user_input.strip():
        add_chat("user", user_input)

        # attempt GPT if key provided
        client = get_client()
        messages = [{"role": "system", "content": "You are EchoSoul — a compassionate personal companion."}]
        # include last few conversation turns for context
        hist = get_chats(20)
        for _id, role, content, created in hist[-12:]:
            messages.append({"role": role if role in ("user", "assistant") else "user", "content": content})
        messages.append({"role": "user", "content": user_input})

        reply = None
        if st.session_state.get("openai_api_key"):
            reply = openai_chat_reply(client, messages, api_key=st.session_state.get("openai_api_key"))
            # If openai returned an error string starting with (OpenAI error...), we show it (not fallback)
            if isinstance(reply, str) and reply.startswith("(OpenAI error:"):
                # store the error as assistant message so user sees it
                add_chat("assistant", reply)
                st.error("OpenAI error — see assistant message")
            else:
                add_chat("assistant", reply)
        else:
            # No API key -> offline fallback NLP + response
            emo, score = analyze_emotion_text(user_input)
            reply = f"I heard you. You seem {emo} (score={score})."
            add_chat("assistant", reply)

        # play TTS if enabled and available
        if tts_enabled and reply:
            audio_bytes = tts_bytes_from_text(reply)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

        st.experimental_rerun()

    if reflect and user_input.strip():
        # simple consciousness mirror: list themes + a reflective question
        emo, score = analyze_emotion_text(user_input)
        mirror = f"Reflection: I detect a {emo} tone (score={score}). Themes: "
        # naive theme extraction (keywords)
        themes = []
        keywords = ["work", "love", "family", "stress", "happy", "sad", "career", "health", "decision", "money"]
        lower = user_input.lower()
        for k in keywords:
            if k in lower:
                themes.append(k)
        mirror += (", ".join(themes) if themes else "general")
        mirror += ". Question: What outcome would make you feel most satisfied about this?"
        add_chat("assistant", mirror)
        st.success("Mirrored back. See assistant reply above.")
        st.experimental_rerun()

# -----------
# Chat history page
# -----------
elif mode == "Chat history":
    st.header("🗂 Chat History")
    rows = get_chats(1000)
    if not rows:
        st.info("No chat messages yet.")
    else:
        for _id, role, content, created in rows:
            st.write(f"{created[:19]} — **{role}**: {content}")

# -----------
# Timeline page
# -----------
elif mode == "Life timeline":
    st.header("📅 Life Timeline")
    with st.form("tl_form"):
        ev_title = st.text_input("Event title")
        ev_date = st.date_input("Event date", value=datetime.now(timezone.utc).date())
        ev_details = st.text_area("Details")
        submit = st.form_submit_button("Add event")
    if submit and ev_title.strip():
        add_timeline(ev_title, ev_date.isoformat(), ev_details)
        st.success("Event added to timeline")
        st.experimental_rerun()

    # list timeline
    items = get_timeline(200)
    if not items:
        st.info("No timeline events yet.")
    else:
        for _id, title, event_date, details, created in items:
            st.markdown(f"**{event_date}** — {title}\n\n{details}")

# -----------
# Vault page
# -----------
elif mode == "Vault":
    st.header("🔐 Private Vault")
    if not _CRYPTO_AVAILABLE:
        st.warning("Vault requires the 'cryptography' package. Please add it to requirements.txt to enable encryption.")
        st.info("While cryptography is not installed, you can still store plain text in memory using 'Save to memory' in Chat.")
    pw = st.text_input("Vault password (session)", type="password")
    if pw:
        st.session_state["vault_password"] = pw
    locked = not bool(st.session_state.get("vault_password"))
    if locked:
        st.info("Enter a vault password to unlock/add items for this session.")
    else:
        st.success("Vault unlocked for this session (in-memory).")
        # show items
        items = get_vault_items()
        if items:
            for _id, label, blob, salt, created in items:
                try:
                    plain = decrypt_str(blob, st.session_state["vault_password"], salt)
                    st.markdown(f"**{label}** ({created[:19]})\n\n{plain}")
                except Exception:
                    st.warning(f"Could not decrypt vault item '{label}' - wrong password or corrupted.")
        else:
            st.info("No vault items yet.")

        with st.form("vault_add"):
            vlabel = st.text_input("Label")
            vcontent = st.text_area("Secret content")
            addv = st.form_submit_button("Save to vault (encrypted)")
        if addv and vlabel.strip():
            try:
                token_b64, salt_b64 = encrypt_str(vcontent, st.session_state["vault_password"])
                add_vault_item(vlabel, token_b64, salt_b64)
                st.success("Saved to vault.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Encryption failed: {e}")

# -----------
# Export page
# -----------
elif mode == "Export":
    st.header("📤 Export / Download")
    payload = {
        "chats": get_chats(1000),
        "memories": get_memories(200),
        "timeline": get_timeline(200),
    }
    st.download_button("Download echo export (JSON)", data=json.dumps(payload, default=str, indent=2), file_name="echosoul_export.json")
    st.info("You can upload this JSON to import elsewhere (not implemented here).")

# -----------
# Call page (WebRTC)
# -----------
elif mode == "Call":
    st.header("📞 Live Call (WebRTC)")
    if not _WEBRTC_AVAILABLE:
        st.warning("streamlit-webrtc is not installed or failed to import. Install streamlit-webrtc and its OS deps to enable live in-browser audio.")
        st.info("You can still use Chat tab for typed conversations and TTS playback.")
    else:
        st.info("Start a live session; microphone audio will be streamed to Python. This demo echoes audio back and can be extended to STT->GPT->TTS.")
        # Minimal AudioProcessor: echo audio frames back.
        class EchoProcessor(AudioProcessorBase):
            def recv(self, frame: "av.AudioFrame") -> "av.AudioFrame":
                # simple echo: return the incoming frame unchanged
                return frame

        webrtc_streamer(
            key="echosoul-webrtc",
            mode=WebRtcM
