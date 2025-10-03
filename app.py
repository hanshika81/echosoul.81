import streamlit as st
import os
import json
import datetime
import hashlib
import base64
from cryptography.fernet import Fernet
from textblob import TextBlob
import pandas as pd

# Optional lazy imports (for voice call only)
try:
    from gtts import gTTS
    import speech_recognition as sr
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
except ImportError:
    gTTS = None
    sr = None
    webrtc_streamer = None

# ---------------- CONFIG ----------------
st.set_page_config(page_title="EchoSoul", page_icon="💠", layout="wide")

# API Key
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- STATE INIT ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "timeline" not in st.session_state:
    st.session_state.timeline = []
if "vault" not in st.session_state:
    st.session_state.vault = {}
if "personality" not in st.session_state:
    st.session_state.personality = "friendly"

# ---------------- HELPERS ----------------
def analyze_emotion(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "😊 Positive"
    elif polarity < -0.2:
        return "😞 Negative"
    return "😐 Neutral"

def encrypt_message(msg, password):
    key = hashlib.sha256(password.encode()).digest()
    fernet = Fernet(base64.urlsafe_b64encode(key))
    return fernet.encrypt(msg.encode()).decode()

def decrypt_message(token, password):
    key = hashlib.sha256(password.encode()).digest()
    fernet = Fernet(base64.urlsafe_b64encode(key))
    return fernet.decrypt(token.encode()).decode()

def generate_ai_response(prompt):
    if not client:
        return "(AI Error) No API key provided."
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are EchoSoul, a personal AI that adapts to the user."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"(AI Error) {e}"

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio("📌 Navigation", [
    "Chat", "Chat History", "Life Timeline", "Vault", "Export", "Brain Mimic", "About"
])

# ---------------- CHAT ----------------
if menu == "Chat":
    st.title("💬 EchoSoul Chat")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Say something to EchoSoul:", "")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        emotion = analyze_emotion(user_input)
        ai_response = generate_ai_response(user_input)

        st.session_state.chat_history.append(
            {"user": user_input, "ai": ai_response, "time": str(datetime.datetime.now()), "emotion": emotion}
        )
        st.session_state.timeline.append(
            {"event": user_input, "ai": ai_response, "date": str(datetime.datetime.now().date())}
        )

    # Show conversation
    for chat in st.session_state.chat_history[::-1]:
        st.markdown(f"**You:** {chat['user']}  \n**EchoSoul:** {chat['ai']}  \n*({chat['emotion']})*")

# ---------------- HISTORY ----------------
elif menu == "Chat History":
    st.title("📜 Chat History")
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.write(f"{chat['time']}: You: {chat['user']} → EchoSoul: {chat['ai']} ({chat['emotion']})")
    else:
        st.info("No history yet.")

# ---------------- TIMELINE ----------------
elif menu == "Life Timeline":
    st.title("🕰 Life Timeline")
    if st.session_state.timeline:
        df = pd.DataFrame(st.session_state.timeline)
        st.table(df)
    else:
        st.info("Timeline is empty.")

# ---------------- VAULT ----------------
elif menu == "Vault":
    st.title("🔐 Private Vault")
    password = st.text_input("Enter vault password:", type="password")
    action = st.radio("Action", ["Save", "View"])

    if action == "Save":
        secret = st.text_area("Enter secret memory:")
        if st.button("Save to Vault") and password and secret:
            key = hashlib.sha256(password.encode()).hexdigest()
            st.session_state.vault[key] = secret
            st.success("Memory saved securely!")

    elif action == "View":
        key = hashlib.sha256(password.encode()).hexdigest()
        if key in st.session_state.vault:
            st.success("Decrypted Memory:")
            st.write(st.session_state.vault[key])
        elif password:
            st.error("Incorrect password or empty vault.")

# ---------------- EXPORT ----------------
elif menu == "Export":
    st.title("📤 Export Data")
    export_data = {
        "chat_history": st.session_state.chat_history,
        "timeline": st.session_state.timeline,
        "vault": st.session_state.vault,
    }
    st.download_button(
        "Download EchoSoul Backup",
        data=json.dumps(export_data, indent=2),
        file_name="echosoul_backup.json",
        mime="application/json"
    )

# ---------------- BRAIN MIMIC ----------------
elif menu == "Brain Mimic":
    st.title("🧠 Brain Mimic Mode")
    st.info("EchoSoul will attempt to answer as if it were you, based on past chats.")
    user_question = st.text_input("Ask something you would ask yourself:")
    if user_question:
        context = " ".join([c["user"] for c in st.session_state.chat_history[-5:]])
        mimic_response = generate_ai_response(f"Answer like the user would. Past style: {context}. Q: {user_question}")
        st.write(f"🌀 {mimic_response}")

# ---------------- ABOUT ----------------
elif menu == "About":
    st.title("ℹ️ About EchoSoul")
    st.markdown("""
    **EchoSoul** is your evolving AI companion 💠  
    - Remembers your history  
    - Adapts personality  
    - Recognizes emotion  
    - Has a secure memory vault  
    - Can mimic your thought style  
    - Supports voice calls (if enabled)  
    """)

# ---------------- VOICE CALL FEATURE ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("📞 Live Voice Call (Beta)")

if webrtc_streamer and gTTS:
    class EchoProcessor(AudioProcessorBase):
        def recv(self, frame):
            return frame  # passthrough for now

    webrtc_streamer(
        key="call",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=EchoProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )
    st.sidebar.success("Voice call running! 🎙")
else:
    st.sidebar.warning("Voice features unavailable (missing packages).")
