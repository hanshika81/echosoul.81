import os
import json
import hashlib
import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from openai import OpenAI
import speech_recognition as sr
from gtts import gTTS
import tempfile

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="EchoSoul", layout="wide", page_icon="💫")

# --------------------------
# Initialize OpenAI client
# --------------------------
api_key = os.getenv("OPENAI_API_KEY")
client = None
if api_key:
    client = OpenAI(api_key=api_key)

# --------------------------
# Session State
# --------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "timeline" not in st.session_state:
    st.session_state.timeline = []
if "vault_unlocked" not in st.session_state:
    st.session_state.vault_unlocked = False
if "vault" not in st.session_state:
    st.session_state.vault = []
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Chat", "Chat History", "Life Timeline", "Vault", "Export", "Brain Mimic", "About"]
)

st.sidebar.markdown("----")
st.sidebar.subheader("📞 Live Voice Call (Beta)")

# --------------------------
# Main Pages
# --------------------------

# --- Chat Page ---
if menu == "Chat":
    st.title("💬 EchoSoul Chat")

    user_input = st.text_input("Say something to EchoSoul:", key="chat_input")
    if st.button("Send"):
        if not api_key:
            st.error("❌ Missing API key. Please add your OpenAI key in Streamlit Secrets.")
        elif user_input.strip() == "":
            st.warning("Please type something first.")
        else:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are EchoSoul, a personal AI that adapts, remembers, and reflects the user."},
                        {"role": "user", "content": user_input}
                    ]
                )
                reply = response.choices[0].message.content

                # Save to session memory
                st.session_state.chat_history.append({"user": user_input, "ai": reply})
                st.session_state.timeline.append(
                    {"time": str(datetime.datetime.now()), "event": f"Chatted: {user_input}"}
                )

                # Clear input
                st.session_state.chat_input = ""

            except Exception as e:
                st.error(f"(AI Error) {str(e)}")

    # Always render conversation below
    if st.session_state.chat_history:
        st.subheader("Conversation")
        for chat in st.session_state.chat_history[-10:]:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**EchoSoul:** {chat['ai']}")
            st.markdown("---")

# --- Chat History ---
elif menu == "Chat History":
    st.title("📜 Chat History")
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**EchoSoul:** {chat['ai']}")
            st.markdown("---")
    else:
        st.info("No chat history yet.")

# --- Life Timeline ---
elif menu == "Life Timeline":
    st.title("🧭 Life Timeline")
    if st.session_state.timeline:
        for event in st.session_state.timeline:
            st.write(f"{event['time']} - {event['event']}")
    else:
        st.info("No timeline events recorded yet.")

# --- Vault ---
elif menu == "Vault":
    st.title("🔐 Private Vault")

    if not st.session_state.vault_unlocked:
        password = st.text_input("Enter vault password:", type="password")
        if st.button("Unlock Vault"):
            hashed = hashlib.sha256(password.encode()).hexdigest()
            # Replace "mysecret" with your actual vault password
            if hashed == hashlib.sha256("mysecret".encode()).hexdigest():
                st.session_state.vault_unlocked = True
                st.success("✅ Vault unlocked!")
            else:
                st.error("❌ Incorrect password.")
    else:
        st.success("Vault is unlocked ✅")
        note = st.text_area("Add a secret memory:")
        if st.button("Save Memory"):
            st.session_state.vault.append({"time": str(datetime.datetime.now()), "memory": note})
        st.subheader("Stored Memories")
        for mem in st.session_state.vault:
            st.write(f"{mem['time']}: {mem['memory']}")

# --- Export ---
elif menu == "Export":
    st.title("📤 Export Data")
    data = {
        "chat_history": st.session_state.chat_history,
        "timeline": st.session_state.timeline,
        "vault": st.session_state.vault
    }
    st.download_button(
        "Download My Data",
        data=json.dumps(data, indent=2),
        file_name="echosoul_data.json"
    )

# --- Brain Mimic ---
elif menu == "Brain Mimic":
    st.title("🧠 Brain Mimic")
    st.info("EchoSoul will reply as if it were you, using your stored chats & memories.")

    mimic_input = st.text_input("Ask EchoSoul-as-You something:")
    if st.button("Mimic Reply"):
        if not api_key:
            st.error("❌ Missing API key.")
        else:
            try:
                context = "\n".join(
                    [f"You: {c['user']} / AI: {c['ai']}" for c in st.session_state.chat_history[-10:]]
                )
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"Imitate the user’s personality based on this history:\n{context}"},
                        {"role": "user", "content": mimic_input}
                    ]
                )
                reply = response.choices[0].message.content
                st.markdown(f"**EchoSoul-as-You:** {reply}")
            except Exception as e:
                st.error(f"(AI Error) {str(e)}")

# --- About ---
elif menu == "About":
    st.title("ℹ️ About EchoSoul")
    st.write("""
    EchoSoul is your evolving AI companion:
    - Persistent Memory
    - Adaptive Personality
    - Emotion Recognition
    - Life Timeline
    - Vault (encrypted memories)
    - Brain Mimic
    - Export & Backup
    - Live Voice (beta)
    - Legacy Mode (coming soon)
    """)

# --------------------------
# Voice Call (Beta)
# --------------------------
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame):
        # Capture audio frame → speech recognition
        r = sr.Recognizer()
        with sr.AudioFile(frame.to_ndarray().tobytes()) as source:
            try:
                text = r.recognize_google(r.record(source))
                if text.strip():
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are EchoSoul speaking in a natural voice."},
                            {"role": "user", "content": text}
                        ]
                    )
                    reply = response.choices[0].message.content

                    # Save chat + timeline
                    st.session_state.chat_history.append({"user": text, "ai": reply})
                    st.session_state.timeline.append(
                        {"time": str(datetime.datetime.now()), "event": f"Voice chat: {text}"}
                    )

                    # Convert reply to audio
                    tts = gTTS(reply)
                    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(tmpfile.name)
                    st.audio(tmpfile.name, format="audio/mp3")
            except Exception:
                pass
        return frame

st.sidebar.markdown("----")
st.sidebar.write("🎙️ Try Live Voice (Beta)")
webrtc_streamer(
    key="voice",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)
