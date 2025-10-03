import os
import streamlit as st
import openai
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="EchoSoul", layout="wide")
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Chat", "Chat History", "Life Timeline", "Vault", "Export", "Brain Mimic", "About"])

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("⚠️ Please set your OpenAI API key in Streamlit Secrets or environment variables.")
openai.api_key = OPENAI_API_KEY

# Memory storage
if "history" not in st.session_state:
    st.session_state.history = []
if "timeline" not in st.session_state:
    st.session_state.timeline = []


# ======================
# AI RESPONSE GENERATION
# ======================
def generate_ai_response(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are EchoSoul, a personal AI that adapts to the user."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"(AI Error) {str(e)}"


# ======================
# CHAT PAGE
# ======================
if page == "Chat":
    st.title("💬 EchoSoul Chat")

    user_input = st.text_input("Say something to EchoSoul:")
    if st.button("Send") and user_input:
        ai_response = generate_ai_response(user_input)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append({"time": timestamp, "user": user_input, "ai": ai_response})
        st.session_state.timeline.append({"time": timestamp, "event": f"You: {user_input} | EchoSoul: {ai_response}"})

    for chat in reversed(st.session_state.history):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**EchoSoul:** {chat['ai']}")
        st.markdown("---")


# ======================
# CHAT HISTORY
# ======================
elif page == "Chat History":
    st.title("📖 Chat History")
    for chat in st.session_state.history:
        st.markdown(f"**[{chat['time']}] You:** {chat['user']}")
        st.markdown(f"**EchoSoul:** {chat['ai']}")
        st.markdown("---")


# ======================
# LIFE TIMELINE
# ======================
elif page == "Life Timeline":
    st.title("⏳ Life Timeline")
    for event in st.session_state.timeline:
        st.markdown(f"**[{event['time']}]** {event['event']}")
        st.markdown("---")


# ======================
# VAULT
# ======================
elif page == "Vault":
    st.title("🔐 Vault")
    st.info("A secure place for your important memories (demo).")


# ======================
# EXPORT
# ======================
elif page == "Export":
    st.title("📤 Export Your Data")
    st.download_button(
        "Download Chat History",
        data=str(st.session_state.history),
        file_name="echosoul_history.txt"
    )


# ======================
# BRAIN MIMIC
# ======================
elif page == "Brain Mimic":
    st.title("🧠 Brain Mimic")
    st.info("This feature will simulate your personality based on chat history (demo).")


# ======================
# ABOUT
# ======================
elif page == "About":
    st.title("ℹ️ About EchoSoul")
    st.write("EchoSoul is your personal adaptive AI. It remembers, adapts, and grows with you.")


# ======================
# VOICE CALL (BETA)
# ======================
st.sidebar.markdown("----")
st.sidebar.subheader("📞 Live Voice Call (Beta)")

class AudioProcessor(AudioProcessorBase):
    def recv_audio(self, frame):
        return frame

webrtc_streamer(
    key="voice",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False}
)
