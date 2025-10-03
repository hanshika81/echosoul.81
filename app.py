import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import datetime
import random
from pathlib import Path
from dotenv import load_dotenv
import openai

# Load .env for API keys
load_dotenv()

# -----------------------------
# Session state initialization
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "timeline" not in st.session_state:
    st.session_state.timeline = []

if "memory" not in st.session_state:
    st.session_state.memory = {}

# -----------------------------
# Helper functions
# -----------------------------
def analyze_emotion(text: str):
    """Naive sentiment analysis fallback."""
    positive_words = ["happy", "great", "good", "love", "awesome", "fantastic"]
    negative_words = ["sad", "bad", "angry", "hate", "horrible", "upset"]

    score = 0
    txt = text.lower()
    for w in positive_words:
        if w in txt:
            score += 1
    for w in negative_words:
        if w in txt:
            score -= 1

    if score > 0:
        return "positive", 1.0
    elif score < 0:
        return "negative", -1.0
    return "neutral", 0.0


def add_timeline_event(title: str, date: str, details: str):
    """Save timeline event into session_state."""
    st.session_state.timeline.append({
        "title": title,
        "date": date,
        "details": details
    })


def openai_chat_reply(messages: list, api_key: str):
    """
    Calls GPT using whichever OpenAI API version is available.
    Compatible with openai <1.0 and >=1.0
    """
    try:
        # Try new API
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return resp.choices[0].message.content
        except Exception:
            # Fallback to old API
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                api_key=api_key
            )
            return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"(OpenAI error: {e})"


# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="EchoSoul", layout="wide")

st.sidebar.title("EchoSoul")
st.sidebar.caption("Adaptive personal companion — chat, call, remember.")

mode = st.sidebar.radio("Mode", ["Chat", "Chat history", "Life timeline", "Vault", "Export", "Brain mimic", "Call", "About"])

st.sidebar.markdown("### Settings")
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

# -----------------------------
# Modes
# -----------------------------
if mode == "Chat":
    st.title("💬 Chat with EchoSoul")

    user_input = st.text_area("Message", key="chat_input")

    if st.button("Send", key="send_btn"):
        if user_input.strip():
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Analyze emotion
            mood, score = analyze_emotion(user_input)

            # Prepare messages for GPT
            messages = [{"role": "system", "content": "You are EchoSoul, a supportive and adaptive companion."}]
            for msg in st.session_state.chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # GPT response if API key provided
            if OPENAI_API_KEY:
                reply = openai_chat_reply(messages, OPENAI_API_KEY)
            else:
                reply = f"I heard you. You seem {mood} (score={score})."

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

            st.rerun()

    # Display conversation
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**user:** {msg['content']}")
        else:
            st.markdown(f"**assistant:** {msg['content']}")

elif mode == "Chat history":
    st.title("📜 Chat History")
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            st.write(f"{msg['role']}: {msg['content']}")
    else:
        st.info("No chat history yet.")

elif mode == "Life timeline":
    st.title("🗓️ Life Timeline")

    with st.form("timeline_form"):
        ev_title = st.text_input("Event Title")
        ev_date = st.date_input("Date", datetime.date.today())
        ev_details = st.text_area("Details")
        submit = st.form_submit_button("Save Event")

    if submit:
        add_timeline_event(ev_title, ev_date.isoformat(), ev_details)
        st.success("Event added to timeline")
        st.rerun()

    if st.session_state.timeline:
        st.write("### Your Timeline")
        for ev in st.session_state.timeline:
            st.markdown(f"- **{ev['date']}**: {ev['title']} — {ev['details']}")
    else:
        st.info("No events yet.")

elif mode == "Vault":
    st.title("🔐 Memory Vault")
    if st.session_state.memory:
        st.json(st.session_state.memory)
    else:
        st.info("Memory is empty.")

elif mode == "Export":
    st.title("📤 Export Data")
    data = {
        "chat_history": st.session_state.chat_history,
        "timeline": st.session_state.timeline,
        "memory": st.session_state.memory,
    }
    st.download_button("Download JSON", data=json.dumps(data, indent=2), file_name="echosoul_data.json")

elif mode == "Brain mimic":
    st.title("🧠 Brain Mimic")
    st.info("This feature will simulate your tone and speaking style in future updates.")

elif mode == "Call":
    st.title("📞 Call (Prototype)")
    st.info("VoIP calling (Agora) integration coming soon!")

elif mode == "About":
    st.title("ℹ️ About EchoSoul")
    st.write("EchoSoul is your adaptive companion: remembers, reflects, and grows with you.")
