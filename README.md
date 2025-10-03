# EchoSoul 🪷

EchoSoul is an evolving AI companion app built with **Streamlit**.  
It remembers you, adapts to you, and even mimics your thought process.  

---

## 🌟 Features
- **Persistent Memory** – Keeps track of conversations across sessions  
- **Adaptive Personality** – Learns and reflects your communication style  
- **Emotion Recognition (basic text-based)**  
- **Life Timeline** – Chronological record of your journey with EchoSoul  
- **Private Vault** – Password-protected secure memory storage  
- **Brain Mimic** – EchoSoul replies like you would  
- **Export Data** – Download all chats, vault entries, and timeline events  
- **Live Voice Call (Beta)** – Talk to EchoSoul via audio (WebRTC powered)  

---

## 🚀 Deployment
### Streamlit Cloud
1. Upload your repo to GitHub.  
2. Go to [share.streamlit.io](https://share.streamlit.io).  
3. Enter:
   - **Repository**: `<your-username>/<your-repo>`  
   - **Branch**: `main`  
   - **Main file path**: `app.py`  

### Hugging Face Spaces
1. Create a new Space → Select **Streamlit** SDK.  
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `apt.txt`
   - `README.md`  

---

## 🔑 API Key
You need an **OpenAI API key** to use EchoSoul.  
- Get it from: [OpenAI API Keys](https://platform.openai.com/account/api-keys)  
- In Streamlit Cloud or Hugging Face Spaces, set it as a **Secret**:  
  - Key: `OPENAI_API_KEY`  
  - Value: `your-key-here`  

---

## ⚠️ Known Issues
- If you see *quota exceeded* errors, check your [OpenAI Billing Dashboard](https://platform.openai.com/account/billing/overview).  
- Voice calling is experimental and may not run in all browsers without HTTPS or microphone permissions.  

---

## ❤️ Credits
Built with [Streamlit](https://streamlit.io), [OpenAI](https://openai.com), and [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc).
