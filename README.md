$readmeContent = @"
# 🩺 VinX Medical Scribe AI

A powerful Python application that uses artificial intelligence to convert doctor-patient conversations into structured medical notes automatically. Built with cutting-edge AI technology to reduce administrative burden in healthcare.

## ✨ Features

- 🎤 Voice Recording: Capture patient consultations with microphone input
- 🗣️ Speech-to-Text: Accurate audio transcription using OpenAI's Whisper model
- ⚕️ Medical Note Generation: AI-powered SOAP note creation with Groq's Llama model
- 📝 Structured Output: Professional medical documentation format
- 💾 Export Capabilities: Save notes as text files
- 🔒 Privacy-First: All processing happens locally (optional cloud AI)


## 🚀 How It Works
- Record → Speak a sample doctor-patient conversation
- Transcribe → AI converts speech to text using Whisper
- Generate → AI structures the conversation into a medical note
- Save → Export the formatted SOAP note

## 🚀 Quick Start

```bash
# Install dependencies
pip install groq openai-whisper sounddevice soundfile python-dotenv torch torchaudio

# Run the application
python main.py