import os
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import whisper
import sounddevice as sd
import io
import wave

# Load the API key from the .env file
load_dotenv()
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def record_audio(duration=5, sample_rate=16000):
    """Records audio from microphone and returns audio data."""
    print(f"\n🎙️  Recording for {duration} seconds... Speak now!")
    
    try:
        # Record audio
        recording = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("✅ Recording complete.")
        return recording.flatten(), sample_rate
        
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None, None

def transcribe_audio_directly(audio_data, sample_rate):
    """Transcribes audio data directly without saving to file."""
    print("📝 Transcribing with local Whisper model...")
    
    try:
        # Load the Whisper model
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Model loaded successfully!")
        
        # Transcribe the audio data directly
        print("Transcribing audio...")
        
        # Convert audio data to the format Whisper expects
        audio_data = audio_data.astype(np.float32)
        
        # Transcribe directly from numpy array
        result = model.transcribe(audio_data, fp16=False)
        transcription = result["text"]
        
        print("✅ Local Whisper transcription complete!")
        return transcription
        
    except Exception as e:
        print(f"❌ Whisper transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_transcription():
    """Gets transcription with multiple options."""
    print("\n📝 How would you like to provide the patient dialogue?")
    print("1. 🎤 Record audio (local Whisper - accurate & free)")
    print("2. 📝 Type manually")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Record audio and transcribe with local Whisper
        audio_data, sample_rate = record_audio(duration=7)
        
        if audio_data is not None:
            print("🔄 Starting transcription...")
            transcription = transcribe_audio_directly(audio_data, sample_rate)
            if transcription:
                return transcription
            else:
                print("❌ Transcription failed.")
        else:
            print("❌ Audio recording failed.")
        
        print("Please type the dialogue manually:")
        return input("> ")
    
    elif choice == "2":
        print("\nPlease type the patient's dialogue:")
        return input("> ")
    
    else:
        print("Invalid choice. Please type manually:")
        return input("> ")

def generate_soap_note(transcript):
    """Uses Groq's API to generate a SOAP note from the transcription."""
    print("\n⚕️  Generating SOAP note with Groq...")

    system_prompt = """
    You are a medical scribe AI. Convert a transcript of a patient conversation into a structured medical note.
    Focus on the 'Subjective' section of a SOAP note. Extract:
    - Chief Complaint (CC)
    - History of Present Illness (HPI)
    - Relevant Symptoms
    Use professional medical language. If details are missing, do not make them up. Use placeholders like [Patient Name].
    """

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Convert this conversation:\n\n{transcript}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1024
        )
        soap_note = chat_completion.choices[0].message.content
        print("✅ SOAP note generated.")
        return soap_note
    except Exception as e:
        return f"[Error generating note: {e}]"

def main():
    output_filename = "medical_note.txt"

    try:
        print("Medical Scribe AI - Voice to SOAP Note Converter")
        print("=" * 50)
        print("Using local Whisper for accurate, free transcription!")
        
        # 1. Get transcription
        transcription = get_transcription()
        if transcription:
            print(f"\n🎯 Sharp Transcription:\n{'-'*30}\n{transcription}\n")

            # 2. Generate Note
            soap_note = generate_soap_note(transcription)

            # 3. Display and Save
            print(f"\n📋 Final Medical Note:\n{'-'*30}\n{soap_note}")
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(soap_note)
            print(f"\n💾 Note saved to '{output_filename}'")
        else:
            print("❌ No transcription available.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()