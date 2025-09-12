# In app/agents/transcription_worker.py

import whisper
import torch
# This is a one-time setup when the server starts.
# It determines if a GPU is available and loads the model into memory.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Transcription Worker: Using device: {DEVICE}")

print("Transcription Worker: Loading Whisper model (small.en)...")
model = whisper.load_model("small.en", device=DEVICE)
print("Transcription Worker: Whisper model loaded successfully.")

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes the given audio file using the pre-loaded Whisper model.
    """
    try:
        print(f"Transcription Worker: Starting transcription for {audio_file_path}...")
        # fp16=False is recommended for CPU-only execution
        result = model.transcribe(audio_file_path, fp16=torch.cuda.is_available())
        transcribed_text = result.get("text", "")
        print("Transcription Worker: Transcription complete.")
        return transcribed_text
    except Exception as e:
        print(f"Transcription Worker: ERROR during transcription - {e}")
        return f"Error: Could not transcribe audio file at {audio_file_path}."