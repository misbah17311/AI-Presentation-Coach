# In app/utils/audio_extractor.py
import subprocess
import tempfile
import os

def extract_audio_from_video(video_path: str) -> str:
    """
    Extracts audio from a video file and saves it as a temporary WAV file.
    Returns the path to the temporary WAV file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")

    # Create a temporary file to store the WAV audio
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()

    print(f"Audio Extractor: Created temporary WAV file at {temp_wav_path}")

    # Command to extract audio using ffmpeg
    command = [
        "ffmpeg",
        "-i", video_path,    # Input video file
        "-vn",               # No video output
        "-acodec", "pcm_s16le", # Audio codec for WAV
        "-ar", "16000",      # Sample rate (16kHz is standard for Whisper)
        "-ac", "1",          # Mono channel
        temp_wav_path,
        "-y"                 # Overwrite output file if it exists
    ]

    try:
        print("Audio Extractor: Running ffmpeg command...")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Audio Extractor: ffmpeg command successful.")
        return temp_wav_path
    except subprocess.CalledProcessError as e:
        print(f"Audio Extractor: ERROR running ffmpeg. Stderr: {e.stderr}")
        # Clean up the failed temp file
        os.remove(temp_wav_path)
        raise RuntimeError(f"ffmpeg failed to extract audio: {e.stderr}") from e