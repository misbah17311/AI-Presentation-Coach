# In app/agents/vocal_worker.py

import parselmouth
from parselmouth.praat import call
import numpy as np
from typing import Dict, Any
from collections import Counter

def analyze_vocal_delivery(audio_file_path: str, transcript: str) -> Dict[str, Any]:
    """
    Analyzes the vocal delivery of an audio file using its transcript.
    Returns a dictionary of vocal metrics.
    """
    print(f"Vocal Worker: Starting enhanced analysis for {audio_file_path}...")
    try:
        sound = parselmouth.Sound(audio_file_path)
        
        pace = _calculate_speaking_pace(sound, transcript)
        filler_counts = _count_filler_words(transcript)
        pitch_variability = _analyze_pitch_variability(sound)
        pause_metrics = _analyze_pauses(sound)
        repetition_metrics = _analyze_repetitions(transcript) # <-- NEW FUNCTION CALL
        
        print("Vocal Worker: Enhanced analysis complete.")
        
        # Combine all metrics into one dictionary
        all_metrics = {
            "speaking_pace_wpm": pace,
            "filler_word_counts": filler_counts,
            "pitch_variability_st": pitch_variability,
        }
        all_metrics.update(pause_metrics)
        all_metrics.update(repetition_metrics) # <-- ADD THE NEW METRICS
        return all_metrics

    except Exception as e:
        print(f"Vocal Worker: ERROR during analysis - {e}")
        return {"error": f"Could not analyze audio file: {e}"}

def _calculate_speaking_pace(sound: parselmouth.Sound, transcript: str) -> float:
    word_count = len(transcript.split())
    duration_minutes = sound.get_total_duration() / 60
    return round(word_count / duration_minutes) if duration_minutes > 0 else 0

def _count_filler_words(transcript: str) -> Dict[str, int]:
    FILLER_WORDS = ["um", "uh", "er", "ah", "like", "okay", "right", "so", "you know"]
    words = transcript.lower().split()
    counts = {filler: words.count(filler) for filler in FILLER_WORDS}
    return {k: v for k, v in counts.items() if v > 0}

def _analyze_pitch_variability(sound: parselmouth.Sound) -> float:
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    voiced_pitch_values = pitch_values[pitch_values != 0]
    if len(voiced_pitch_values) < 2:
        return 0.0
    semitones = 12 * np.log2(voiced_pitch_values / 100)
    return round(np.std(semitones), 2)


def _analyze_pauses(sound: parselmouth.Sound) -> Dict[str, Any]:
    """
    Analyzes pauses using a direct, robust, intensity-based method in Python.
    """
    intensity = sound.to_intensity()
    # Use the 15th percentile of intensity as the silence threshold.
    # This is more robust than using the maximum intensity.
    silence_threshold = call(intensity, "Get quantile", 0.0, 0.0, 0.15)
    
    # Get the raw intensity values and their corresponding timestamps
    intensity_values = intensity.values[0]  # Intensity values in dB
    times = intensity.xs()                  # Timestamps for each value

    in_silence = False
    silence_start_time = 0
    pauses = []

    for i, time in enumerate(times):
        is_currently_silent = intensity_values[i] < silence_threshold
        
        if is_currently_silent and not in_silence:
            # We have just entered a silent interval
            in_silence = True
            silence_start_time = time
        elif not is_currently_silent and in_silence:
            # We have just exited a silent interval
            in_silence = False
            duration = time - silence_start_time
            if duration > 0.3:  # Our 300ms threshold for a meaningful pause
                pauses.append(duration)
    
    pause_count = len(pauses)
    avg_pause_duration = sum(pauses) / pause_count if pause_count > 0 else 0
    
    return {
        "pause_count": pause_count,
        "avg_pause_duration_s": round(avg_pause_duration, 2)
    }



def _analyze_repetitions(transcript: str) -> Dict[str, Any]:
    """Analyzes the transcript for repeated words."""
    words = transcript.lower().split()
    repeated_words = [word for word, count in Counter(words).items() if count > 2]
    
    # Find immediate repetitions (e.g., "the the")
    immediate_repetitions = 0
    for i in range(len(words) - 1):
        if words[i] == words[i+1]:
            immediate_repetitions += 1
            
    return {
        "repetition_count": immediate_repetitions,
        "frequently_repeated_words": repeated_words
    }