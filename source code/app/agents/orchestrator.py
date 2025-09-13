# In app/agents/orchestrator.py

import os
from typing import Dict, Any
from .transcription_worker import transcribe_audio
from .vocal_worker import analyze_vocal_delivery
from .visual_worker import analyze_visual_presentation # Import the new worker
from .content_worker import analyze_content # Import the new worker
from .synthesis_worker import generate_feedback_report # Import the final worker
from ..utils.audio_extractor import extract_audio_from_video

async def run_analysis_pipeline(video_file_path: str) -> Dict[str, Any]:
    """
    Orchestrator now with a final Synthesis stage.
    """
    print(f"ORCHESTRATOR: Starting analysis for '{video_file_path}'")
    temp_audio_path = None
    try:
        # --- STAGE 1: DATA GATHERING ---
        temp_audio_path = extract_audio_from_video(video_file_path)
        transcript = transcribe_audio(audio_file_path=temp_audio_path)
        vocal_metrics = analyze_vocal_delivery(
            audio_file_path=temp_audio_path, transcript=transcript
        )
        visual_metrics = analyze_visual_presentation(video_file_path=video_file_path)
        content_metrics = analyze_content(transcript=transcript) # New worker call
        # Consolidate all metrics for the synthesis worker
        all_metrics = {
            "transcript": transcript,
            "vocal_metrics": vocal_metrics,
            "visual_metrics": visual_metrics,
            "content_metrics": content_metrics
        }
        # --- STAGE 2: SYNTHESIS ---
        final_report = generate_feedback_report(all_metrics)
        # --- STAGE 3: REPORTING(all the steps are now completed) ---
        plan = [
            f"1. Extract audio from video. [COMPLETED]",
            f"2. Transcribe audio. [COMPLETED]",
            f"3. Analyze vocal delivery. [COMPLETED]",
            f"4. Analyze visual presentation. [COMPLETED]",
            f"5. Analyze content from transcript. [COMPLETED]",
            f"6. Synthesize final feedback report. [COMPLETED]"
            
        ]

        intermediate_steps = "Generated Plan:\n" + "\n".join(plan)
        intermediate_steps += f"\n\n--- TRANSCRIPT ---\n{transcript}"
        intermediate_steps += f"\n\n--- VOCAL METRICS ---\n{str(vocal_metrics)}"
        intermediate_steps += f"\n\n--- VISUAL METRICS ---\n{str(visual_metrics)}"
        intermediate_steps += f"\n\n--- CONTENT METRICS ---\n{str(content_metrics)}"
        return {
            "intermediate_steps": intermediate_steps,
            "final_report": final_report
        }
    except Exception as e:
        # ... (error handling remains the same)
        print(f"ORCHESTRATOR: Error occurred - {e}")
        return {
            "intermediate_steps": "Error during analysis pipeline.",
            "final_report": f"An error occurred: {e}"
        }
    finally:
        # --- CLEANUP ---
        if temp_audio_path and os.path.exists(temp_audio_path):
            print(f"ORCHESTRATOR: Cleaning up temporary audio file: {temp_audio_path}")
            os.remove(temp_audio_path)