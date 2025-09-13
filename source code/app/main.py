# In app/main.py
import os
import tempfile
# Use temporary writable directoriess
from fastapi import FastAPI, File, UploadFile
from .agents.orchestrator import run_analysis_pipeline

app = FastAPI(
    title="AI Presentation Coach Agent",
    description="An agent for analyzing and providing feedback on communication skills.",
    version="1.0.0",
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Presentation Coach Agent is online."}

@app.post("/run")
async def run_endpoint(file: UploadFile = File(...)):
    """
    Accepts a video file upload and returns a full analysis report.
    """
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        temp_video_path = tmp.name

    print(f"API: Received file '{file.filename}', saved to '{temp_video_path}'")

    # Call the orchestrator with the path to the temporary file
    analysis_result = await run_analysis_pipeline(video_file_path=temp_video_path)

    # Clean up the temporary file
    os.remove(temp_video_path)
    print(f"API: Cleaned up temporary file '{temp_video_path}'")

    return {
        "intermediate_steps": analysis_result.get("intermediate_steps"),
        "final_report": analysis_result.get("final_report"),
        "is_intermediate": False,
        "complete": True
    }