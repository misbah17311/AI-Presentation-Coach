# 🤖 AI Presentation Coach

**Author:** MD Misbah Ur Rahman
**University:** Indian Institute of Technology Kharagpur
**Department:** Chemical Engineering

---

## 1. Project Overview

This repository contains the complete source code and documentation for the **AI Presentation Coach**, an advanced prototype of an AI agent designed to automate the analysis of solo on-camera spoken performances. It provides users with data-driven, actionable feedback on their presentation, public speaking, or interview practice sessions to help them improve their communication skills.

This project was developed for a Data Science internship assignment from I'm Beside You Inc.

### Live Demo

A live, interactive version of this agent has been deployed and is publicly accessible. *(Note: As we decided, you will replace this with a link to your demo video once it is recorded and uploaded to a platform like YouTube or Google Drive.)*

**[Link to Live Demo / Demo Video Here]**

## 2. Deliverables

This repository is structured to provide all required deliverables for the assignment.

* **Source Code:** The complete source code for the agent is located in the `app/` directory, with the Streamlit UI in `ui.py`.
* **AI Agent Architecture Document:** A detailed breakdown of the system's multi-agent architecture, components, and interaction flows. [**Link:** `ARCHITECTURE.md`]
* **Data Science Report:** A comprehensive report detailing the fine-tuning of the synthesis agent, including the dataset creation, training methodology, and evaluation protocol. [**Link:** `DATA_SCIENCE_REPORT.md`]
* **Interaction Logs:** The complete chat history detailing the development process of this agent with the AI assistant. [**Link:** `INTERACTION_LOGS.md`]
* **Demo Video:** A screen recording showcasing the live application in use. *(Link to be added)*

## 3. Technical Architecture

The agent is built on a sophisticated **Orchestrator-Workers** architecture. Upon receiving a user's video, a central orchestrator plans and delegates analytical tasks to a suite of specialized, independent workers, including:

* **Transcription-Worker** (OpenAI Whisper)
* **Vocal-Worker** (Parselmouth/Praat)
* **Visual-Worker** (Google MediaPipe)
* **Content-Worker** (Hugging Face Transformers)
* **Synthesis-Worker** (A fine-tuned `gemma-2b-it` model)

This multi-modal, multi-agent approach allows for a deep and comprehensive analysis of a user's communication effectiveness. For a complete breakdown, please see the full **Architecture Document**.

## 4. Running the Project Locally

To run this application on a local machine with a compatible NVIDIA GPU, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ai-presentation-coach.git](https://github.com/your-username/ai-presentation-coach.git)
    cd ai-presentation-coach
    ```
2.  **Set up the environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Authenticate with Hugging Face:**
    ```bash
    huggingface-cli login
    # Paste your HF token when prompted
    ```
4.  **Launch the servers:**
    * In one terminal, start the backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
    * In a second terminal, start the frontend: `streamlit run ui.py`
5.  Open the local URL provided by Streamlit in your browser.
