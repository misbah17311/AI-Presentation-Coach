# In ui.py - FINAL DEFINITIVE VERSION
import streamlit as st
import requests
import json
import numpy as np # Import numpy to handle numpy-specific data types

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Presentation Coach",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Backend API Endpoint ---
BACKEND_URL = "http://127.0.0.1:8000/run"

# --- Helper Function for Safe Data Extraction ---
def safe_metric_display(metric_dict, keys, label, unit=""):
    value = metric_dict
    try:
        for key in keys:
            value = value[key]
        # Handle N/A strings
        if isinstance(value, str) and "N/A" in value:
             st.metric(label, value)
        else:
            st.metric(label, f"{value}{unit}")
    except (KeyError, TypeError):
        st.metric(label, "N/A")

# --- UI Layout ---
st.title("🤖 AI Presentation Coach")
st.markdown(
    "Upload a video of your solo performance (presentation, interview practice, monologue). "
    "Our AI agent will provide a detailed, data-driven analysis of your communication skills."
)

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Analyze Performance", use_container_width=True, type="primary"):
        with st.spinner("The agent is analyzing your performance... This may take several minutes."):
            try:
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(BACKEND_URL, files=files, timeout=600)
                
                if response.status_code == 200:
                    st.success("Analysis Complete!")
                    st.session_state['results'] = response.json()
                else:
                    st.error(f"Error from server: {response.status_code} - {response.text}")
                    st.session_state['results'] = None
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend analysis server. Please ensure it is running. Error: {e}")
                st.session_state['results'] = None

# --- Display Results ---
if 'results' in st.session_state and st.session_state['results'] is not None:
    results = st.session_state['results']
    final_report = results.get("final_report", "No report generated.")
    intermediate_steps = results.get("intermediate_steps", "")

    # --- Robust Parsing of Metrics ---
    try:
        def parse_json_with_numpy(json_string):
            return json.loads(json_string.replace("'", "\"").replace("np.float64(", "").replace(")", ""))
        
        transcript = intermediate_steps.split("--- TRANSCRIPT ---")[1].split("--- VOCAL METRICS ---")[0].strip()
        vocal_metrics_str = intermediate_steps.split("--- VOCAL METRICS ---")[1].split("--- VISUAL METRICS ---")[0].strip()
        visual_metrics_str = intermediate_steps.split("--- VISUAL METRICS ---")[1].split("--- CONTENT METRICS ---")[0].strip()
        content_metrics_str = intermediate_steps.split("--- CONTENT METRICS ---")[1].strip()

        vocal_metrics = parse_json_with_numpy(vocal_metrics_str)
        visual_metrics = parse_json_with_numpy(visual_metrics_str)
        content_metrics = parse_json_with_numpy(content_metrics_str)

    except (IndexError, json.JSONDecodeError) as e:
        transcript, vocal_metrics, visual_metrics, content_metrics = "Could not parse data.", {}, {}, {}
        st.error(f"Error parsing analysis data: {e}")

    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📊 Synthesized Report", "📈 Detailed Metrics", "📝 Transcript"])

    with tab1:
        st.subheader("Synthesized Feedback Report")
        st.markdown(final_report)

    with tab2:
        st.subheader("Detailed Metric Analysis")
        
        # --- Vocal Metrics Display (Corrected and Complete) ---
        st.markdown("#### Vocal Delivery")
        col1, col2, col3, col4 = st.columns(4)
        with col1: safe_metric_display(vocal_metrics, ['speaking_pace_wpm'], "Speaking Pace", " WPM")
        with col2: safe_metric_display(vocal_metrics, ['pitch_variability_st'], "Pitch Variability", " ST")
        with col3: safe_metric_display(vocal_metrics, ['pause_count'], "Pause Count")
        with col4: safe_metric_display(vocal_metrics, ['avg_pause_duration_s'], "Avg. Pause", " s")

        col1a, col2a = st.columns(2)
        with col1a: 
            filler_count = sum(vocal_metrics.get('filler_word_counts', {}).values())
            st.metric("Filler Word Count", filler_count)
        with col2a:
            repetition_count = vocal_metrics.get('repetition_count', 0)
            st.metric("Repetition Count", repetition_count)
        
        repeated_words = vocal_metrics.get('frequently_repeated_words', [])
        if repeated_words:
            st.markdown("**Frequently Repeated Words:**")
            st.write(", ".join(f"`{word}`" for word in repeated_words))
        
        # --- Visual Metrics Display (Corrected and Complete) ---
        st.markdown("#### Visual Presence")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Detection Quality**")
            safe_metric_display(visual_metrics, ['analysis_quality', 'face_detection_percent'], "Face Detection", "%")
            safe_metric_display(visual_metrics, ['analysis_quality', 'hands_detection_percent'], "Hands Detection", "%")
        
        with col2:
            st.markdown("**Engagement Metrics**")
            engagement_data = visual_metrics.get('metrics', {}).get('engagement', {})
            st.metric(f"Engagement ({engagement_data.get('method', 'N/A')})", f"{engagement_data.get('percent', 'N/A')}")
            safe_metric_display(visual_metrics, ['metrics', 'smile_percent'], "Smile Presence", "%")
            gesture_data = visual_metrics.get('metrics', {}).get('gestures', {})
            st.metric(f"Gestures ({gesture_data.get('method', 'N/A')})", f"{gesture_data.get('percent', 'N/A')}")
        
        # --- Content Metrics Display (Corrected and Complete) ---
        st.markdown("#### Content & Clarity")
        safe_metric_display(content_metrics, ['readability_score_flesch'], "Readability (Flesch)%")
        
        keywords = content_metrics.get('keywords', [])
        themes = content_metrics.get('main_themes', [])
        
        if keywords:
            st.markdown("**Keywords Detected:**")
            st.write(", ".join(f"`{kw}`" for kw in keywords))
        if themes:
            st.markdown("**Main Themes Detected:**")
            st.write(", ".join(f"`{th}`" for th in themes))

    with tab3:
        st.subheader("Full Transcript")
        st.markdown(transcript)