# In app/agents/content_worker.py

import textstat
import yake
from transformers import pipeline
from typing import Dict, Any, List

# --- Initialize models and tools once on startup ---

# Yake for fast keyword extraction
kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=5)
print("Content Worker: YAKE keyword extractor initialized.")

# Transformers pipeline for sophisticated topic analysis.
# This will download a model on the first run.
print("Content Worker: Loading zero-shot classification model...")
try:
    topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Content Worker: Zero-shot model loaded.")
    CLASSIFIER_LOADED = True
except Exception as e:
    print(f"Content Worker: FAILED to load zero-shot model. Error: {e}")
    topic_classifier = None
    CLASSIFIER_LOADED = False

def analyze_content(transcript: str) -> Dict[str, Any]:
    """
    Performs a definitive, multi-faceted analysis of the presentation transcript.
    """
    print("Content Worker: Starting definitive content analysis...")
    if not transcript or not transcript.strip():
        return {"error": "Transcript is empty, cannot analyze content."}

    try:
        # 1. Readability Analysis (fast, statistical)
        readability_score = textstat.flesch_reading_ease(transcript)
        
        # 2. Keyword Extraction (fast, statistical)
        keywords_raw = kw_extractor.extract_keywords(transcript)
        keywords = [kw for kw, score in keywords_raw]

        # 3. Topic Analysis (powerful, model-based)
        main_themes = []
        if CLASSIFIER_LOADED:
            # We define candidate labels relevant to a presentation/pitch
            candidate_labels = ["introduction", "technical details", "project goals", "business impact", "conclusion", "personal story"]
            # We analyze the transcript against these labels.
            topic_results = topic_classifier(transcript, candidate_labels, multi_label=False)
            main_themes = topic_results['labels'][:3] # Report the top 3 detected themes

        print("Content Worker: Definitive analysis complete.")
        return {
            "readability_score_flesch": readability_score,
            "keywords": keywords,
            "main_themes": main_themes
        }
    except Exception as e:
        print(f"Content Worker: ERROR during analysis - {e}")
        return {"error": f"Could not analyze content: {e}"}