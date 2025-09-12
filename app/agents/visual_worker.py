# In app/agents/visual_worker.py

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Any

def analyze_visual_presentation(video_file_path: str) -> Dict[str, Any]:
    """
    Analyzes visual presentation with a definitive, tiered, graceful degradation approach for all metrics.
    """
    print(f"Visual Worker: Starting final universal tiered analysis for {video_file_path}...")
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands

    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    # Frame counters
    total_frames, face_detected_frames, hands_detected_frames, iris_landmarks_detected_count = 0, 0, 0, 0
    
    # Metric counters
    gaze_forward_frames, head_pose_engaged_frames, smiling_frames, gesturing_frames, motion_detected_frames = 0, 0, 0, 0, 0
    
    image_h, image_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    prev_gray_frame = None # For motion detection fallback

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        total_frames += 1
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        face_results = face_mesh.process(image_rgb)
        hand_results = hands.process(image_rgb)

        if face_results.multi_face_landmarks:
            face_detected_frames += 1
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Tier 1 Gaze: Iris Tracking
                if len(landmarks) > 473:
                    iris_landmarks_detected_count += 1
                    left_pupil = landmarks[473]
                    left_eye_l_corner, left_eye_r_corner = landmarks[33], landmarks[133]
                    left_eye_width = abs(left_eye_r_corner.x - left_eye_l_corner.x)
                    left_pupil_relative = (left_pupil.x - left_eye_l_corner.x) / left_eye_width if left_eye_width > 0 else 0.5
                    if 0.3 < left_pupil_relative < 0.7:
                        gaze_forward_frames += 1

                # Tier 2 Gaze: Head Pose Fallback
                nose_tip = landmarks[1]
                if 0.25 < nose_tip.x < 0.75 and 0.20 < nose_tip.y < 0.80:
                    head_pose_engaged_frames += 1

                # Smile Detection
                mouth_width = abs(landmarks[291].x - landmarks[61].x)
                mouth_height = abs(landmarks[17].y - landmarks[0].y)
                aspect_ratio = mouth_width / mouth_height if mouth_height > 0.01 else 0
                if aspect_ratio > 4.5:
                    smiling_frames += 1

        # Tier 1 Gestures: MediaPipe Hands
        if hand_results.multi_hand_landmarks:
            hands_detected_frames += 1
            # Simple check for gesturing
            gesturing_frames +=1
        
        # Tier 2: ROI Motion Detection Fallback
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if prev_gray_frame is not None:
            # Define Regions of Interest (left and right thirds of the lower half of the frame)
            roi_height = int(image_h / 2)
            roi_width = int(image_w / 3)
            
            left_roi = gray_frame[roi_height:, :roi_width]
            right_roi = gray_frame[roi_height:, -roi_width:]
            
            prev_left_roi = prev_gray_frame[roi_height:, :roi_width]
            prev_right_roi = prev_gray_frame[roi_height:, -roi_width:]
            
            # Calculate motion only within ROIs
            left_diff = cv2.absdiff(prev_left_roi, left_roi)
            right_diff = cv2.absdiff(prev_right_roi, right_roi)
            
            _, left_thresh = cv2.threshold(left_diff, 30, 255, cv2.THRESH_BINARY)
            _, right_thresh = cv2.threshold(right_diff, 30, 255, cv2.THRESH_BINARY)

            motion = np.sum(left_thresh) + np.sum(right_thresh)
            if motion > (left_roi.size + right_roi.size) * 0.01: # If >1% of pixels in ROIs changed
                motion_detected_frames += 1
        prev_gray_frame = gray_frame
                
    cap.release()
    face_mesh.close()
    hands.close()

    # --- Final Metric Calculation & Reporting ---
    face_detection_percent = round((face_detected_frames / total_frames) * 100) if total_frames > 0 else 0
    
    # Engagement Metric
    if total_frames > 0 and (iris_landmarks_detected_count / total_frames) > 0.5:
        engagement = {"method": "Gaze (Tier 1)", "percent": round((gaze_forward_frames / face_detected_frames) * 100) if face_detected_frames > 0 else 0}
    elif face_detected_frames > 0:
        engagement = {"method": "Head Pose (Tier 2)", "percent": round((head_pose_engaged_frames / face_detected_frames) * 100)}
    else:
        engagement = {"method": "N/A", "percent": "N/A - Face not detected"}

    # Smile Metric
    smile_percent = round((smiling_frames / face_detected_frames) * 100) if face_detected_frames > 0 else "N/A - Face not detected"

    # Gesture Metric
    if hands_detected_frames / total_frames > 0.1: # If hands model is reliable
        gestures = {"method": "Hand Tracking (Tier 1)", "percent": round((gesturing_frames / total_frames) * 100)}
    else: # Fallback to general motion
        gestures = {"method": "Motion Detection (Tier 2)", "percent": round((motion_detected_frames / total_frames) * 100)}

    print("Visual Worker: Final universal tiered analysis complete.")
    return {
        "analysis_quality": {"face_detection_percent": face_detection_percent},
        "metrics": {
            "engagement": engagement,
            "smile_percent": smile_percent,
            "gestures": gestures
        }
    }