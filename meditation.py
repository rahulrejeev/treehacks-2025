import json
import os
import time
import boto3
import cv2
import dlib
from google import genai
from dotenv import load_dotenv
import numpy as np
import streamlit as st
from imutils import face_utils
from helpers import eye_aspect_ratio, average_point
import logging
from elevenlabs.client import ElevenLabs
import base64
import re

logger = logging.Logger('logger')
logger.setLevel(logging.DEBUG)
# Create a stream handler (console output)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Create a formatter and attach it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to your logger
logger.addHandler(handler)
prompts_asked = []
visual_cues_history = []
telemetry_history = []

load_dotenv()

# Load environment variables
try:
    GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
    AWS_ACCESS_KEY=os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION=os.getenv('AWS_REGION')
    AWS_BUCKET_NAME=os.getenv('AWS_BUCKET_NAME')
    ELEVEN_LABS_API_KEY=os.getenv('ELEVEN_LABS_KEY')
    VOICE_ID=os.getenv('VOICE_ID')
    s3 = boto3.client('s3', 
                      aws_access_key_id=AWS_ACCESS_KEY, 
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                      region_name=AWS_REGION) 
    elevenlabs_client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

except Exception as e:
    raise e

# Config
gemini_client =  genai.Client(api_key=GOOGLE_API_KEY)
TIME_INTERVAL = 40  # seconds
ELEVEN_LABS_API = f"https://api.elevenlabs.ai/v1/text-to-speech/{VOICE_ID}"

# Video config
# Load dlib's face detector and landmark predictor.
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    st.error(f"Missing {predictor_path} file!")
    st.stop()
predictor = dlib.shape_predictor(predictor_path)

# --- Session State Initialization ---

def main_loop():
    questions = 0

    while not run_live:
        time.sleep(0.1)
    
    initialize()
    while questions < 5:
        curr = time.time()
        handle_audio()
        while time.time() - curr < TIME_INTERVAL:
            frame, rgb_frame = update_camera()
            render_frame(rgb_frame)
            main(frame, rgb_frame)
        update_scipt()
        questions += 1

    camera.release()
    FRAME_WINDOW.image(image="./landing.jpeg", use_container_width=True)
        
def extract_readable_script(text):
    """
    Given a text string in the format with bold section titles and minute markers,
    remove those formatting markers and return the clean script.
    """
    # Regex explanation:
    # \*\*.*?\*\*   -> Matches any text enclosed in ** (non-greedy)
    # \(Minute\s*\d+\s*-\s*\d+\) -> Matches minute markers like (Minute 1-3)
    # pattern = r'\*\*.*?\*\*|\(Minute\s*\d+\s*-\s*\d+\)'
    
    # # Remove the matched patterns from the text.
    # cleaned_text = re.sub(pattern, '', text)
    
    # # Optionally, remove extra blank lines or surrounding whitespace.
    # # This splits the text by lines, strips them, and rejoins non-empty lines.
    # cleaned_lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    # return "\n".join(cleaned_lines)
    return text

def initialize():
    st.session_state.metrics = {
            'total_frames': 0,
            'blink_count': 0,
            'closed_frames': 0,
            'tension_score_accum': 0,
            'tension_count': 0,
            'total_head_movement': 0,
            'prev_face_center': None,
            'prev_eyebrow_position': None,
            'micro_expression_events': 0
        }
    st.session_state.last_update = time.time()
    st.session_state.current_script = "Alright, let's begin. Find a comfortable position, either sitting or lying down, and gently close your eyes. Take a deep breath in, and as you exhale, let go of any tension you might be holding. Now, what do you notice right now? What sensations are present in your body, and what sounds do you hear around you? Just observe without judgment."
    prompts_asked.append(st.session_state.current_script)



# --- Custom CSS for styling ---
st.markdown("""
    <style>
    /* Neural Network Background Animation */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)),
                    url("data:image/svg+xml,%3Csvg width='100%' height='100%' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3Cpattern id='neural' x='0' y='0' width='50' height='50' patternUnits='userSpaceOnUse'%3E%3Ccircle cx='25' cy='25' r='1' fill='rgba(255,255,255,0.15)'/%3E%3C/pattern%3E%3C/defs%3E%3Crect width='100%' height='100%' fill='%23111'/%3E%3Crect width='100%' height='100%' fill='url(%23neural)'/%3E%3C/svg%3E");
        background-color: #111 !important;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 50% 50%, rgba(76, 175, 80, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 30% 30%, rgba(255, 107, 107, 0.05) 0%, transparent 40%);
        pointer-events: none;
        animation: pulse 8s ease-in-out infinite;
        z-index: 0;
    }

    /* Make content more visible */
    .stApp > header {
        background-color: transparent !important;
        z-index: 10;
    }

    .main .block-container {
        z-index: 10;
        position: relative;
    }

    .title-center {
        text-align: center;
        color: white;
        padding: 20px;
        font-size: 50px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        font-family: 'Times New Roman', Times, serif;
        position: relative;
        z-index: 2;
    }
    
    .meditation-text {
        text-align: center;
        font-size: 18px;
        color: white;
        line-height: 1.6;
        margin: 20px 0;
        font-family: 'Times New Roman', Times, serif;
        position: relative;
        z-index: 2;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        transition: all 0.3s ease;
        position: relative;
        z-index: 2;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stCheckbox>label {
        transition: all 0.3s ease;
        font-family: 'Times New Roman', Times, serif;
        color: white;
        position: relative;
        z-index: 2;
    }
    
    .stCheckbox>label:hover {
        color: #4CAF50;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* Video container styling */
    [data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        background: rgba(17, 17, 17, 0.7);
        backdrop-filter: blur(5px);
    }

    footer {
        color: white !important;
    }

    /* Override any Streamlit default text colors */
    .stMarkdown, .stText {
        color: white !important;
    }

    /* Ensure all headings are white */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    /* Style for checkbox label */
    .stCheckbox {
        color: white !important;
    }
    
    .stCheckbox > label > div[role="checkbox"] {
        color: white !important;
    }
    
    /* Style for checkbox text */
    .stCheckbox > label > div:last-child {
        color: white !important;
    }

    /* Set Times New Roman as default font */
    * {
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* Info Block Styling */
    .info-block {
        background: rgba(17, 17, 17, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(5px);
        width: 250px;
        height: 100%;
        transition: all 0.3s ease;  /* Smooth transition for hover effects */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* Initial shadow */
    }

    .info-block:hover {
        background: rgba(17, 17, 17, 0.8);  /* Slightly darker background on hover */
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.15);  /* Glowing white shadow */
        transform: translateY(-2px);  /* Slight lift effect */
    }

    .info-block-title {
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }

    .info-block-text {
        color: white;
        font-size: 16px;
        text-align: center;
        line-height: 1.5;
    }

    /* Container for description text */
    .description-container {
        width: 80%;
        margin: 0 auto;
        max-width: 800px;
    }

    .left-info {
        float: left;
        margin-left: -250px;  /* Bring it closer */
        margin-top: 135px;
    }

    .right-info {
        float: right;
        margin-right: -250px;  /* Bring it closer */
        margin-top: 135px;
    }
            
    /* 1. Animated Start Button */
    .stCheckbox > label {
        padding: 10px 15px;
        border-radius: 8px;
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }

            
    /* 3. Enhanced Floating Particles */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 25% 25%, white 1.5px, transparent 1.5px),
            radial-gradient(circle at 75% 75%, white 1.5px, transparent 1.5px),
            radial-gradient(circle at 50% 50%, rgba(255,255,255,0.8) 1px, transparent 1px);
        background-size: 50px 50px, 50px 50px, 30px 30px;
        opacity: 0.1;  /* Increased from 0.05 */
        animation: float 20s linear infinite;
        pointer-events: none;
        z-index: 1;
    }

    @keyframes float {
        0% { 
            transform: translateY(0) rotate(0deg); 
        }
        50% {
            transform: translateY(-25px) rotate(0.5deg);
        }
        100% { 
            transform: translateY(-50px) rotate(1deg); 
        }
    }
    
    /* Style the checkbox container */
    .stCheckbox {
        display: flex !important;
        justify-content: center !important;
        width: 100%;
    }

    /* Style the checkbox label container */
    .stCheckbox > label {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: auto !important;
        margin: 0 auto !important;
    }

    /* Style the checkbox text */
    .stCheckbox > label > div:last-child {
        text-align: center !important;
        margin-left: 8px !important;
    }            
    </style>
    """, unsafe_allow_html=True)

    # Title
st.markdown("<h1 class='title-center'>ॐ: Optimized Meditation</h1>", unsafe_allow_html=True)

# Description in container
st.markdown("""
    <div class='description-container meditation-text'>
        This application uses your webcam feed to analyze your visual cues in real time.
        Based on your facial expressions and movements, the system updates the guided meditation script dynamically.
    </div>
    """, unsafe_allow_html=True)

# Columns with balanced spacing
col1, col2, col3 = st.columns([0.4,3.2,0.4])  # Slightly wider side columns

# Left info block
with col1:
    st.markdown("""
        <div class='info-block left-info'>
            <div class='info-block-title'>Bio-Wearable Data</div>
            <div class='info-block-text'>
                Wearables detect physiological markers of meditation efficacy
            </div>
        </div>
    """, unsafe_allow_html=True)

# Center content
with col2:
    run_live = st.checkbox("✨ Start Live Feed")
    FRAME_WINDOW = st.image("./landing.jpeg", use_container_width=True)
    
# Right info block
with col3:
    st.markdown("""
        <div class='info-block right-info'>
            <div class='info-block-title'>VLM Analysis</div>
            <div class='info-block-text'>
                Uses vision-language models to assess meditation efficacy
            </div>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        Made with ❤️ for TreeHacks 2025
    </div>
""", unsafe_allow_html=True)
camera = cv2.VideoCapture(0)

# --- Main Live Feed Loop ---

def update_camera():
    if not camera.isOpened():
        st.error("Error: Could not open webcam.")
        st.stop()
    ret, frame = camera.read()
    if not ret: return
    # Convert frame to RGB (for display) and grayscale (for processing).
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame

def main(frame, rgb_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    m = st.session_state.metrics  # current metrics

    # Process each detected face.
    for face in faces:
        shape = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)
        # --- Eye Blink Detection ---
        rightEye = shape_np[36:42]
        leftEye = shape_np[42:48]
        rightEAR = eye_aspect_ratio(rightEye)
        leftEAR = eye_aspect_ratio(leftEye)
        ear = (rightEAR + leftEAR) / 2.0
        m['total_frames'] += 1
        EAR_THRESHOLD = 0.2
        if ear < EAR_THRESHOLD:
            m['closed_frames'] += 1
            # A simplistic blink count (in real-world use, detect transitions)
            m['blink_count'] += 1

        # Draw eye landmarks.
        for (x, y) in np.concatenate((rightEye, leftEye), axis=0):
            cv2.circle(rgb_frame, (x, y), 2, (0, 255, 0), -1)
        
        # --- Facial Tension via Eyebrow-Eye Distance ---
        left_eyebrow = shape_np[17:22]
        right_eyebrow = shape_np[22:27]
        left_eye_center = average_point(shape_np[36:42])
        right_eye_center = average_point(shape_np[42:48])
        left_distance = np.linalg.norm(average_point(left_eyebrow) - left_eye_center)
        right_distance = np.linalg.norm(average_point(right_eyebrow) - right_eye_center)
        avg_eyebrow_eye_distance = (left_distance + right_distance) / 2.0
        face_width = face.right() - face.left()
        normalized_distance = avg_eyebrow_eye_distance / face_width
        EYEBROW_EYE_NORM_THRESHOLD = 0.15
        tension = normalized_distance < EYEBROW_EYE_NORM_THRESHOLD
        if tension:
            m['tension_score_accum'] += 1
        m['tension_count'] += 1

        # --- Head Movement ---
        face_center = average_point(shape_np)
        if m['prev_face_center'] is not None:
            movement = np.linalg.norm(face_center - m['prev_face_center'])
            m['total_head_movement'] += movement
        m['prev_face_center'] = face_center

        # --- Micro-Expression Detection (Simplistic) ---
        current_eyebrow_position = average_point(np.concatenate((left_eyebrow, right_eyebrow), axis=0))
        if m['prev_eyebrow_position'] is not None:
            diff = np.linalg.norm(current_eyebrow_position - m['prev_eyebrow_position'])
            MICRO_EXPR_THRESHOLD = 3.0
            if diff > MICRO_EXPR_THRESHOLD:
                m['micro_expression_events'] += 1
        m['prev_eyebrow_position'] = current_eyebrow_position

def get_terra_from_s3():
    logger.info("Fetching Terra payloads from S3...")
    # Paginator to list all objects in the bucket.
    paginator = s3.get_paginator("list_objects_v2")
    try:
        page = [page for page in paginator.paginate(Bucket=AWS_BUCKET_NAME, MaxKeys=1)][0]
        json_lst = []
        keys_to_delete = [] 
        logger.info(f"Got paginator")

        # Iterate through all pages of objects.
        logger.info(type(page))
        logger.info(page)
        for obj in page["Contents"]:
            key = obj["Key"]
            # Process only .json files
            if key.endswith(".json"):
                try:
                    # Fetch object content.
                    response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=key)
                    content = response['Body'].read().decode('utf-8')
                    # Attempt to parse JSON.
                    json_data = json.loads(content)
                    json_lst.append(json_data)
                    logger.info(f"Fetched JSON from: {key}")
                except Exception as e:
                    logger.error(f"Error processing {key}: {e}")
                
                # Add key to deletion list.
                keys_to_delete.append(key)

        logger.info(f"Found {len(json_lst)} JSON payloads.")
        logger.info(f"Deleting {len(keys_to_delete)} objects...")

        for key in keys_to_delete:
            try:
                delete_response = s3.delete_object(Bucket=AWS_BUCKET_NAME, Key=key)
                logger.info(f"Deleted {key} with response: {delete_response}")
            except Exception as e:
                logger.error(f"Error deleting {key}: {e}")
        return json_lst[0]
    except Exception as e:
        logger.error(f"Error fetching Terra payloads: {e}")


def build_combined_prompt(terra_payload, vlm_payload):
    """
    Combines Terra biometric data and visual cues into a structured meditation prompt,
    with additional observations if the data suggests focus needs to be re-established.
    """
    prompt = (
        "Analyze the following biometric and visual data for an AI-guided meditation session.\n"
        "Assess relaxation and stress indicators based on physiological and behavioral cues.\n\n"
        
        "## Physiological Data\n"
        "**1. Heart Rate Variability (HRV):**\n"
        "- Effective: HRV > 65ms (Indicates relaxation, parasympathetic activity)\n"
        "- Adverse: HRV < 50ms (Indicates stress, sympathetic activation)\n\n"

        "**2. Respiratory Rate (RR):**\n"
        "- Effective: 4–6 breaths/min (Deep, rhythmic breathing)\n"
        "- Adverse: >12 breaths/min (Hyperventilation, stress response)\n\n"

        "**3. Electrodermal Activity (EDA):**\n"
        "- Effective: SCL < 5μS (Low emotional arousal)\n"
        "- Adverse: SCL spike > 1μS (Emotional trigger or stress response)\n\n"

        "**4. Blood Volume Pulse via PPG:**\n"
        "- Effective: Stable waveform, augmentation index < 10%\n"
        "- Adverse: Erratic dicrotic notch (Vascular stress, tension)\n\n"

        "**5. Body Temperature:**\n"
        "- Effective: ±0.5°C from baseline (Thermal stability)\n"
        "- Adverse: Drop >1°C (Stress-induced vasoconstriction)\n\n"

        "## Visual Data from Video\n"
        "**1. Facial Muscle Tension:**\n"
        "- Effective: Relaxed jaw, neutral lips, softened forehead\n"
        "- Adverse: Furrowed brows, lip tightening, asymmetrical nasolabial folds\n\n"

        "**2. Eye Movement & Blink Rate:**\n"
        "- Effective: Closed eyes, stable gaze (≤2° deviation)\n"
        "- Adverse: Frequent blinking (>20 blinks/min), rapid saccades\n\n"

        "**3. Head Movement & Postural Sway:**\n"
        "- Effective: Minimal head motion (≤1 cm displacement over 5 mins)\n"
        "- Adverse: Frequent adjustments, shoulder shrugging\n\n"

        "**4. Micro-Expression Recognition:**\n"
        "- Effective: Smooth, neutral expressions\n"
        "- Adverse: Sudden brow tension, lip pursing, nostril flaring\n\n"
    )

    prompt += "### Terra Biometric Data:\n" + json.dumps(terra_payload, indent=2) + "\n\n"
    prompt += "### Visual Cues Data:\n" + json.dumps(vlm_payload, indent=2) + "\n\n"

    # --- Conditional Observations Based on Visual Cues ---
    observations = ""
    if vlm_payload["blink_count"] > 6:
        observations += "- I notice you're blinking frequently. Try to keep your eyes relaxed.\n"
    # (A low focus score is a positive sign, so no suggestion is needed.)
    if vlm_payload.get("avg_tension", 0) > 0:  # Adjust threshold as needed.
        observations += "- Your tension appears elevated. Relax your muscles and take a deep breath.\n"
    if vlm_payload["total_head_movement"] > 350:
        observations += "- Your head movement is quite large; please try to remain still.\n"
    if vlm_payload["micro_expression_events"] > 35:
        observations += "- There are frequent micro-expressions. Reflect on a happy thought and let go of any stress.\n"

    if observations:
        prompt += "### Observations:\n" + observations + "\n"

    prompt += (
        "Provide a holistic summary of the individual's stress levels, relaxation state, and personalized recommendations for meditation.\n\n"
        "**Meditation Guidance**\n"
        "Your role: Act as a meditation guide. Avoid analysis; instead, use the data to create a structured session. Use the template below:\n\n"

        "**Framework for a 20-Minute Meditation**\n"
        "- **Grounding in the Present**: 'What do you notice right now?'\n"
        "- **Emotional Check-In**: 'What emotion feels most present?'\n"
        "- **Reflection on the Day**: 'What moment from today stands out?'\n"
        "- **Gratitude Exploration**: 'What is one thing you're grateful for?'\n"
        "- **Releasing Tension**: 'What are you ready to let go of?'\n"
        "- **Setting an Intention**: 'What intention would you like to carry forward?'\n"
        "- **Closing Reflection**: 'What does your heart need to hear?'\n\n"
        """Use a calm and supportive tone, guiding the user through the session. EVERYTHING YOU WRITE WILL BE SPOKEN ALOUD TO THE USER. SO WRITE LIKE YOU ARE SPEAKING TO THEM DIRECTLY. DO NOT WRITE TIME STAMPS LIKE (Minute 1-3: Grounding in the Present) or adverbs like **(slowly)**. Just write the content of the meditation.
You will return a SINGLE QUESTION AT A TIME, FOLLOWING THIS TEMPLATE. IN PARTICULAR, YOU ARE GIVEN THE PROMPTS YOU HAVE PREVIOUSLY ASKED, AND MUST PROVIDE ONLY THE NEXT QUESTION. You should consider the previous questions you have asked to ensure that the questions you ask logically follows the previous ones and consider the user's emotional state. If blinks over 6 suggest to focus on keeping eyes relaxed. If focus score is low that is a good thing. If high tension, tell to relax muscles perhaps. If head movement over 350, tell the user to sit still. If micro expressions are over 35 tell to relax and think about happy thoughts. Also consider how this data has changed with each prompt. 
Each question should take about 45 seconds to read aloud. Where relevant, acknowledge your observations about how the user responded or reacted to your previous question. Ask follow-ups where necessary. Do not overly-ask follow-ups or over-acknowledge. Here are the questions you have already asked:
""" + "\n".join(prompts_asked)) + "\n\n Here are the past visual cue payloads: " + "\n".join(visual_cues_history) + "\n\n Here are the past telemetry payloads: " + "\n".join(telemetry_history) 
    prompt += + '\n\n At the end suggest what kind of matcha or tea would be best to end off the day. Choose between lemon ginger or vanilla matcha.' if len(prompts_asked) == 5 else ''

    logger.info("Built combined prompt.")

    return prompt

def call_google_gemini(prompt):
    """
    Calls the Google Gemini API with the combined prompt.
    """
    try:
        response = gemini_client.models.generate_content(
            contents=prompt,
            model="models/gemini-2.0-flash"
        )
        if response and response.text:
            logger.info("Gemini API call successful.")
            logger.info(f"Response: {response.text}")
            return response.text
        else:
            logger.error("No text generated. Check the Gemini API response. f{response}")
    except Exception as e:
        logger.error(f"Error calling Gemini: {str(e)}")
  
def update_scipt():
    FRAME_WINDOW.image(image="./closeyoureyes.jpg", use_container_width=True)
    state_metrics = st.session_state.metrics

    total_frames = max(state_metrics['total_frames'], 1)
    focus_score = ((total_frames - state_metrics['closed_frames']) / total_frames) * 100
    avg_tension = (state_metrics['tension_score_accum'] / state_metrics['tension_count'] * 100) if state_metrics['tension_count'] > 0 else 0

    # Prepare the visual cues payload.
    vlm_payload = {
        "blink_count": state_metrics['blink_count'],
        "focus_score": focus_score,
        "avg_tension": avg_tension,
        "total_head_movement": state_metrics['total_head_movement'],
        "micro_expression_events": state_metrics['micro_expression_events'],
        "user_id": 1
    }
    logger.info(f"Visual cues payload: {vlm_payload}")
    terra_json = get_terra_from_s3()
    prompt = build_combined_prompt(
            terra_payload=terra_json, 
            vlm_payload=vlm_payload,
    )
    gemini_result = call_google_gemini(prompt)
    st.session_state.current_script = extract_readable_script(gemini_result)
    prompts_asked.append(st.session_state.current_script)

    # Reset metrics for the next interval.
    st.session_state.metrics = {
        'total_frames': 0,
        'blink_count': 0,
        'closed_frames': 0,
        'tension_score_accum': 0,
        'tension_count': 0,
        'total_head_movement': 0,
        'prev_face_center': None,
        'prev_eyebrow_position': None,
        'micro_expression_events': 0
    }

def render_frame(rgb_frame):
    # cv2.putText(rgb_frame, st.session_state.current_script, (10, 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    FRAME_WINDOW.image(rgb_frame, channels="RGB")

def handle_audio():
    audio_stream = elevenlabs_client.text_to_speech.convert_as_stream(
        text=st.session_state.current_script,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        optimize_streaming_latency=3
    )

    audio_bytes = bytearray()
    for chunk in audio_stream:
        if isinstance(chunk, int):
            audio_bytes.append(chunk)
        elif isinstance(chunk, bytes):
            audio_bytes.extend(chunk)
        else:
            st.error(f"Unexpected type in audio stream: {type(chunk)}")
    b64_data = base64.b64encode(audio_bytes).decode("utf-8")

    audio_html = f"""
    <audio id="audio-player" autoplay style="display: none;">
      <source src="data:audio/mp3;base64,{b64_data}" type="audio/mp3">
      Your browser does not support the audio element.
    </audio>
    <script>
      document.getElementById("audio-player").play();
    </script>
    """
    audio_placeholder = st.empty()
    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)


main_loop()




