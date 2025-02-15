import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import cv2
import time
from datetime import datetime
import base64
import google.generativeai as genai
from PIL import Image
import io

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# List available models to debug
print("Available models:", genai.list_models())

# Initialize Gemini Vision model with the correct version
model = genai.GenerativeModel('gemini-2.0-flash')  # Updated model name

# Predefined list of triggers
TRIGGER_LIST = ['iphone', 'waterbottle', 'chairs']

def classify_frame(frame):
    """
    This function sends a frame to the Google Gemini API and analyzes the content
    for objects matching the trigger list.
    """
    try:
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Create a byte stream for the image
        byte_stream = io.BytesIO()
        pil_image.save(byte_stream, format='JPEG')
        byte_stream.seek(0)
        
        # More specific prompt for better object detection
        prompt = f"""
        Look at this image carefully. You are an object detection system.
        Your task is to check if any of these specific objects are present: {', '.join(TRIGGER_LIST)}.
        
        Respond in this exact format:
        OBJECTS_FOUND:
        - [object name]
        
        Only list objects that are exactly matching the provided list.
        If no objects from the list are found, respond with:
        OBJECTS_FOUND:
        None
        """
        
        # Generate content using Gemini with generation config
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Lower temperature for more focused responses
            top_p=0.1,
            top_k=32,
        )
        
        response = model.generate_content(
            contents=[prompt, Image.open(byte_stream)],
            generation_config=generation_config
        )
        
        # Make sure the response is complete
        response.resolve()
        
        # Debug print
        print(f"Raw Gemini response: {response.text}")
        
        # Parse the response
        response_text = response.text.lower()
        detected_objects = []
        
        # Check if we got a response and it contains our marker
        if 'objects_found:' in response_text:
            # Split at our marker and get the list part
            objects_part = response_text.split('objects_found:')[1].strip()
            
            # If not "none", process the items
            if 'none' not in objects_part:
                # Split by newlines and process each line
                items = [item.strip('- []').strip() for item in objects_part.split('\n') if item.strip()]
                detected_objects = [item for item in items if item in TRIGGER_LIST]
        
        # Debug print
        print(f"Detected objects: {detected_objects}")
        
        return detected_objects
        
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return []

# Streamlit UI
st.title("Real-Time Trigger Recognition with Google Gemini")
st.write("This demo captures video frames and uses Gemini to detect specific objects.")

# Initialize session state for running flag if it doesn't exist
if 'running' not in st.session_state:
    st.session_state.running = False

# Single button to toggle video state
if not st.session_state.running:
    if st.button("Start Video"):
        st.session_state.running = True
        st.rerun()
else:
    if st.button("Stop Video"):
        st.session_state.running = False
        st.rerun()

# Add a frame counter to track processing
frame_counter = 0

# Main video processing loop
if st.session_state.running:
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Create placeholders
    video_placeholder = st.empty()
    log_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Initialize logs
    logs = []
    
    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture video frame.")
                break
            
            frame_counter += 1
            
            # Process every 30th frame (approximately every 1 second at 30fps)
            if frame_counter % 30 == 0:
                status_placeholder.text("Processing frame...")
                print(f"Processing frame {frame_counter}")
                
                detected_objects = classify_frame(frame)
                
                # Log any triggers
                for obj in detected_objects:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"{timestamp}: Detected {obj}"
                    logs.append(log_entry)
                    # Console log with clear marker
                    print(f"🔔 TRIGGER DETECTED: {obj} at {timestamp}")
                
                status_placeholder.text("Waiting for next frame...")
            
            # Display the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")
            
            # Update logs
            if logs:
                log_placeholder.text("\n".join(logs[-10:]))
            
            # Brief sleep to control processing rate
            time.sleep(0.03)  # Approximately 30 fps
            
    except Exception as e:
        print(f"Application error: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        st.error(f"An error occurred: {str(e)}")
    finally:
        cap.release()
        st.write("Video stream ended.")
        st.session_state.running = False