import streamlit as st
import cv2
import numpy as np
import dlib
from imutils import face_utils
import tempfile
import time

# --- Helper Functions ---
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute the euclidean distance between horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    # Return the eye aspect ratio (EAR)
    return (A + B) / (2.0 * C)

def average_point(points):
    # Compute the average (x, y) for a set of points
    return np.mean(points, axis=0)

# --- Streamlit Layout ---
st.title("Meditation Quality Analyzer")
st.write("""
Upload a video of a person meditating. This app will analyze:
1. **Facial Muscle Tension Dynamics**  
2. **Eye Movement & Blink Rate**  
3. **Head Movement & Postural Sway**  
4. **Micro-Expression Recognition**
""")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.write("Processing video...")

    # --- Initialize dlib's face detector and landmark predictor ---
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(video_path)

    # --- Metrics and Variables ---
    total_frames = 0
    closed_frames = 0
    blink_count = 0
    eye_closed = False  # To detect transitions for blink counting

    # For head movement tracking:
    prev_face_center = None
    total_head_movement = 0

    # For micro-expression detection (tracking eyebrow position changes)
    micro_expression_events = 0
    prev_eyebrow_position = None

    # For facial tension (heuristic: distance between eyebrow and eye)
    tension_score_accum = 0
    tension_count = 0

    # --- Thresholds and Parameters ---
    EAR_THRESHOLD = 0.2        # Below this, eyes are considered closed
    EYEBROW_EYE_NORM_THRESHOLD = 0.15  # Normalized distance threshold for tension (heuristic)
    MICRO_EXPR_THRESHOLD = 3.0   # Pixel change threshold (requires calibration)

    # Create a placeholder to show video frames in Streamlit
    frame_placeholder = st.empty()

    # Process the video frame-by-frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        # Convert frame to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape_np = face_utils.shape_to_np(shape)

            # ----- Eye Movement & Blink Rate -----
            rightEye = shape_np[36:42]
            leftEye = shape_np[42:48]
            rightEAR = eye_aspect_ratio(rightEye)
            leftEAR = eye_aspect_ratio(leftEye)
            ear = (rightEAR + leftEAR) / 2.0

            # Blink detection: count a blink when transitioning from open to closed
            if ear < EAR_THRESHOLD:
                closed_frames += 1
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                eye_closed = False

            # Draw eye landmarks
            for (x, y) in np.concatenate((rightEye, leftEye), axis=0):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # ----- Facial Muscle Tension Dynamics -----
            # Extract left and right eyebrows and corresponding eyes
            left_eyebrow = shape_np[17:22]
            right_eyebrow = shape_np[22:27]
            left_eye_center = average_point(shape_np[36:42])
            right_eye_center = average_point(shape_np[42:48])
            left_distance = np.linalg.norm(average_point(left_eyebrow) - left_eye_center)
            right_distance = np.linalg.norm(average_point(right_eyebrow) - right_eye_center)
            avg_eyebrow_eye_distance = (left_distance + right_distance) / 2.0

            # Normalize by face width for scale invariance
            face_width = face.right() - face.left()
            normalized_distance = avg_eyebrow_eye_distance / face_width

            # If the normalized distance is too low, assume tension (e.g., furrowed brows)
            tension = normalized_distance <  
            tension_score_accum += (1 if tension else 0)
            tension_count += 1

            # Draw the eyebrows for visualization
            for (x, y) in np.concatenate((left_eyebrow, right_eyebrow), axis=0):
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # ----- Head Movement & Postural Sway -----
            # Calculate face center from all landmarks
            face_center = average_point(shape_np)
            cv2.circle(frame, (int(face_center[0]), int(face_center[1])), 3, (0, 0, 255), -1)
            if prev_face_center is not None:
                movement = np.linalg.norm(face_center - prev_face_center)
                total_head_movement += movement
            prev_face_center = face_center

            # ----- Micro-Expression Recognition -----
            # Use the average eyebrow position to detect sudden changes
            current_eyebrow_position = average_point(np.concatenate((left_eyebrow, right_eyebrow), axis=0))
            if prev_eyebrow_position is not None:
                diff = np.linalg.norm(current_eyebrow_position - prev_eyebrow_position)
                if diff > MICRO_EXPR_THRESHOLD:
                    micro_expression_events += 1
            prev_eyebrow_position = current_eyebrow_position

            # ----- Annotate Frame with Metrics -----
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Head Move: {total_head_movement:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Micro-Expr: {micro_expression_events}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Show the processed frame in Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        time.sleep(0.03)  # Small delay for smoother playback

    cap.release()

    # --- Compute Summary Metrics ---
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    duration_minutes = (total_frames / fps) / 60
    blink_rate = blink_count / duration_minutes if duration_minutes > 0 else blink_count
    focus_score = ((total_frames - closed_frames) / total_frames * 100) if total_frames > 0 else 0
    avg_tension = (tension_score_accum / tension_count * 100) if tension_count > 0 else 0

    # --- Display Summary ---
    st.write("### Meditation Analysis Summary")
    st.write(f"**Total Frames Processed:** {total_frames}")
    st.write(f"**Blinks Detected:** {blink_count}")
    st.write(f"**Blink Rate:** {blink_rate:.2f} blinks/min")
    st.write(f"**Focus Score (Frames with eyes open):** {focus_score:.2f}%")
    st.write(f"**Facial Tension Detected:** {avg_tension:.2f}% of frames (lower normalized eyebrow-eye distance indicates tension)")
    st.write(f"**Total Head Movement:** {total_head_movement:.2f} pixels displacement")
    st.write(f"**Micro-Expression Events Detected:** {micro_expression_events}")

    st.write("**Note:** Thresholds and heuristics used in this demo may need calibration for robust real-world performance.")
