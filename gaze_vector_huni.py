import cv2
import pandas as pd
import numpy as np
from collections import deque

# SETUP
video_input = 'User_14_Short_10.mp4'
csv_input = 'general_eye_gaze.csv'
video_output = 'FINAL_SMOOTH_SCANPATH14.mp4'

# LOAD DATA
df = pd.read_csv(csv_input)

# VIDEO SPECS
cap = cv2.VideoCapture(video_input)
fps_vid = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps_vid, (w, h))

# --- CALIBRATION & LENS ---
# Manual calibration to adjust errors in pitch and yaw
# (The camera seems to be located on the top of the head and a bit to the left)
FISHEYE_FOV_DEG = 125
PITCH_CORRECTION = 0.165 # Add to move up
YAW_CORRECTION = 0.12 # Add to move to the right

# Calculates focal distance based on the Fisheye lens and the raw video
center_x, center_y = w // 2, h // 2
diag_px = np.sqrt(w ** 2 + h ** 2)
f_px = (diag_px / 2) / np.radians(FISHEYE_FOV_DEG / 2)

# SMOOTHING & SCANPATH CONFIG
# Moving average buffer (higher = steadier but more lag)
avg_window = 5
coord_buffer = deque(maxlen=avg_window)

# Scanpath history
path_length = 25
path_history = deque(maxlen=path_length)

# Adaptive Smoothing Alpha
alpha_slow = 0.15  # For fixations
alpha_fast = 0.85  # For saccades (snapping)
saccade_threshold = 60  # Pixels

# INITIALIZE POSITIONS
disp_x, disp_y = float(center_x), float(center_y)
frame_idx = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Sync CSV to Video
        cur_us = (frame_idx / fps_vid) * 1e6 + df['tracking_timestamp_us'].iloc[0]
        row = df.iloc[(df['tracking_timestamp_us'] - cur_us).abs().idxmin()]

        # RAW GAZE MATH + Adding corrections
        yaw = ((row['left_yaw_rads_cpf'] + row['right_yaw_rads_cpf']) / 2) + YAW_CORRECTION
        pitch = row['pitch_rads_cpf'] + PITCH_CORRECTION

        # Fisheye Projection
        cos_theta = np.cos(yaw) * np.cos(pitch)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        r = f_px * theta
        phi = np.arctan2(yaw, pitch)

        raw_x = center_x + (r * np.sin(phi))
        raw_y = center_y - (r * np.cos(phi))

        # MOVING AVERAGE (Primary Jitter Filter)
        # In order to optimize the smoothnes we calculate the average movement for a certain number of frames
        coord_buffer.append((raw_x, raw_y))
        tgt_x = np.mean([p[0] for p in coord_buffer])
        tgt_y = np.mean([p[1] for p in coord_buffer])

        # ADAPTIVE SMOOTHING (Secondary Movement Filter) based on the magnitude of the jump
        dist = np.sqrt((tgt_x - disp_x) ** 2 + (tgt_y - disp_y) ** 2)

        # Decide how fast to follow the target
        alpha = alpha_fast if dist > saccade_threshold else alpha_slow

        disp_x += (tgt_x - disp_x) * alpha
        disp_y += (tgt_y - disp_y) * alpha

        # SCANPATH UPDATE - deciding current position based on the dispersion values above
        current_pos = (int(disp_x), int(disp_y))
        path_history.append(current_pos)

        # DRAWING
        # Draw Scanpath Lines (Yellow)
        for i in range(1, len(path_history)):
            # Fade thickness for older segments
            thick = int(max(1, (i / path_length) * 4))
            cv2.line(frame, path_history[i - 1], path_history[i], (0, 255, 255), thick)

        # Draw Current Gaze Point (Red with White border)
        cv2.circle(frame, current_pos, 20, (255, 255, 255), 2)
        cv2.circle(frame, current_pos, 16, (0, 0, 255), -1)

        out.write(frame)
        frame_idx += 1

finally:
    cap.release()
    out.release()
    print(f"Output: {video_output}")
