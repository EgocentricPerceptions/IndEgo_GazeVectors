import cv2
import pandas as pd
import numpy as np

# 1. SETUP
video_input = 'User_14_Short_10.mp4'
csv_input = 'general_eye_gaze.csv'
video_output = 'gaze_final_calibrated.mp4'

# 2. LOAD DATA
df = pd.read_csv(csv_input)

# 3. VIDEO SPECS
cap = cv2.VideoCapture(video_input)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# --- CALIBRATION SETTINGS ---
FISHEYE_FOV_DEG = 125  # Decrease if dot stays too centered, Increase if too far out
PITCH_CORRECTION = -0.1  # Negative moves the dot UP (adjust this to hit the tape)
YAW_CORRECTION = 0.12  # Adjust if the dot is consistently left or right

center_x, center_y = w // 2, h // 2
diag_px = np.sqrt(w ** 2 + h ** 2)
f_px = (diag_px / 2) / np.radians(FISHEYE_FOV_DEG / 2)

# --- INITIALIZE FILTERING VARIABLES ---
disp_x, disp_y = float(center_x), float(center_y)
tgt_x, tgt_y = float(center_x), float(center_y)

frame_idx = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Sync CSV row to Frame
        cur_us = (frame_idx / fps) * 1e6 + df['tracking_timestamp_us'].iloc[0]
        row = df.iloc[(df['tracking_timestamp_us'] - cur_us).abs().idxmin()]

        # Get and Correct Angles
        yaw = ((row['left_yaw_rads_cpf'] + row['right_yaw_rads_cpf']) / 2) + YAW_CORRECTION
        pitch = row['pitch_rads_cpf'] + PITCH_CORRECTION

        # --- FISHEYE PROJECTION ---
        cos_theta = np.cos(yaw) * np.cos(pitch)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        r = f_px * theta
        phi = np.arctan2(yaw, pitch)

        tgt_x = center_x + (r * np.sin(phi))
        tgt_y = center_y - (r * np.cos(phi))

        # --- ADAPTIVE SMOOTHING (Saccade Detection) ---
        dist_to_target = np.sqrt((tgt_x - disp_x) ** 2 + (tgt_y - disp_y) ** 2)

        # If jump is > 80px, it's likely a fast eye movement (Saccade)
        if dist_to_target > 80:
            alpha = 0.8  # Snap to target
        else:
            alpha = 0.15  # Smooth drift for fixations

        disp_x += (tgt_x - disp_x) * alpha
        disp_y += (tgt_y - disp_y) * alpha

        # --- DRAWING ---
        # Outer glow/border
        cv2.circle(frame, (int(disp_x), int(disp_y)), 22, (255, 255, 255), 2)
        # Solid Red Gaze Dot
        cv2.circle(frame, (int(disp_x), int(disp_y)), 18, (0, 0, 255), -1)

        out.write(frame)
        frame_idx += 1

finally:
    cap.release()
    out.release()
    print(f"Success! Video rendered to {video_output}")