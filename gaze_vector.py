import cv2
import pandas as pd
import numpy as np

# 1. SETUP
video_input = 'User_14_Short_10.mp4'
csv_input = 'general_eye_gaze.csv'
video_output = 'gaze_fisheye_fixed.mp4'

# 2. LOAD DATA
df = pd.read_csv(csv_input)

# 3. VIDEO SPECS
cap = cv2.VideoCapture(video_input)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# --- FISHEYE CALIBRATION PARAMETERS ---
# Adjust these if the dot doesn't reach the edges correctly
FISHEYE_FOV_DEG = 130
center_x, center_y = w // 2, h // 2
# Calculate focal length in pixels based on the diagonal FOV
diag_px = np.sqrt(w**2 + h**2)
f_px = (diag_px / 2) / np.radians(FISHEYE_FOV_DEG / 2)

# Smoothing variables
alpha = 0.15        # Movement fluidity (0.0 to 1.0)
disp_x, disp_y = center_x, center_y

frame_idx = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Sync CSV row to Frame
        cur_us = (frame_idx / fps) * 1e6 + df['tracking_timestamp_us'].iloc[0]
        row = df.iloc[(df['tracking_timestamp_us'] - cur_us).abs().idxmin()]

        # Get angles
        yaw = row['avg_yaw'] if 'avg_yaw' in row else (row['left_yaw_rads_cpf'] + row['right_yaw_rads_cpf'])/2
        pitch = row['pitch_rads_cpf']

        # --- FISHEYE PROJECTION LOGIC ---
        # 1. Theta is the total angular deviation from the center axis
        # cos(theta) = cos(yaw) * cos(pitch)
        cos_theta = np.cos(yaw) * np.cos(pitch)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # 2. Radial distance from center in pixels (Equidistant model: r = f * theta)
        r = f_px * theta

        # 3. Projection direction (using the ratio of yaw and pitch)
        # We use arctan2 to find the 2D angle of the gaze on the image sensor
        phi = np.arctan2(yaw, pitch)

        # 4. Convert polar (r, phi) to Cartesian (x, y)
        # Note: We subtract from pitch because in images, Y increases downwards
        tgt_x = center_x + (r * np.sin(phi))
        tgt_y = center_y - (r * np.cos(phi))

        # --- SMOOTHING ---
        disp_x += (tgt_x - disp_x) * alpha
        disp_y += (tgt_y - disp_y) * alpha

        # Draw Output
        # Red center, White border for visibility
        cv2.circle(frame, (int(disp_x), int(disp_y)), 18, (0, 0, 255), -1)
        cv2.circle(frame, (int(disp_x), int(disp_y)), 20, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

finally:
    cap.release()
    out.release()
    print(f"Render Complete. Saved to {video_output}")
