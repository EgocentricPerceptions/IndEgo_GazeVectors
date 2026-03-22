import cv2
import pandas as pd
import numpy as np

# 1. SETUP
video_input = 'User_14_Short_10.mp4'  # Ensure your video file is in the same folder
csv_input = 'general_eye_gaze.csv'
video_output = 'gaze_smooth_overlay.mp4'

# 2. LOAD & PREPARE DATA
df = pd.read_csv(csv_input)

# Calculate positions in meters
df['avg_yaw'] = (df['left_yaw_rads_cpf'] + df['right_yaw_rads_cpf']) / 2
df['g_x'] = np.sin(df['avg_yaw']) * np.cos(df['pitch_rads_cpf']) * df['depth_m']
df['g_y'] = np.sin(df['pitch_rads_cpf']) * df['depth_m']

# 3. VIDEO SPECS
cap = cv2.VideoCapture(video_input)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Conversion Factors
px_m_x, px_m_y = w / 0.5, h / 0.4

# 4. FILTERING VARIABLES
curr_x, curr_y = w // 2, h // 2
disp_x, disp_y = w // 2, h // 2  # This is the "displayed" smooth position

# Tweak these to your liking:
alpha = 0.12        # Smoothness (lower = more fluid/slower, higher = snappier)
deadzone_px = 30    # Minimum pixel jump required to trigger a move

frame_idx = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Match Video frame to CSV timestamp
        cur_us = (frame_idx / fps) * 1e6 + df['tracking_timestamp_us'].iloc[0]
        row = df.iloc[(df['tracking_timestamp_us'] - cur_us).abs().idxmin()]

        # Target Pixel Position
        tgt_x = int(w/2 + (row['g_x'] * px_m_x))
        tgt_y = int(h/2 - (row['g_y'] * px_m_y))

        # DEADZONE: Only update 'curr' if change is significant
        dist = np.sqrt((tgt_x - curr_x)**2 + (tgt_y - curr_y)**2)
        if dist > deadzone_px:
            curr_x, curr_y = tgt_x, tgt_y

        # INTERPOLATION: Move the displayed dot smoothly toward 'curr'
        disp_x += (curr_x - disp_x) * alpha
        disp_y += (curr_y - disp_y) * alpha

        # Draw the Gaze Dot
        cv2.circle(frame, (int(disp_x), int(disp_y)), 20, (0, 0, 255), -1)
        cv2.circle(frame, (int(disp_x), int(disp_y)), 22, (255, 255, 255), 3)

        out.write(frame)
        frame_idx += 1

finally:
    cap.release()
    out.release()
    print("Video rendered with decisive smoothing. ")

