import cv2
import pandas as pd
import numpy as np

# 1. SETUP
video_input = 'User_15_Short_10.mp4'
csv_input = 'general_eye_gaze15.csv'
video_output = 'UNIFIED_FISHEYE_FIXATION.mp4'

# 2. LOAD DATA
df = pd.read_csv(csv_input)


# 3. FIND BEST FIXATION (Data Analysis Step)
def find_best_fixation(df, window_seconds=1.5):
    # Calculate sampling rate from timestamps
    diffs = df['tracking_timestamp_us'].diff().dropna()
    fps_data = 1 / (diffs.mean() / 1e6)
    window_size = int(window_seconds * fps_data)

    # Search first 30s of data
    search_limit = min(len(df) - window_size, int(30 * fps_data))

    best_idx = 0
    min_dispersion = float('inf')

    for i in range(0, search_limit):
        window = df.iloc[i: i + window_size]
        # Dispersion = average standard deviation of gaze angles
        dispersion = window[['left_yaw_rads_cpf', 'right_yaw_rads_cpf', 'pitch_rads_cpf']].std().mean()
        if dispersion < min_dispersion:
            min_dispersion = dispersion
            best_idx = i

    start_time = (df.iloc[best_idx]['tracking_timestamp_us'] - df['tracking_timestamp_us'].iloc[0]) / 1e6
    print(f"Targeting Fixation at {start_time:.2f}s (Stability: {min_dispersion:.4f})")
    return best_idx


# 4. LENS & CALIBRATION (Unified Parameters)
cap = cv2.VideoCapture(video_input)
fps_vid = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps_vid, (w, h))

# FISHEYE PARAMETERS
FISHEYE_FOV_DEG = 125
PITCH_CORRECTION = 0.165  # Fixed "Up-Down" offset
YAW_CORRECTION = 0.12  # Fixed "Left-Right" offset

center_x, center_y = w // 2, h // 2
diag_px = np.sqrt(w ** 2 + h ** 2)
f_px = (diag_px / 2) / np.radians(FISHEYE_FOV_DEG / 2)

# 5. PROCESSING LOOP
disp_x, disp_y = float(center_x), float(center_y)
frame_idx = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Sync CSV to Video
        cur_us = (frame_idx / fps_vid) * 1e6 + df['tracking_timestamp_us'].iloc[0]
        row = df.iloc[(df['tracking_timestamp_us'] - cur_us).abs().idxmin()]

        # Combined Gaze Angle + Calibration
        yaw = ((row['left_yaw_rads_cpf'] + row['right_yaw_rads_cpf']) / 2) + YAW_CORRECTION
        pitch = row['pitch_rads_cpf'] + PITCH_CORRECTION

        # --- THE FISHEYE PROJECTION ---
        # Calculate angular distance from optical center
        cos_theta = np.cos(yaw) * np.cos(pitch)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # Map to radial distance (Equidistant model: r = f * theta)
        r = f_px * theta

        # Calculate 2D angle on the sensor plane
        phi = np.arctan2(yaw, pitch)

        # Convert Polar to Cartesian Pixels
        # Adding to X, Subtracting from Y (Standard OpenCV orientation)
        tgt_x = center_x + (r * np.sin(phi))
        tgt_y = center_y - (r * np.cos(phi))

        # --- ADAPTIVE SMOOTHING ---
        dist = np.sqrt((tgt_x - disp_x) ** 2 + (tgt_y - disp_y) ** 2)
        alpha = 0.8 if dist > 80 else 0.15

        disp_x += (tgt_x - disp_x) * alpha
        disp_y += (tgt_y - disp_y) * alpha

        # --- DRAWING ---
        cv2.circle(frame, (int(disp_x), int(disp_y)), 20, (255, 255, 255), 2)
        cv2.circle(frame, (int(disp_x), int(disp_y)), 16, (0, 0, 255), -1)

        out.write(frame)
        frame_idx += 1

finally:
    cap.release()
    out.release()
    print(f"Video saved as {video_output}")