import cv2
import pandas as pd
import numpy as np
from scipy.optimize import minimize

video_input = 'User_15_Short_10.mp4'
csv_input = 'general_eye_gaze.csv'
video_output = 'AUTO_FIXATION_CALIBRATED.mp4'

#choosing the middle 1920x1640
TARGET_X, TARGET_Y = 960, 820


#we find the point in the video where we have the longest fixation (we found that also in the data analysis step) and
#in that interval
def find_best_fixation(df, window_seconds=1.5, search_limit_sec=30):

    fps_data = 1 / (df['tracking_timestamp_us'].diff().mean() / 1e6)
    window_size = int(window_seconds * fps_data)
    search_limit_idx = int(search_limit_sec * fps_data)

    best_idx = 0
    min_dispersion = float('inf')

    # Calculate STD
    for i in range(0, search_limit_idx - window_size):
        window = df.iloc[i: i + window_size]
        #calculate the average
        dispersion = window[['left_yaw_rads_cpf', 'right_yaw_rads_cpf', 'pitch_rads_cpf']].std().mean()

        if dispersion < min_dispersion:
            min_dispersion = dispersion
            best_idx = i

    start_time = (df.iloc[best_idx]['tracking_timestamp_us'] - df['tracking_timestamp_us'].iloc[0]) / 1e6
    print(f"Fixation (optimum/longest) starts at: {start_time:.2f} (Stability: {min_dispersion:.4f})")
    return best_idx, window_size


#We have an egocentric perception so we need to inverse the y axis for our projection
def get_projection(params, yaw, pitch, w, h):
    v_fx, v_fy, v_offx, v_offy = params
    X = np.cos(pitch) * np.sin(yaw)
    Y = np.sin(pitch)
    Z = np.cos(pitch) * np.cos(yaw)
    if abs(Z) < 1e-6: Z = 1e-6

    px = v_fx * (X / Z) + (w / 2) + v_offx
    py = (h / 2) - (v_fy * (Y / Z)) + v_offy
    return px, py


#A manual calibration is not optimal, so we use the data from where we have found the interval for the optimal fixation
#and choose our calibration interval
df = pd.read_csv(csv_input)
cap = cv2.VideoCapture(video_input)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_vid = cap.get(cv2.CAP_PROP_FPS)

start_idx, w_size = find_best_fixation(df)
calib_data = df.iloc[start_idx: start_idx + w_size]

avg_yaw = (calib_data['left_yaw_rads_cpf'] + calib_data['right_yaw_rads_cpf']).mean() / 2
avg_pitch = calib_data['pitch_rads_cpf'].mean()


#Optimization -performs a constrained optimization to align the projected gaze with the target area
def objective(p):
    px, py = get_projection(p, avg_yaw, avg_pitch, w, h)
    return (px - TARGET_X) ** 2 + (py - TARGET_Y) ** 2


res = minimize(objective, [900, 900, 0, 400], method='L-BFGS-B',
               bounds=[(600, 1500), (600, 1500), (-300, 300), (100, 800)])
f_x, f_y, o_x, o_y = res.x

#Generating end video
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps_vid, (w, h))
smooth_x, smooth_y = w // 2, h // 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    f_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cur_us = (f_num / fps_vid) * 1e6 + df['tracking_timestamp_us'].iloc[0]
    row = df.iloc[(df['tracking_timestamp_us'] - cur_us).abs().idxmin()]

    rx, ry = get_projection([f_x, f_y, o_x, o_y],
                            (row['left_yaw_rads_cpf'] + row['right_yaw_rads_cpf']) / 2,
                            row['pitch_rads_cpf'], w, h)

    smooth_x += (np.clip(rx, 0, w - 1) - smooth_x) * 0.15
    smooth_y += (np.clip(ry, 0, h - 1) - smooth_y) * 0.15

    cv2.circle(frame, (int(smooth_x), int(smooth_y)), 10, (0, 0, 255), -1)
    out.write(frame)

cap.release()
out.release()
print(f"Finalized process now, with the next parameters for best fixation: fx={f_x:.1f}, fy={f_y:.1f}, offset_y={o_y:.1f}")