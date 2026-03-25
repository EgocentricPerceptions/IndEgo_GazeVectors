import cv2
import pandas as pd
import numpy as np
from collections import deque

video_input = 'User_15_Short_10.mp4'
csv_input = 'general_eye_gaze.csv'
video_output = 'FINAL_HEATMAP_SCANPATH14.mp4'

#loading data
df = pd.read_csv(csv_input)

#video specifications
cap = cv2.VideoCapture(video_input) #we capture the video and open it in video_imput to process frame by frame
fps_vid = cap.get(cv2.CAP_PROP_FPS) #we make sure that the processed video has the same speed-fps-as the original
#we determine the dimension in pixels - openCv resturns a float so we use a wrapper to int
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#param needed: the output, fourcc = four parameter code - for mp4 format, fps, dimension in pixels
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps_vid, (w, h))

#CALIBRATION & LENS - with manual calibration
#transforming the coordinates from 2d in 3d
FISHEYE_FOV_DEG = 125
PITCH_CORRECTION = 0.165
YAW_CORRECTION = 0.12

#the optic trajectory is build with the premise that (0,0,0) is equivalent to the center of the video frame
center_x, center_y = w // 2, h // 2
diag_px = np.sqrt(w ** 2 + h ** 2)
f_px = (diag_px / 2) / np.radians(FISHEYE_FOV_DEG / 2)

#heatmap config
heatmap_accum = np.zeros((h, w), dtype=np.float32)

hm_kernel_size = 80    # Size of the "heat" spot. Larger = blurrier/wider areas.
hm_sigma = 25          # STD. Higher = smoother edges.
hm_decay = 0.99        # Temporal decay (0.0 to 1.0)
hm_intensity = 0.6
hm_alpha = 0.4         # Transparency of the heatmap overlay

#smoothing - we calculate the average on a queue of 5 frames
avg_window = 5
coord_buffer = deque(maxlen=avg_window)
path_length = 25
path_history = deque(maxlen=path_length)

alpha_slow = 0.15
alpha_fast = 0.85 #for adaptive smoothing
saccade_threshold = 60

#initializing positions
disp_x, disp_y = float(center_x), float(center_y)
frame_idx = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        #sync CSV to Video
        cur_us = (frame_idx / fps_vid) * 1e6 + df['tracking_timestamp_us'].iloc[0]
        row = df.iloc[(df['tracking_timestamp_us'] - cur_us).abs().idxmin()]

        #raw gaze math
        yaw = ((row['left_yaw_rads_cpf'] + row['right_yaw_rads_cpf']) / 2) + YAW_CORRECTION
        pitch = row['pitch_rads_cpf'] + PITCH_CORRECTION

        #fisheye Projection
        cos_theta = np.cos(yaw) * np.cos(pitch)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        r = f_px * theta
        phi = np.arctan2(yaw, pitch)

        raw_x = center_x + (r * np.sin(phi))
        raw_y = center_y - (r * np.cos(phi))

        #moving average
        coord_buffer.append((raw_x, raw_y))
        tgt_x = np.mean([p[0] for p in coord_buffer])
        tgt_y = np.mean([p[1] for p in coord_buffer])

        #adaptive smoothing
        dist = np.sqrt((tgt_x - disp_x) ** 2 + (tgt_y - disp_y) ** 2)
        alpha = alpha_fast if dist > saccade_threshold else alpha_slow
        disp_x += (tgt_x - disp_x) * alpha
        disp_y += (tgt_y - disp_y) * alpha

        current_pos = (int(disp_x), int(disp_y))
        path_history.append(current_pos)

        # 1.Create a single-point heat mask for the current gaze position
        point_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(point_mask, current_pos, hm_kernel_size // 2, (hm_intensity), -1)
        point_mask = cv2.GaussianBlur(point_mask, (hm_kernel_size | 1, hm_kernel_size | 1), hm_sigma)

        # 2.Add current heat to the accumulation buffer and apply decay
        heatmap_accum = cv2.add(heatmap_accum, point_mask)
        heatmap_accum *= hm_decay

        # 3.Convert accumulation buffer to a visible 8-bit color map
        heatmap_norm = np.clip(heatmap_accum, 0, 1) * 255
        heatmap_norm = heatmap_norm.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # 4.Blend the heatmap with the original frame
        frame = cv2.addWeighted(frame, 1.0, heatmap_color, hm_alpha, 0)

        # SCANPATH
        for i in range(1, len(path_history)):
            thick = int(max(1, (i / path_length) * 4))
            cv2.line(frame, path_history[i - 1], path_history[i], (0, 255, 255), thick)

        #draw Current Gaze Point
        cv2.circle(frame, current_pos, 10, (255, 255, 255), 2)
        cv2.circle(frame, current_pos, 6, (0, 0, 255), -1)

        out.write(frame)
        frame_idx += 1

finally:
    cap.release()
    out.release()
    print(f"Process finished. Output saved as: {video_output}")