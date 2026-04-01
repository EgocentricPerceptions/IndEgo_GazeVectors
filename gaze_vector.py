import cv2
import pandas as pd
import numpy as np
from collections import deque
from projectaria_tools.core import data_provider
import json


def get_angular_corrections_from_jsonl(jsonl_path):
    try:
        with open(jsonl_path, 'r') as f:
            first_line = f.readline()
            data = json.loads(first_line)
    except FileNotFoundError:
        raise Exception(f"Error: File {jsonl_path} missing")

    slam_left_yaw = None
    slam_left_pitch = None
    rgb_yaw = None
    rgb_pitch = None

    for cam in data['CameraCalibrations']:
        if cam['Label'] in ['camera-slam-left', 'camera-rgb']:
            quat = cam['T_Device_Camera']['UnitQuaternion']
            qw, qx, qy, qz = quat[0], quat[1][0], quat[1][1], quat[1][2]

            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw_val = np.arctan2(siny_cosp, cosy_cosp)

            sinp = 2.0 * (qw * qy - qz * qx)
            pitch_val = np.arcsin(np.clip(sinp, -1, 1))

            if cam['Label'] == 'camera-slam-left':
                slam_left_yaw, slam_left_pitch = yaw_val, pitch_val
            else:
                rgb_yaw, rgb_pitch = yaw_val, pitch_val

    if slam_left_yaw is not None and rgb_yaw is not None:
        return rgb_yaw - slam_left_yaw, rgb_pitch - slam_left_pitch

    raise Exception("Error: could not find parameters in JSONL.")


vrs_path = "User_15_Short_10.vrs"
jsonl_path = "online_calibration.jsonl"
video_input = 'User_15_Short_10.mp4'
csv_input = 'general_eye_gaze.csv'
video_output = 'Gaze_Vector_Heatmap_V3.mp4'

YAW_CORR, PITCH_CORR = get_angular_corrections_from_jsonl(jsonl_path)

provider = data_provider.create_vrs_data_provider(vrs_path)
device_calib = provider.get_device_calibration()
rgb_calib = device_calib.get_camera_calib("camera-rgb")

T_device_rgb = device_calib.get_transform_device_sensor("camera-rgb")
pos_flat = T_device_rgb.translation().flatten()

OFFSET_X, OFFSET_Y = -pos_flat[0], -pos_flat[1]
FX, FY = rgb_calib.get_focal_lengths()
CX, CY = rgb_calib.get_principal_point()
native_size = rgb_calib.get_image_size()

cap = cv2.VideoCapture(video_input)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

s_w, s_h = w / native_size[0], h / native_size[1]
FXs, FYs, CXs, CYs = FX * s_w, FY * s_h, CX * s_w, CY * s_h

df = pd.read_csv(csv_input)
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

video_start_ts = df['tracking_timestamp_us'].iloc[0]

heatmap_accum = np.zeros((h, w), dtype=np.float32)
hm_kernel_size = 80
hm_sigma = 25
hm_decay = 0.97
hm_intensity = 0.6

avg_window = 5
coord_buffer = deque(maxlen=avg_window)
path_length = 25
path_history = deque(maxlen=path_length)

alpha_slow = 0.15
alpha_fast = 0.85
saccade_threshold = 60

disp_x, disp_y = float(CXs), float(CYs)
frame_idx = 0

print(f"\nCalibration results:")
print(f"Offset: Yaw={np.rad2deg(YAW_CORR):.2f}°, Pitch={np.rad2deg(PITCH_CORR):.2f}°")
print(f"Translation: X={OFFSET_X * 1000:.2f}mm, Y={OFFSET_Y * 1000:.2f}mm")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cur_us = (frame_idx / fps) * 1e6 + video_start_ts
    row = df.iloc[(df['tracking_timestamp_us'] - cur_us).abs().idxmin()]

    yaw = (row['left_yaw_rads_cpf'] + row['right_yaw_rads_cpf']) / 2 + YAW_CORR
    pitch = row['pitch_rads_cpf'] + PITCH_CORR
    depth = row['depth_m'] if (row['depth_m'] > 0 and not np.isnan(row['depth_m'])) else 1.2

    x_c, y_c = (OFFSET_X / depth) * FXs, (OFFSET_Y / depth) * FYs
    raw_x = CXs + (FXs * np.tan(yaw)) + x_c
    raw_y = CYs - (FYs * (np.tan(pitch) / np.cos(yaw))) - y_c

    if not np.isnan(raw_x) and not np.isnan(raw_y):
        coord_buffer.append((raw_x, raw_y))
        tgt_x = np.mean([p[0] for p in coord_buffer])
        tgt_y = np.mean([p[1] for p in coord_buffer])

        dist = np.sqrt((tgt_x - disp_x) ** 2 + (tgt_y - disp_y) ** 2)
        alpha = alpha_fast if dist > saccade_threshold else alpha_slow
        disp_x += (tgt_x - disp_x) * alpha
        disp_y += (tgt_y - disp_y) * alpha

        current_pos = (int(disp_x), int(disp_y))
        path_history.append(current_pos)

        if 0 <= current_pos[0] < w and 0 <= current_pos[1] < h:
            point_mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(point_mask, current_pos, hm_kernel_size // 2, hm_intensity, -1)
            point_mask = cv2.GaussianBlur(point_mask, (hm_kernel_size | 1, hm_kernel_size | 1), hm_sigma)
            heatmap_accum = cv2.add(heatmap_accum, point_mask)
            heatmap_accum *= hm_decay

            max_val = np.max(heatmap_accum)
            if max_val > 1e-6:
                heatmap_norm = (heatmap_accum / max_val * 255).astype(np.uint8)
            else:
                heatmap_norm = np.zeros((h, w), dtype=np.uint8)

            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.55, heatmap_color, 0.65, 0)

        for i in range(1, len(path_history)):
            thick = int(max(1, (i / path_length) * 4))
            cv2.line(frame, path_history[i - 1], path_history[i], (0, 255, 255), thick)

        heatmap_value = heatmap_accum[current_pos[1], current_pos[0]]
        nonzero_vals = heatmap_accum[heatmap_accum > 1e-6]
        if len(nonzero_vals) > 0:
            ref_val = np.percentile(nonzero_vals, 90)
            intensity = min(1.0, heatmap_value / (ref_val + 1e-6))
        else:
            intensity = 0.0

        red_val = int(120 + 135 * intensity)
        green_val = int(80 * (1.0 - intensity))
        blue_val = int(60 * (1.0 - intensity))

        cv2.circle(frame, current_pos, 14, (220, 220, 220), -1)
        cv2.circle(frame, current_pos, 14, (0, 0, 0), 2)
        cv2.circle(frame, current_pos, 11, (blue_val, green_val, red_val), -1)

    out.write(frame)
    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"Frame processed {frame_idx}...")

print(f"Total frames processed: {frame_idx}")
print(f"Max heatmap intensity: {np.max(heatmap_accum):.2f}")

cap.release()
out.release()
print(f"\nDone, result: {video_output}")