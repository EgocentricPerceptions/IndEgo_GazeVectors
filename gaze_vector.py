import cv2
import pandas as pd
import numpy as np
from collections import deque
from projectaria_tools.core import data_provider
import json


def get_yaw_correction_from_jsonl(jsonl_path):
    """Extract yaw correction from online_calibration.jsonl"""
    try:
        with open(jsonl_path, 'r') as f:
            first_line = f.readline()
            data = json.loads(first_line)

        # Găsește camera-slam-left și camera-rgb
        slam_left_yaw = None
        rgb_yaw = None

        for cam in data['CameraCalibrations']:
            if cam['Label'] == 'camera-slam-left':
                quat_data = cam['T_Device_Camera']['UnitQuaternion']
                qw = quat_data[0]
                qx, qy, qz = quat_data[1]

                siny_cosp = 2.0 * (qw * qz + qx * qy)
                cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                slam_left_yaw = np.arctan2(siny_cosp, cosy_cosp)

            elif cam['Label'] == 'camera-rgb':
                quat_data = cam['T_Device_Camera']['UnitQuaternion']
                qw = quat_data[0]
                qx, qy, qz = quat_data[1]

                siny_cosp = 2.0 * (qw * qz + qx * qy)
                cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                rgb_yaw = np.arctan2(siny_cosp, cosy_cosp)

        if slam_left_yaw is not None and rgb_yaw is not None:
            yaw_correction = rgb_yaw - slam_left_yaw
            return yaw_correction
        else:
            print("Warning: Could not find required cameras in JSONL")
            return 0.0

    except Exception as e:
        print(f"Error reading JSONL: {e}")
        return 0.0


# =========================
# 1. SETUP HARDWARE
# =========================
vrs_path = "User_15_Short_10.vrs"
jsonl_path = "online_calibration.jsonl"
video_input = 'User_15_Short_10.mp4'
csv_input = 'general_eye_gaze.csv'
video_output = 'ARIA_GAZE_V2.mp4'

print("Extracting yaw correction from JSONL...")
YAW_CORRECTION = get_yaw_correction_from_jsonl(jsonl_path)
print(f"Yaw correction from calibration: {YAW_CORRECTION:.4f} rad ({np.rad2deg(YAW_CORRECTION):.2f}°)")

provider = data_provider.create_vrs_data_provider(vrs_path)
device_calib = provider.get_device_calibration()
rgb_calib = device_calib.get_camera_calib("camera-rgb")

# Extragere date SE3 (Aria Gen 2)
T_device_rgb = device_calib.get_transform_device_sensor("camera-rgb")
pos_flat = T_device_rgb.translation().flatten()

# Offset geometric
OFFSET_X_METERS = -pos_flat[0]
OFFSET_Y_METERS = -pos_flat[1]

# Parametri optici
FX, FY = rgb_calib.get_focal_lengths()
CX, CY = rgb_calib.get_principal_point()

# =========================
# 2. CONFIGURARE VIDEO
# =========================
cap = cv2.VideoCapture(video_input)
if not cap.isOpened():
    print("Eroare: Nu s-a putut deschide video-ul!")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Scalare automată la rezoluția video
native_size = rgb_calib.get_image_size()
s_w, s_h = w / native_size[0], h / native_size[1]
FX_s, FY_s, CX_s, CY_s = FX * s_w, FY * s_h, CX * s_w, CY * s_h

df = pd.read_csv(csv_input)
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

smooth_buffer = deque(maxlen=3)
video_start_ts = df['tracking_timestamp_us'].iloc[0]

# =========================
# 3. LOOP PROCESARE
# =========================
print(f"Geometric offset: X={OFFSET_X_METERS * 1000:.2f}mm, Y={OFFSET_Y_METERS * 1000:.2f}mm")
print(f"Yaw correction: {YAW_CORRECTION:.4f} rad ({np.rad2deg(YAW_CORRECTION):.2f}°)")
print("Pornire procesare...")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Sincronizare temporală
    cur_us = (frame_idx / fps) * 1e6 + video_start_ts
    idx = (df['tracking_timestamp_us'] - cur_us).abs().idxmin()
    row = df.iloc[idx]

    # Date gaze cu yaw correction din JSONL
    yaw = (row['left_yaw_rads_cpf'] + row['right_yaw_rads_cpf']) / 2 + YAW_CORRECTION
    pitch = row['pitch_rads_cpf']
    depth = row['depth_m'] if (row['depth_m'] > 0 and not np.isnan(row['depth_m'])) else 1.2

    # Calcul Paralaxă Dinamică
    x_corr = (OFFSET_X_METERS / depth) * FX_s
    y_corr = (OFFSET_Y_METERS / depth) * FY_s

    # Proiecția (Pinhole Model)
    raw_x = CX_s + (FX_s * np.tan(yaw)) + x_corr
    raw_y = CY_s - (FY_s * (np.tan(pitch) / np.cos(yaw))) - y_corr

    # Smoothing
    if not np.isnan(raw_x) and not np.isnan(raw_y):
        smooth_buffer.append((raw_x, raw_y))
        px = int(np.mean([p[0] for p in smooth_buffer]))
        py = int(np.mean([p[1] for p in smooth_buffer]))

        if 0 <= px < w and 0 <= py < h:
            # Desenăm doar un cerc roșu (fără cruce)
            cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)  # Cerc roșu plin
            cv2.circle(frame, (px, py), 12, (0, 0, 255), 2)  # Contur roșu mai mare

    out.write(frame)
    frame_idx += 1
    if frame_idx % 500 == 0:
        print(f"Cadru: {frame_idx}...")

cap.release()
out.release()
print(f"Gata! Rezultat salvat în {video_output}")