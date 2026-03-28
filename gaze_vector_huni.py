import cv2
import pandas as pd
import numpy as np
from collections import deque
from projectaria_tools.core import data_provider

vrs_path = "User_15_Short_10.vrs"
provider = data_provider.create_vrs_data_provider(vrs_path)
device_calib = provider.get_device_calibration()

rgb_calib = device_calib.get_camera_calib("camera-rgb") or device_calib.get_camera_calib("device-rgb")
if rgb_calib is None:
    raise ValueError("RGB calibration not found")

FX, FY = rgb_calib.get_focal_lengths()
CX, CY = rgb_calib.get_principal_point()
f_avg = (FX + FY) / 2

video_input = 'User_15_Short_10.mp4'
csv_input = 'general_eye_gaze.csv'
video_output = 'PRECISION.mp4'

df = pd.read_csv(csv_input)
cap = cv2.VideoCapture(video_input)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

smooth_buffer = deque(maxlen=5)
all_errors = []
frame_idx = 0
video_start_ts = df['tracking_timestamp_us'].iloc[0]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    cur_us = (frame_idx / fps) * 1e6 + video_start_ts
    idx = (df['tracking_timestamp_us'] - cur_us).abs().idxmin()
    row = df.iloc[idx]

    yaw = (row['left_yaw_rads_cpf'] + row['right_yaw_rads_cpf']) / 2
    pitch = row['pitch_rads_cpf']

    raw_x = CX + (FX * np.tan(yaw))
    raw_y = CY - (FY * (np.tan(pitch) / np.cos(yaw)))

    smooth_buffer.append((raw_x, raw_y))
    avg_x, avg_y = int(np.mean([p[0] for p in smooth_buffer])), int(np.mean([p[1] for p in smooth_buffer]))

    pixel_error = np.sqrt((raw_x - avg_x) ** 2 + (raw_y - avg_y) ** 2)
    eroare_grade = np.degrees(np.arctan(pixel_error / f_avg))
    all_errors.append(eroare_grade)

    accuracy_percent = max(0, 100 - (eroare_grade * 20))

    cv2.putText(frame, f"Accuracy: {accuracy_percent:.1f}%", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Error: {eroare_grade:.2f} deg", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.circle(frame, (avg_x, avg_y), 15, (0, 0, 255), -1)
    cv2.circle(frame, (avg_x, avg_y), 20, (255, 255, 255), 2)

    out.write(frame)
    frame_idx += 1

    if frame_idx % 200 == 0:
        print(f"Procesat cadru {frame_idx} Accuracy curent: {accuracy_percent:.1f}%")

cap.release()
out.release()

print("\n" + "=" * 30)
print(f"Eroare medie: {np.mean(all_errors):.4f} grade")
print(f"Accuracy mediu: {max(0, 100 - (np.mean(all_errors) * 20)):.2f}%")
print(f"Video salvat: {video_output}")
print("=" * 30)
#filtered mi-a dat 95% mediu e pentru salturi