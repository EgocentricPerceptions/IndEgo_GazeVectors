import json
from projectaria_tools.core import data_provider

vrs_path = "User_15_Short_10.vrs"
provider = data_provider.create_vrs_data_provider(vrs_path)

# Luăm calibrarea pentru camera RGB și cea de Eye Tracking
device_calib = provider.get_device_calibration()
rgb_calib = device_calib.get_camera_calib("device-rgb")
et_calib = device_calib.get_camera_calib("device-et-left")


calib_data = {
    "fx": rgb_calib.get_focal_lengths()[0],
    "fy": rgb_calib.get_focal_lengths()[1],
    "cx": rgb_calib.get_principal_point()[0],
    "cy": rgb_calib.get_principal_point()[1],
    "t_device_et": et_calib.get_transform_device_camera().translation().tolist(),
    "r_device_et": et_calib.get_transform_device_camera().rotation().to_euler_angles().tolist()
}

with open('calibration_data.json', 'w') as f:
    json.dump(calib_data, f)

