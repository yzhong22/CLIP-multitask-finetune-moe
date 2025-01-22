import os
import glob
import numpy as np
from tqdm import tqdm

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt


def dicom2array(path):
    dicom = pydicom.dcmread(path)

    intercept = dicom.RescaleIntercept if "RescaleIntercept" in dicom else -1024
    slope = dicom.RescaleSlope if "RescaleSlope" in dicom else 1

    window_level = -600
    window_width = 1500

    data = np.clip(
        dicom.pixel_array * slope + intercept, window_level - window_width // 2, window_level + window_width // 2
    )
    data = data - (window_level - window_width // 2)
    data = data / window_width
    data = (data * 255).astype(np.uint8)
    return data


count = 0

files = glob.glob("/research/d5/gds/yzhong22/datasets/multitask-moe/rsna-str-pulmonary-embolism-detection/*/*/*/*.dcm")

for file in tqdm(files):
    split, subject, exam, sample = file.split("/")[-4:]

    img = dicom2array(file)

    output_path = f"/research/d5/gds/yzhong22/datasets/multitask-moe/rsna-str-pulmonary-embolism-detection/{split}-png/{subject}/{exam}"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cv2.imwrite(
        f"{output_path}/{sample.replace('.dcm', '.png')}",
        img,
    )

    # count += 1

    # if count >= 10:
    #     break
