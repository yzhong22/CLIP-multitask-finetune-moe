import os
import glob
import numpy as np
from tqdm import tqdm

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt


def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.dcmread(path)
    # dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


count = 0

files = glob.glob("/research/d5/gds/yzhong22/datasets/multitask-moe/siim-covid19-detection/*/*/*/*.dcm")

for file in tqdm(files):
    split, subject, exam, sample = file.split("/")[-4:]

    # print(file)

    img = dicom2array(file)

    H, W = img.shape
    # print(img.shape)
    scale = 320 / min(H, W)
    img = cv2.resize(img, (int(W * scale + 0.5), int(H * scale + 0.5)))

    output_path = (
        f"/research/d5/gds/yzhong22/datasets/multitask-moe/siim-covid19-detection/{split}-png/{subject}/{exam}"
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cv2.imwrite(
        f"{output_path}/{sample.replace('.dcm', '.png')}",
        img,
    )

    # count += 1

    # if count >= 10:
    #     break
