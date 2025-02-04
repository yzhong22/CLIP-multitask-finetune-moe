import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import glob
import pydicom
import SimpleITK as sitk
import sys
import glob
from PIL import Image

# read metadata
path = "/research/d5/gds/yzhong22/datasets/multitask-moe/COVID-CT-MD/"

demo_data = pd.read_csv(path + "Clinical-data.csv")


def read_resort_dcm(folder):
    files = []
    # print(f"glob: {sys.argv[1]}")
    for fname in glob.glob(os.path.join(folder, "*.dcm"), recursive=False):
        # print(f"loading: {fname}")
        files.append(pydicom.dcmread(fname))

    # print(f"file count: {len(files)}")

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, "SliceLocation"):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    if skipcount > 0:
        print(f"folder {folder}: skipped, no SliceLocation: {skipcount}")

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation, reverse=True)

    return slices


def save_slices(slices, out_folder, window=[-600 - 1500 // 2, -600 + 1500 // 2]):
    for i, ds in enumerate(slices):
        array = ds.pixel_array.astype(float)
        array = array * ds.RescaleSlope + ds.RescaleIntercept
        # ds[ds < window[0]] = window[0]
        # ds[ds > window[1]] = window[1]
        array = np.clip(array, window[0], window[1])
        array = (array - window[0]) / (window[1] - window[0]) * 255
        img = Image.fromarray(array.astype(np.uint8))

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        img.save(os.path.join(out_folder, f"IM{str(i+1).zfill(4)}.png"))


for d, f in zip(demo_data["Diagnosis"].values, demo_data["Folder"].values):
    if d == "CAP":
        d = "Cap"
    slices = read_resort_dcm(os.path.join(path, f"{d} Cases", f))
    save_slices(slices, os.path.join(path, "slices", f"{d} Cases", f))
    # break
