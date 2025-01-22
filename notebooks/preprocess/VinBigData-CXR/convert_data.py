import os

from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_xray(path, voi_lut=True, fix_monochrome=True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
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


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    return im


image_id = []
dim0 = []
dim1 = []

root = "/research/d5/gds/yzhong22/datasets/multitask-moe/vinbigdata-cxr"

for split in ["train", "test"]:
    load_dir = os.path.join(root, split)
    save_dir = os.path.join(root, f"{split}_png")

    os.makedirs(save_dir, exist_ok=True)

    for file in tqdm(os.listdir(load_dir)):
        # set keep_ratio=True to have original aspect ratio
        xray = read_xray(load_dir + file)
        im = resize(xray, size=512, keep_ratio=True)
        im.save(save_dir + file.replace("dicom", "png"))

        if split == "train":
            image_id.append(file.replace(".dicom", ""))
            dim0.append(xray.shape[0])
            dim1.append(xray.shape[1])
