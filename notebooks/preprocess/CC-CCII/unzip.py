import numpy as np
import os
import tqdm
import pandas as pd
import glob
import zipfile

from urllib.request import urlretrieve

root = "/research/d5/gds/yzhong22/datasets/multitask-moe/CC-CCII"

df = pd.read_csv(os.path.join(root, "unzip_filenames.csv"))

# files = df["zip_file"].unique()

files = [x.split("/")[-1] for x in glob.glob(os.path.join(root, "images-zip-3", "*.zip"))]

print(files)

error_list = []

for file in tqdm.tqdm(files):
    if "NCP" in file:
        file = file.replace("NCP-", "COVID19-")

    try:
        with zipfile.ZipFile(os.path.join(root, "images-zip-3", file), "r") as f:
            f.extractall(os.path.join(root, "images"))
    except:
        error_list.append(file)

print(error_list)
