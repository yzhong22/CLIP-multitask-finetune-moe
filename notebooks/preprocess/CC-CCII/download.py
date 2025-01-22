import numpy as np
import os
import tqdm
import pandas as pd

from urllib.request import urlretrieve

root = "/research/d5/gds/yzhong22/datasets/multitask-moe/CC-CCII"

df = pd.read_csv(os.path.join(root, "unzip_filenames.csv"))

files = df["zip_file"].unique()


def reduce_down(url, filename, num_count=0, num_repeat=5):
    if num_count >= num_repeat:
        return

    try:
        urlretrieve(url, filename)
    except:
        reduce_down(url, filename, num_count + 1)


print(files)

for file in tqdm.tqdm(files):
    if "NCP" in file:
        file = file.replace("NCP-", "COVID19-")

    reduce_down(f"https://download.cncb.ac.cn/covid-ct/{file}", os.path.join(root, "images-zip", file))

    # break
