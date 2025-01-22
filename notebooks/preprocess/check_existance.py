import os
import glob
from tqdm import tqdm

import pandas as pd

root = "/research/d5/gds/yzhong22/datasets/multitask-moe/"
df = pd.read_csv(os.path.join(root, "metadata.csv"))

missing_files = []

for file in tqdm(df["path"].to_list()):
    if not os.path.isfile(os.path.join(root, file)):
        missing_files.append(file)
        print(f"missing {file}")

if len(missing_files) == 0:
    print("All files are downlowded successfully!")
else:
    with open(os.path.join(root, "missing_files.txt"), "r") as fp:
        fp.writelines(missing_files)
