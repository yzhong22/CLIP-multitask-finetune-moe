import pandas as pd
import numpy as np
import glob
import os
import tqdm

root = "/research/d5/gds/yzhong22/datasets/multitask-moe/CC-CCII"

df = pd.read_csv(os.path.join(root, "unzip_filenames.csv"))

error_rows = []

for i in tqdm.tqdm(range(len(df))):
    row = df.iloc[i]

    zip_file, label, patient_id, scan_id, n_slice = (
        row["zip_file"],
        row["label"],
        str(row["patient_id"]),
        str(row["scan_id"]),
        int(row["n_slice"]),
    )

    path = os.path.join(root, "images", label, patient_id, scan_id)

    if not os.path.exists(path):
        error_rows.append(i)
        continue

    if len(glob.glob(os.path.join(path, "*"))) != n_slice:
        error_rows.append(i)
        continue

print(error_rows)

if len(error_rows) > 0:
    df_missing = df.iloc[error_rows]
    df_missing.to_csv(os.path.join(root, "missing_download.csv"), index=False)
