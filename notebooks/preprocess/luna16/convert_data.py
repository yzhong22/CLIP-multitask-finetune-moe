import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import cv2

root = "/research/d5/gds/yzhong22/datasets/multitask-moe/luna16"

df_annotation = pd.read_csv(os.path.join(root, "annotations.csv"))

path_dict = {"seriesuid": [], "path": []}


for file in glob.glob(f"{root}/subset*/subset*/*.mhd"):
    path_dict["seriesuid"].append(os.path.basename(file)[:-4])
    path_dict["path"].append(file.replace(root, "."))

df_path = pd.DataFrame.from_dict(path_dict)


def xyz2irc(xyz_coord, xyz_origin, xyz_sizes, irc_transform_mat):
    """
    Map the patient coordidates to the voxel-based coordinates through data included in the meta-data file of CT.

    Parameters:
        - xyz_coord (XYZ_tuple): the patient coordinates to be transformed.
        - xyz_origin (XYZ_tuple): the exact origin point in the patient space for reference.
        - xyz_sizes (XYZ_tuple): the voxel size for scaling purposes.
        - irc_transform_mat (np.array): the transformation matrix between the two spaces.

    return:
        transformed coordinate as IRC_tuple
    """
    coordinate_xyz = np.array(xyz_coord)
    physical_origin = np.array(xyz_origin)
    physical_sizes = np.array(xyz_sizes)
    # reverse the computations
    cri_coord = ((coordinate_xyz - physical_origin) @ np.linalg.inv(irc_transform_mat)) / physical_sizes
    rounded_cri_coord = np.round(cri_coord).astype(int)
    return rounded_cri_coord[::-1]


dict_out = {"seriesid": [], "id": [], "path": [], "is_node": [], "slice (from 0)": []}

window = [-600 - 1500 // 2, -600 + 1500 // 2]


for i in tqdm(range(len(df_annotation))):
    item = df_annotation.iloc[i]

    series_id, x, y, z, diameter = item

    if not series_id in df_path["seriesuid"].tolist():
        continue

    path = df_path.loc[df_path["seriesuid"] == series_id]["path"].tolist()[0]

    ct_mhd = sitk.ReadImage(os.path.join(root, path))
    ct_arr = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32).clip(window[0], window[1])

    seg_path = os.path.join(root, "seg-lungs-LUNA16", "seg-lungs-LUNA16", f"{series_id}.mhd")

    seg = np.array(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)))

    assert ct_arr.shape == seg.shape

    origin = ct_mhd.GetOrigin()
    spacing = ct_mhd.GetSpacing()
    transform_mat = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    irc = xyz2irc((x, y, z), origin, spacing, transform_mat)

    diameter_scaled = diameter / spacing[-1]

    output_path = os.path.join(root, "images", series_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for idx in range(ct_arr.shape[0]):
        if np.max(seg[idx]) == 0:
            continue

        img = ct_arr[idx] - window[0]
        H, W = img.shape
        img = img / (window[1] - window[0])
        img = (img * 255).astype(np.uint8)

        scale = 320 / min(H, W)
        img = cv2.resize(img, (int(W * scale + 0.5), int(H * scale + 0.5)))

        is_node = (idx >= irc[0] - int(diameter_scaled / 2 + 1)) & (idx <= irc[0] + int(diameter_scaled / 2 + 1))

        img_path = os.path.join(output_path, f"{idx}.png")
        cv2.imwrite(
            img_path,
            img,
        )

        dict_out["seriesid"].append(series_id)
        dict_out["id"].append(f"{series_id}/{idx}")
        dict_out["path"].append(img_path)
        dict_out["is_node"].append(int(is_node))
        dict_out["slice (from 0)"].append(idx)

df_out = pd.DataFrame.from_dict(dict_out)
df_out.to_csv(os.path.join(root, "annotations_2d.csv"), index=False)
