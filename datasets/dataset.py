import os
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

from PIL import Image
import pydicom
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.data import create_transform

import utils.constants as constants


class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, args, subsets, split):
        super().__init__()

        if not type(subsets) is list:
            subsets = [subsets]

        self.args = args
        self.subsets = subsets
        self.split = split
        self.target_disease = None

        df = pd.read_csv(args.metadata_path)
        if "chexpert" in subsets[0]:
            self.target_disease = subsets[0][9:]
            self.df = df.loc[(df["dataset"] == "chexpert") & (df["split"] == split)].reset_index(drop=True)
        else:
            self.df = df.loc[(df["dataset"].isin(subsets)) & (df["split"] == split)].reset_index(drop=True)
        # self.df = df.loc[(df["dataset"].isin(subsets)) & (df["split"] == split)].reset_index(drop=True)

        self.text_template = "{} in the image"
        self.modality = self.df.iloc[0]["modality"]
        self.classes, self.class_texts = self._build_classes(self.text_template)
        self.transform = build_transform(args, is_train=(split == "train"))

    def _build_classes(self, text_template):
        classes = set()
        for subset in self.subsets:
            classes.update(constants.CLASSES[subset])
        classes = sorted(classes)

        classes_dict = {}
        class_texts = []

        if "no findings" in classes:
            classes.remove("no findings")
            classes_dict["no findings"] = 0

            class_texts.append(
                [text_template.format(x) for x in ["abnormalities are present", "no findings are present"]]
            )

        for cls in classes:
            current_idx = len(classes_dict)
            classes_dict[cls] = current_idx

            class_texts.append([text_template.format(x) for x in [f"{cls} is not present", f"{cls} is present"]])

        return classes_dict, class_texts

    def get_balance_sample_weights(self):
        sample_count = dict(self.df["dataset"].value_counts())
        sample_weight = np.array([1.0 / sample_count[t] for t in self.df["dataset"].tolist()])

        return torch.from_numpy(sample_weight)

    def set_classes(self, classes, class_texts):
        self.classes = classes
        self.class_texts = class_texts

    def __len__(self):
        return len(self.df)

    def get_class_num(self):
        num_dict = {k: 0 for k in self.classes.keys()}

        for i in range(len(self.df)):
            item = self.df.iloc[i]
            class_raw = item["class"]

            classes = [x.strip() for x in class_raw.split(",")]

            for cls in classes:
                if cls in self.classes.keys():
                    num_dict[cls] +=1

        return num_dict


    def __getitem__(self, index):
        item = self.df.iloc[index]
        id, img_path, classes_raw, dataset = item["id"], item["path"], item["class"], item["dataset"]

        if any(img_path.endswith(x) for x in [".jpg", ".JPG", ".png", ".bmp", ".BMP", ".jpeg", ".tif", ".Jpg"]):
            img = Image.open(os.path.join(self.args.data_root, img_path)).convert("RGB")
        elif img_path.endswith(".dcm"):
            dicom = pydicom.dcmread(os.path.join(self.args.data_root, img_path))

            window = [-600 - 750, -600 + 750]
            intercept = dicom.RescaleIntercept if "RescaleIntercept" in dicom else -1024
            slope = dicom.RescaleSlope if "RescaleSlope" in dicom else 1

            img = np.clip(dicom.pixel_array * slope + intercept, window[0], window[1])

            img = (img - window[0]) / (window[1] - window[0]) * 255
            img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        else:
            raise NotImplementedError(f"Img format {os.path.basename(img_path)} is not supported.")

        img = self.transform(img)

        classes = [x.strip() for x in classes_raw.split(",")]
        label = torch.tensor([0] * len(self.classes), dtype=torch.int64)

        for cls in classes:
            if cls in self.classes.keys():
                label[self.classes[cls]] = 1

        return {
            "image": img,
            "class_text": self.class_texts,
            "label": label,
            "id": id,
            "dataset": dataset,
            "class_raw": classes_raw,
            "path": img_path,
        }


class SingleExpertDataset(torch.utils.data.Dataset):
    def __init__(self, args, subsets, split):
        super().__init__()

        if not type(subsets) is list:
            subsets = [subsets]

        assert len(subsets) == 1

        self.args = args
        self.subsets = subsets
        self.split = split

        df = pd.read_csv(args.metadata_path)
        if "chexpert" in subsets[0]:
            self.target_disease = subsets[0][9:]
            df = df.loc[(df["dataset"] == "chexpert") & (df["split"] == split)].reset_index(drop=True)
        else:
            df = df.loc[(df["dataset"].isin(subsets)) & (df["split"] == split)].reset_index(drop=True)
        df.drop_duplicates(subset=["id", "class"])
        self.df = df

        self.is_multi_label = "chexpert" in subsets[0]

        self.text_template = "An image of {}"
        self.modality = self.df.iloc[0]["modality"]
        self.classes, self.class_texts = self._build_classes(self.text_template)
        self.transform = build_transform(args, is_train=(split == "train"))

    def _build_classes(self, text_template):
        classes = set()
        for subset in self.subsets:
            classes.update(constants.CLASSES[subset])
        classes = sorted(classes)

        classes_dict = {}
        class_texts = []

        if "no findings" in classes:
            classes.remove("no findings")
            # if not multi label, no findings will be used as the negative prompt
            if not self.is_multi_label:
                classes = ["no findings"] + classes
                classes_dict["no findings"] = 0
            else:
                assert len(classes) == 1, f"For multi label dataset, expected one posivite class, while got {classes}"
                assert classes == [
                    self.target_disease
                ], f"For multi label dataset, expected target disease {self.target_disease}, while got {classes}"
                classes = [f"no {self.target_disease} found"] + classes
                classes_dict[f"no {self.target_disease} found"] = 0

            class_texts.append([text_template.format(x) for x in ["diseased", "no findings"]])

        for cls in classes:
            if cls in classes_dict.keys():
                continue
            current_idx = len(classes_dict)
            classes_dict[cls] = current_idx

        class_texts.append([text_template.format(x) for x in classes])

        return classes_dict, class_texts

    def get_balance_sample_weights(self):
        sample_count = dict(self.df["dataset"].value_counts())
        sample_weight = np.array([1.0 / sample_count[t] for t in self.df["dataset"].tolist()])

        return torch.from_numpy(sample_weight)

    def set_classes(self, classes, class_texts):
        self.classes = classes
        self.class_texts = class_texts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        id, img_path, class_raw, subset = item["id"], item["path"], item["class"], item["dataset"]
        assert all(
            x.strip() in constants.CLASSES[subset] for x in class_raw.split(",")
        ), f"Subset {subset} contains {constants.CLASSES[subset]}, while got {class_raw}"

        if any(img_path.endswith(x) for x in [".jpg", ".JPG", ".png", ".bmp", ".BMP", ".jpeg", ".tif", ".Jpg"]):
            img = Image.open(os.path.join(self.args.data_root, img_path)).convert("RGB")
        elif img_path.endswith(".dcm"):
            dicom = pydicom.dcmread(os.path.join(self.args.data_root, img_path))

            window = [-600 - 750, -600 + 750]
            intercept = dicom.RescaleIntercept if "RescaleIntercept" in dicom else -1024
            slope = dicom.RescaleSlope if "RescaleSlope" in dicom else 1

            img = np.clip(dicom.pixel_array * slope + intercept, window[0], window[1])

            img = (img - window[0]) / (window[1] - window[0]) * 255
            img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        else:
            raise NotImplementedError(f"Img format {os.path.basename(img_path)} is not supported.")

        img = self.transform(img)

        label = torch.tensor([0] * 2, dtype=torch.int64)

        if class_raw == "no findings":
            label[0] = 1
        if self.is_multi_label:
            if self.target_disease in class_raw:
                label[-1] = 1
        else:
            label[-1] = self.classes[class_raw]

        return {
            "image": img,
            "class_text": self.class_texts,
            "label": label,
            "id": id,
            "dataset": subset,
            "class_raw": class_raw,
            "path": img_path,
        }


def build_transform(args, is_train):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # transform = create_transform(
        #     input_size=args.input_size,
        #     is_training=True,
        #     color_jitter=args.color_jitter,
        #     auto_augment=args.aa,
        #     interpolation=args.train_interpolation,
        #     re_prob=args.reprob,
        #     re_mode=args.remode,
        #     re_count=args.recount,
        #     mean=OPENAI_CLIP_MEAN,
        #     std=OPENAI_CLIP_STD,
        # )
        transform = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize(
                    (args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
            ]
        )
        return transform
    else:
        return transforms.Compose(
            [
                transforms.Resize(
                    (args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
            ],
        )
