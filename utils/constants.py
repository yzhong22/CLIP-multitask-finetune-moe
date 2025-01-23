# data
# TRAIN_DATASETS = ["rsna-pulmonary-embolism", "chexpert", "lung-pet-ct-dx", "CC-CCII", "ssim-covid19"]
TRAIN_DATASETS = [
    "rsna-pulmonary-embolism",
    "chexpert-cardiomegaly",
    "chexpert-edema",
    "chexpert-pleural effusion",
    "chexpert-consolidation",
    # "chexpert-pneumonia",
    "chexpert-atelectasis",
    "chexpert-pneumothorax",
    "chexpert",
    "lung-pet-ct-dx",
    "CC-CCII",
    "ssim-covid19",
]
OOD_DATASETS = [
    "mimic-cxr",
]
ZERO_SHOT_DATASETS = ["vinbigdata-cxr"]
DATASETS = TRAIN_DATASETS + OOD_DATASETS + ZERO_SHOT_DATASETS

CLASSES = {
    "rsna-pulmonary-embolism": ["no findings", "pulmonary embolism"],
    "chexpert": [
        "no findings",
        "cardiomegaly",
        "edema",
        "pleural effusion",
        "consolidation",
        # "pneumonia",
        "atelectasis",
        "pneumothorax",
    ],
    "chexpert-cardiomegaly": ["no findings", "cardiomegaly"],
    "chexpert-edema": ["no findings", "edema"],
    "chexpert-pleural effusion": ["no findings", "pleural effusion"],
    "chexpert-consolidation": ["no findings", "consolidation"],
    # "chexpert-pneumonia": ["no findings", "pneumonia"],
    "chexpert-atelectasis": ["no findings", "atelectasis"],
    "chexpert-pneumothorax": ["no findings", "pneumothorax"],
    "lung-pet-ct-dx": [
        "no findings",
        "adenocarcinoma",
        "squamous cell carcinoma",
        "small cell carcinoma",
        "large cell carcinoma",
    ],
    "CC-CCII": ["no findings", "typical appearance pneumonia", "atypical appearance pneumonia"],
    "ssim-covid19": [
        "no findings",
        "typical appearance pneumonia",
        "indeterminate appearance pneumonia",
        "atypical appearance pneumonia",
    ],
    "mimic-cxr": [
        "no findings",
        "cardiomegaly",
        "edema",
        "pleural effusion",
        "consolidation",
        # "pneumonia",
        "atelectasis",
        "pneumothorax",
    ],
    "vinbigdata-cxr": [
        "no findings",
        "cardiomegaly",
        "pleural effusion",
        "pleural thickening",
        "aortic enlargement",
        "pulmonary fibrosis",
        "ild",
        "nodule/mass",
        "other lesion",
        "lung opacity",
        "infiltration",
        "consolidation",
        "calcification",
        "atelectasis",
        "pneumothorax",
    ],
}
