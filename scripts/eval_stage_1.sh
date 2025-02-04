CUDA_LAUNCH_BLOCKING=1 python eval_adaptation.py \
--backbone biomedclip \
--batch_size 128 \
--residual_scale 0.6 \
--experts rsna-pulmonary-embolism \
--eval_subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--ckpt_dir /research/d5/gds/yzhong22/experiments/multitask-moe/biomedclip/seed0/lp \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--save_pred True


CUDA_LAUNCH_BLOCKING=1 python eval_adaptation.py \
--backbone biomedclip \
--batch_size 128 \
--method lp \
--experts rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--eval_subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx,mimic-cxr,vinbigdata-cxr,luna16,covid-ct-md \
--ckpt_dir /research/d5/gds/yzhong22/experiments/multitask-moe/biomedclip/seed0/ \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--save_pred True &&

CUDA_LAUNCH_BLOCKING=1 python eval_adaptation.py \
--backbone biomedclip \
--batch_size 128 \
--method r-adapter \
--experts rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--eval_subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx,mimic-cxr,vinbigdata-cxr,luna16,covid-ct-md \
--ckpt_dir /research/d5/gds/yzhong22/experiments/multitask-moe/biomedclip/seed0/ \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--save_pred True &&

CUDA_LAUNCH_BLOCKING=1 python eval_adaptation.py \
--backbone biomedclip \
--batch_size 128 \
--method taskres \
--experts rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--eval_subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx,mimic-cxr,luna16,covid-ct-md \
--ckpt_dir /research/d5/gds/yzhong22/experiments/multitask-moe/biomedclip/seed0/ \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--save_pred True


CUDA_LAUNCH_BLOCKING=1 python eval_adaptation.py \
--backbone biomedclip \
--batch_size 128 \
--method wise-ft \
--experts rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--eval_subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx,mimic-cxr,luna16,covid-ct-md \
--ckpt_dir /research/d5/gds/yzhong22/experiments/multitask-moe/biomedclip/seed0/ \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--save_pred True

CUDA_LAUNCH_BLOCKING=1 python eval_adaptation.py \
--backbone biomedclip \
--batch_size 128 \
--method ft \
--experts rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--eval_subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx,mimic-cxr,luna16,covid-ct-md \
--ckpt_dir /research/d5/gds/yzhong22/experiments/multitask-moe/biomedclip/seed0/ \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--save_pred True


# zeroshot
CUDA_LAUNCH_BLOCKING=1 python eval_adaptation.py \
--backbone biomedclip \
--batch_size 128 \
--eval_subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx,mimic-cxr,vinbigdata-cxr \
--ckpt_dir /research/d5/gds/yzhong22/experiments/multitask-moe/biomedclip/seed0/zero-shot \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--save_pred True \
--use_amp True