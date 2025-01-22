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
--experts rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--eval_subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--ckpt_dir /research/d5/gds/yzhong22/experiments/multitask-moe/biomedclip/seed0/lp \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--save_pred True


# zeroshot
CUDA_LAUNCH_BLOCKING=1 python eval_adaptation.py \
--backbone biomedclip \
--batch_size 128 \
--eval_subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--ckpt_dir /research/d5/gds/yzhong22/experiments/multitask-moe/biomedclip/seed0/lp/zero-shot \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--save_pred True \
--use_amp True