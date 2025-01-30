CUDA_LAUNCH_BLOCKING=1 python train_experts.py \
--backbone biomedclip \
--batch_size 128 \
--blr 2e-4 \
--epochs 30 \
--warmup_epochs 5 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--subsets lung-pet-ct-dx \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--output_dir /research/d5/gds/yzhong22/experiments/multitask-moe



CUDA_LAUNCH_BLOCKING=1 python train_adaptation.py \
--backbone biomedclip \
--batch_size 128 \
--blr 2e-4 \
--residual_scale 0.6 \
--epochs 10 \
--warmup_epochs 2 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--output_dir /research/d5/gds/yzhong22/experiments/multitask-moe \
--balance_subset True \
--use_amp True

CUDA_LAUNCH_BLOCKING=1 python train_adaptation.py \
--backbone biomedclip \
--peft True \
--method r-adapter \
--batch_size 128 \
--blr 2e-4 \
--residual_scale 0.6 \
--epochs 10 \
--warmup_epochs 2 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--output_dir /research/d5/gds/yzhong22/experiments/multitask-moe \
--balance_subset True \
--use_amp True

CUDA_LAUNCH_BLOCKING=1 python train_adaptation.py \
--backbone biomedclip \
--method cocoop \
--gradient_accumulation_steps 1 \
--batch_size 8 \
--blr 2e-4 \
--residual_scale 0.6 \
--epochs 10 \
--warmup_epochs 1 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--output_dir /research/d5/gds/yzhong22/experiments/multitask-moe \
--balance_subset True \
--use_amp True

CUDA_LAUNCH_BLOCKING=1 python train_adaptation.py \
--backbone biomedclip \
--method taskres \
--gradient_accumulation_steps 1 \
--batch_size 128 \
--blr 2e-4 \
--residual_scale 0.6 \
--epochs 10 \
--warmup_epochs 1 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--subsets rsna-pulmonary-embolism,chexpert,CC-CCII,ssim-covid19,lung-pet-ct-dx \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--output_dir /research/d5/gds/yzhong22/experiments/multitask-moe \
--balance_subset True \
--use_amp True


CUDA_LAUNCH_BLOCKING=1 python train_adaptation.py \
--backbone biomedclip \
--batch_size 128 \
--blr 2e-4 \
--epochs 10 \
--warmup_epochs 2 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--subsets rsna-pulmonary-embolism,chexpert,CC-CCII,lung-pet-ct-dx \
--input_size 224 \
--data_root /research/d5/gds/yzhong22/datasets/multitask-moe \
--output_dir /research/d5/gds/yzhong22/experiments/multitask-moe \
--balance_subset True \
--use_amp True