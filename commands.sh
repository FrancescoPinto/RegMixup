#cross-entropy training
python ./train.py --default_root_dir "./r50_c10" --model_type vanilla --backbone resnet50_vanilla --dataset cifar10 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./train.py --default_root_dir "./r50_c100" --model_type vanilla --backbone resnet50_vanilla --dataset cifar100 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./train.py --default_root_dir "./wrn_c10" --model_type vanilla --backbone wideresnet_vanilla --dataset cifar10 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./train.py --default_root_dir "./wrn_c100" --model_type vanilla --backbone wideresnet_vanilla --dataset cifar100 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn

#RegMixup training with default parameters
python ./train.py --default_root_dir "./regmixup_r50_c10" --model_type regmixup --mixup_alpha 20 --mixup_beta 20 --backbone resnet50_vanilla --dataset cifar10 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./train.py --default_root_dir "./regmixup_r50_c100" --model_type regmixup --mixup_alpha 10 --mixup_beta 10 --backbone resnet50_vanilla --dataset cifar100 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./train.py --default_root_dir "./regmixup_wrn_c10" --model_type regmixup --mixup_alpha 20 --mixup_beta 20 --backbone wideresnet_vanilla --dataset cifar10 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./train.py --default_root_dir "./regmixup_wrn_c100" --model_type regmixup --mixup_alpha 10 --mixup_beta 10 --backbone wideresnet_vanilla --dataset cifar100 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn

#cross-entropy test
python ./test.py --resume_training --default_root_dir "./r50_c10"  --model_type vanilla --backbone resnet50_vanilla --dataset cifar10 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./test.py --resume_training --default_root_dir "./r50_c100"  --model_type vanilla --backbone resnet50_vanilla --dataset cifar100 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./test.py --resume_training --default_root_dir "./wrn_c10"  --model_type vanilla --backbone wideresnet_vanilla --dataset cifar10 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./test.py --resume_training --default_root_dir "./wrn_c100"  --model_type vanilla --backbone wideresnet_vanilla --dataset cifar100 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn

#RegMixup test with default parameters
python ./test.py --resume_training --default_root_dir "./regmixup_r50_c10" --model_type regmixup --mixup_alpha 20 --mixup_beta 20 --backbone resnet50_vanilla --dataset cifar10 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./test.py --resume_training --default_root_dir "./regmixup_r50_c100" --model_type regmixup --mixup_alpha 10 --mixup_beta 10 --backbone resnet50_vanilla --dataset cifar100 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./test.py --resume_training --default_root_dir "./regmixup_wrn_c10" --model_type regmixup --mixup_alpha 20 --mixup_beta 20 --backbone wideresnet_vanilla --dataset cifar10 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn
python ./test.py --resume_training --default_root_dir "./regmixup_wrn_c100" --model_type regmixup --mixup_alpha 10 --mixup_beta 10 --backbone wideresnet_vanilla --dataset cifar100 --max_epochs 350 --lr 0.1 --gpus 1 -b 128 --seed 0 --accelerator dp --num_sanity_val_steps 0 --network_category cnn


