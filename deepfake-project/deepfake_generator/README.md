# 4. Lệnh train cho Generator. (thay X bằng số thứ tự epoch cuối cùng trong checkpoint)
py -m deepfake_generator.src.train --data_root data/processed/faces --epochs 2 --max_steps 2000 --batch_size 2 --grad_accum 4 --rank 16 --save_every_steps 500

py -m deepfake_generator.src.train --resume deepfake_generator/outputs/generic_lora/checkpoints/lora_epochX.pt 