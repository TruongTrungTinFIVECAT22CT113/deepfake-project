# file readme thư mục deepfake-project\deepfake_detector

# 1. Lệnh train cho Detector. 
py -m deepfake_detector.src.train --epochs 30 --freeze_epochs 0 --batch_size 128 --micro_batch 128 --ema --model vit_base_patch16_224 --img_size 480 --mixup 0.05 --cutmix 0.05 --mixup_prob 0.5 --label_smoothing 0.05 --lr 3e-4 --warmup_steps 8715 --balance_by_method --face_crop --method_boost "Face2Face=2.5,FaceShifter=2.5,NeuralTextures=2.5,FaceSwap=2.5"

# 2. Tiếp tục từ lần train trước đã tạm dừng.
# (tự tìm checkpoint tốt nhất trong thư mục checkpoints/).
py -m deepfake_detector.src.train --epochs 30 --freeze_epochs 0 --batch_size 128 --micro_batch 128 --ema --model vit_base_patch16_224 --img_size 480 --mixup 0.05 --cutmix 0.05 --mixup_prob 0.5 --label_smoothing 0.05 --lr 3e-4 --warmup_steps 8715 --balance_by_method --face_crop --method_boost "Face2Face=2.5,FaceShifter=2.5,NeuralTextures=2.5,FaceSwap=2.5" --auto_resume

# Hoặc chỉ định checkpoint cụ thể ( XX là STT chỉ định ).
py -m deepfake_detector.src.train --epochs 30 --freeze_epochs 0 --batch_size 128 --micro_batch 128 --ema --model vit_base_patch16_224 --img_size 480 --mixup 0.05 --cutmix 0.05 --mixup_prob 0.5 --label_smoothing 0.05 --lr 3e-4 --warmup_steps 8715 --balance_by_method --face_crop --method_boost "Face2Face=2.5,FaceShifter=2.5,NeuralTextures=2.5,FaceSwap=2.5" --resume deepfake_detector/checkpoints/detector_epochXX.pt

# 3. Đánh giá mô hình tốt nhất.
py -m deepfake_detector.eval.eval_val --ckpt deepfake_detector/checkpoints/detector_best.pt --batch 64 --workers 4 --tta 2 --face_crop

# 4. Tạo hiệu chuẩn ngưỡng mô hình tốt nhất, để tạo file detector_best_calib.
py -m deepfake_detector.eval.eval_val --ckpt deepfake_detector/checkpoints/detector_best.pt --batch 64 --workers 4 --tta 2 --face_crop --calibrate --out_ckpt deepfake_detector/checkpoints/detector_best_calib.pt

# 5. Đánh giá mô hình tốt nhất đã calib, theo từng phương pháp và báo cáo chi tiết.
py -m deepfake_detector.eval.eval_by_method --ckpt deepfake_detector/checkpoints/detector_best_calib.pt --batch 64 --workers 4 --tta 2 --face_crop

# 6. Chi tiết kiểm tra mô hình tốt nhất đã calib, cho từng folder method (đếm đúng/sai/nhầm, xuất CSV).
py -m deepfake_detector.eval.eval_method_detailed --ckpt deepfake_detector/checkpoints/detector_best_calib.pt --root data/processed/faces/val/fake --tta 2 --face_crop