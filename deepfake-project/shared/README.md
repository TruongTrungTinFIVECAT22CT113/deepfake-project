# file readme thư mục deepfake-project\shared

# 1. Kiểm tra môi trường.
env_check.py  : kiểm tra nhanh phiên bản Python, PyTorch, GPU, thư viện.
## Câu lệnh:  
python -m shared.env_check 

# 2. Tách frames từ mỗi video.
## Câu lệnh:                     
python preprocess_balanced.py --data_root H:\Deepfake\deepfake-project\data\videos_h264 --out_root  H:\Deepfake\deepfake-project\data\processed_multi --fps 25 --seg_len 0.5 --img_size 512 --frames_per_face 64 --frames_per_head 64 --frames_per_full 64