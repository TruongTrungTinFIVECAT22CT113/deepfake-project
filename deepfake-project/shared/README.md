# file readme thư mục deepfake-project\shared

# 1. Kiểm tra môi trường.
env_check.py  : kiểm tra nhanh phiên bản Python, PyTorch, GPU, thư viện.
## Câu lệnh:  
python -m shared.env_check 

# 2. Tách frames từ mỗi video.
## Cấu trúc dữ liệu (đã chuẩn hoá).
data/processed/faces/
                 ├─ train/
                 |    ├─ fake/
                 |    │   ├─ Deepfakes/  Face2Face/  FaceShifter/  FaceSwap/  NeuralTextures/
                 |    └─ real/
                 └─ val/
                     ├─ fake/
                     │   ├─ Deepfakes/  Face2Face/  FaceShifter/  FaceSwap/  NeuralTextures/
                     └─ real/
preprocess_balanced.py: cân bằng dữ liệu. Chuẩn hoá dữ liệu từ video đầu vào thành ảnh .jpg về cấu trúc data/processed/faces/...
## Câu lệnh:                     
py -m shared.preprocess_balanced --data_root data/videos --out_root  data/processed/faces --val_split 0.2 --methods Deepfakes Face2Face FaceShifter FaceSwap NeuralTextures --real_dir original --exclude DeepFakeDetection --mode segments --fps 30 --segments 5 --train_segments 5 --val_segments 2 --seg_len 1.0 --margin 0.20 --strategy uniform --seed 42