# file readme thư mục deepfake-project\shared

# 1. Kiểm tra môi trường.
env_check.py  : kiểm tra nhanh phiên bản Python, PyTorch, GPU, thư viện.
## Câu lệnh:  
py -m shared.env_check 

# 2. Tách 10 frames từ mỗi video.
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

preprocess.py : chuẩn hoá dữ liệu từ video đầu vào thành ảnh .jpg về cấu trúc data/processed/faces/...
## Câu lệnh:                     
py -m shared.preprocess --data_root data/videos --out_root data/processed/faces --frame_every 10 --val_split 0.2

> Gợi ý: Với video dài hoặc nhiều, tăng --frame_every để giảm số ảnh.