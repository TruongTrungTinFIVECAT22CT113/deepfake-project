import cv2
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--resolution", type=str, required=True,
                help="VD: 640x480 | 960x720 | 1024x768")
args = ap.parse_args()

base = rf"H:\deepfake-project\deepfake-project\data\videos_test\{args.resolution}"
base_fake = os.path.join(base, "fake")
base_real = os.path.join(base, "real", "Faceforensics")

fake_folders = ["Audio2Animation","Deepfakes","Face2Face",
                "FaceShifter","FaceSwap","NeuralTextures","Video2VideoID"]

print(f"Kiểm tra resolution trong: {base}")
print(f"Sample video: 900.mp4")
print("="*55)

all_ok = True

for folder in fake_folders:
    path = os.path.join(base_fake, folder, "900.mp4")
    if not os.path.exists(path):
        print(f"{folder:20s}: FILE KHÔNG TỒN TẠI")
        all_ok = False
        continue
    cap = cv2.VideoCapture(path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    expected = args.resolution.lower()
    actual   = f"{w}x{h}"
    match    = "✓" if actual == expected else f"⚠ KHÁC (expected {expected})"
    print(f"{folder:20s}: {actual}  @ {fps:.2f} fps  {match}")
    if actual != expected:
        all_ok = False

# Real
real_path = os.path.join(base_real, "900.mp4")
if not os.path.exists(real_path):
    print(f"{'Real (Faceforensics)':20s}: FILE KHÔNG TỒN TẠI")
    all_ok = False
else:
    cap = cv2.VideoCapture(real_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    expected = args.resolution.lower()
    actual   = f"{w}x{h}"
    match    = "✓" if actual == expected else f"⚠ KHÁC (expected {expected})"
    print(f"{'Real (Faceforensics)':20s}: {actual}  @ {fps:.2f} fps  {match}")
    if actual != expected:
        all_ok = False

print("="*55)
print(f"Kết quả: {'✓ Tất cả đúng resolution' if all_ok else '⚠ Có video không đúng resolution'}")