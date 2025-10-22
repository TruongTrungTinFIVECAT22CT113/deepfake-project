import torch

ckpt_path = r"H:\deepfake-project\deepfake-project\deepfake_detector\models\vitb384_512\checkpoints\detector_best_calib.pt"
names = ['Audio2Animation','Deepfakes','Face2Face','FaceShifter','FaceSwap','NeuralTextures','Video2VideoID']

ck = torch.load(ckpt_path, map_location='cpu')
ck.setdefault("meta", {})["method_names"] = names
torch.save(ck, ckpt_path)
print("✅ Patched method_names:", names)
