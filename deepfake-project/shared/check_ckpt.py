# tools/check_ckpt.py
import os, sys, torch, pprint

ckpt_path = r"H:\deepfake-project\backend\models\vitb384_512\detector_best.pt"
if len(sys.argv) > 1:
    ckpt_path = sys.argv[1]

print(">>> ckpt:", ckpt_path)
ck = torch.load(ckpt_path, map_location="cpu")

print(">>> top-level keys:", list(ck.keys())[:10])

meta_top = ck.get("meta", None)
meta_in_model = None
if isinstance(ck.get("model", None), dict):
    meta_in_model = ck["model"].get("meta", None)

print(">>> meta at top-level:", None if meta_top is None else meta_top.keys())
print(">>> meta inside model:", None if meta_in_model is None else meta_in_model.keys())

names = None
if isinstance(meta_top, dict) and "method_names" in meta_top:
    names = meta_top["method_names"]
elif isinstance(meta_in_model, dict) and "method_names" in meta_in_model:
    names = meta_in_model["method_names"]

print(">>> method_names:", names)

# Đếm số lớp method từ state_dict (mọi pattern head_m(.k)?.weight)
import re
state = ck.get("model", ck)
n_methods = None
pat = re.compile(r"^(module\.)?head_(m|met|mth|method)(\.\d+)?\.weight$")
for k,v in state.items():
    if pat.match(k) and hasattr(v, "shape"):
        n_methods = int(v.shape[0]); break
print(">>> detected n_methods from weight:", n_methods)
