# -*- coding: utf-8 -*-
"""
Cách gọi:
python -m deepfake_detector.src.sweep_ensemble_thr_spatial ^
  --data_root data/processed_multi ^
  --ckpt deepfake_detector/models/vitb384_spatial/checkpoints/detector_best.pt ^
        deepfake_detector/models/convnextb384_opt/checkpoints/detector_best.pt ^
  --thr_min 0.0 --thr_max 1.0 --thr_steps 201 ^
  --val_batch 64 --workers 6 ^
  --val_tta scale --val_repeat 1 ^
  --out_json deepfake_detector/models/ensemble_vit_conv_thr.json
"""

import os
import json
import argparse
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import lại dataset/model/const từ train_spatial để nhất quán với lúc train
from src.train_spatial import (
    MultiBranchDataset,
    MultiHeadViT,
    IMAGENET_MEAN,
    IMAGENET_STD,
)  # :contentReference[oaicite:0]{index=0}


# ----------------- Helpers đọc ckpt giống backend_eval -----------------
def _infer_head_sizes_from_ckpt_state(ckpt_model_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    sizes = {"num_methods": 0, "num_face_classes": 1, "num_head_classes": 1, "num_full_classes": 1}
    if "head_met.1.weight" in ckpt_model_state:
        sizes["num_methods"] = ckpt_model_state["head_met.1.weight"].shape[0]
    if "head_face.1.weight" in ckpt_model_state:
        sizes["num_face_classes"] = ckpt_model_state["head_face.1.weight"].shape[0]
    if "head_head.1.weight" in ckpt_model_state:
        sizes["num_head_classes"] = ckpt_model_state["head_head.1.weight"].shape[0]
    if "head_full.1.weight" in ckpt_model_state:
        sizes["num_full_classes"] = ckpt_model_state["head_full.1.weight"].shape[0]
    return sizes


def _filter_state_dict_by_shape(dst_state: Dict[str, torch.Tensor], src_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in src_state.items() if k in dst_state and dst_state[k].shape == v.shape}


def load_checkpoint_build_model(
    ckpt_path: str,
    device: torch.device,
    img_size_arg: Optional[int] = None,
    model_name_arg: Optional[str] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Load checkpoint detector_best.pt được train bởi train_spatial.py
    Trả về: (model, meta)
    - model: MultiHeadViT đã load weight (dùng EMA nếu có)
    - meta: dict chứa ít nhất: model_name, img_size, method_names, best_thr (nếu có)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    model_state = ckpt.get("model", {})
    if not model_state:
        raise RuntimeError(f"Checkpoint thiếu key 'model': {ckpt_path}")

    head_sizes = _infer_head_sizes_from_ckpt_state(model_state)
    num_methods = head_sizes["num_methods"] or len(meta.get("method_names", [])) or 7
    num_face_classes = head_sizes["num_face_classes"]
    num_head_classes = head_sizes["num_head_classes"]
    num_full_classes = head_sizes["num_full_classes"]

    model_name = model_name_arg or meta.get("model_name") or meta.get("backbone_model") or "vit_base_patch16_384"
    img_size = img_size_arg or meta.get("img_size", 384)

    model = MultiHeadViT(
        model_name=model_name,
        img_size=img_size,
        num_methods=num_methods,
        num_face_classes=num_face_classes,
        num_head_classes=num_head_classes,
        num_full_classes=num_full_classes,
        drop_rate=0.0,
        drop_path_rate=0.0,
    ).to(device)

    dst = model.state_dict()
    # Nếu ckpt có EMA thì ưu tiên load EMA rồi mới đè phần thiếu bằng model_state
    if "ema" in ckpt and ckpt["ema"]:
        ema_state = ckpt["ema"]
        dst.update(_filter_state_dict_by_shape(dst, ema_state))
        dst.update(_filter_state_dict_by_shape(dst, model_state))
        model.load_state_dict(dst, strict=False)
    else:
        dst.update(_filter_state_dict_by_shape(dst, model_state))
        model.load_state_dict(dst, strict=False)

    model.eval()

    if not meta.get("method_names"):
        meta["method_names"] = [f"method_{i}" for i in range(num_methods)]
    if "img_size" not in meta:
        meta["img_size"] = img_size
    if "model_name" not in meta:
        meta["model_name"] = model_name
    if "best_thr" not in meta:
        meta["best_thr"] = 0.5
    return model, meta


# ----------------- Eval ENSEMBLE trên VAL -----------------
@torch.no_grad()
def evaluate_ensemble(
    models: List[nn.Module],
    loader: DataLoader,
    device: torch.device,
    method_names: List[str],
    thr_min: float,
    thr_max: float,
    thr_steps: int,
    val_tta: str = "none",
    val_repeat: int = 1,
    cons_bacc_min: float = 0.90,
    cons_rec_real_min: float = 0.90,
    fake_index: int = 0,
    phase_name: str = "VAL_ENSEMBLE",
):
    """
    Giống hàm evaluate() trong train_spatial.py nhưng:
    - Nhận list models (>=1).
    - Nếu 1 model: behave như cũ.
    - Nếu >=2 models: ensemble bằng cách:
        + p_bin_ensemble  = mean_j softmax(logits_bin_j)
        + p_method_ens    = mean_j softmax(logits_method_j)
        + p_fake_ensemble = p_bin_ensemble[:, fake_index]
    Sau đó quét threshold trên p_fake_ensemble.
    """

    K = thr_steps
    thrs = torch.linspace(thr_min, thr_max, steps=K, device=device)

    correct_fake = torch.zeros(K, dtype=torch.float64, device=device)
    correct_real = torch.zeros(K, dtype=torch.float64, device=device)
    tot_fake = 0
    tot_real = 0

    method_correct = 0
    method_total = 0
    per_m_correct = {m: 0 for m in method_names}
    per_m_total = {m: 0 for m in method_names}

    for r in range(val_repeat):
        for x, yb, ym, ybr, _ in tqdm(loader, desc=f"[{phase_name} r{r+1}]", dynamic_ncols=True):
            x = x.to(device)
            yb = yb.to(device)
            ym = ym.to(device)

            # ---- TTA + ENSEMBLE ----
            # Tính prob_bin (B,2), prob_met (B,M)
            if val_tta == "hflip":
                # Với mỗi model: average (orig, hflip), sau đó average giữa models
                prob_bin_acc = []
                prob_met_acc = []
                x_flip = torch.flip(x, [-1])
                for m in models:
                    lb1, lm1, _, _, _ = m(x)
                    lb2, lm2, _, _, _ = m(x_flip)
                    pb1 = torch.softmax(lb1.float(), dim=1)
                    pb2 = torch.softmax(lb2.float(), dim=1)
                    pm1 = torch.softmax(lm1.float(), dim=1)
                    pm2 = torch.softmax(lm2.float(), dim=1)
                    prob_bin_acc.append((pb1 + pb2) * 0.5)
                    prob_met_acc.append((pm1 + pm2) * 0.5)
                prob_bin = torch.stack(prob_bin_acc, dim=0).mean(dim=0)
                prob_met = torch.stack(prob_met_acc, dim=0).mean(dim=0)

            elif val_tta == "scale":
                scales = [0.9, 1.0, 1.1]
                prob_bin_acc = []
                prob_met_acc = []
                for m in models:
                    pb_scales = []
                    pm_scales = []
                    for s in scales:
                        hs = int(x.shape[2] * s)
                        ws = int(x.shape[3] * s)
                        xs = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False)
                        # đảm bảo về img_size chuẩn
                        # Ở đây assume img_size = loader.dataset.tfm size; F.interpolate sẽ resize lại
                        xs = F.interpolate(xs, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
                        lb_s, lm_s, _, _, _ = m(xs)
                        pb_scales.append(torch.softmax(lb_s.float(), dim=1))
                        pm_scales.append(torch.softmax(lm_s.float(), dim=1))
                    pb = torch.stack(pb_scales, dim=0).mean(dim=0)
                    pm = torch.stack(pm_scales, dim=0).mean(dim=0)
                    prob_bin_acc.append(pb)
                    prob_met_acc.append(pm)
                prob_bin = torch.stack(prob_bin_acc, dim=0).mean(dim=0)
                prob_met = torch.stack(prob_met_acc, dim=0).mean(dim=0)

            else:  # "none"
                prob_bin_acc = []
                prob_met_acc = []
                for m in models:
                    lb, lm, _, _, _ = m(x)
                    prob_bin_acc.append(torch.softmax(lb.float(), dim=1))
                    prob_met_acc.append(torch.softmax(lm.float(), dim=1))
                prob_bin = torch.stack(prob_bin_acc, dim=0).mean(dim=0)
                prob_met = torch.stack(prob_met_acc, dim=0).mean(dim=0)

            # ---- metrics method (fake-only) ----
            mask_fake = (yb == 0) & (ym >= 0)
            if mask_fake.any():
                pred_m = prob_met[mask_fake].argmax(1)
                mt = ym[mask_fake]
                method_correct += int((pred_m == mt).sum().item())
                method_total += int(mask_fake.sum().item())
                for i in range(mt.numel()):
                    t = int(mt[i])
                    p = int(pred_m[i])
                    name = method_names[t] if t < len(method_names) else f"method_{t}"
                    per_m_total[name] += 1
                    if p == t:
                        per_m_correct[name] += 1

            # ---- metrics binary (threshold sweep) ----
            p_fake = prob_bin[:, fake_index]  # yb==0 là fake
            comp = (p_fake.unsqueeze(1) >= thrs.unsqueeze(0))  # [B,K]
            # pred_bin: >=thr → fake(0); <thr → real(1)
            pred_bin = torch.where(comp, torch.zeros_like(yb).unsqueeze(1), torch.ones_like(yb).unsqueeze(1))
            yexp = yb.unsqueeze(1).expand_as(pred_bin)

            is_fake = (yexp == 0)
            is_real = (yexp == 1)
            correct_fake += (pred_bin == 0).logical_and(is_fake).sum(0).to(torch.float64)
            correct_real += (pred_bin == 1).logical_and(is_real).sum(0).to(torch.float64)
            tot_fake += int((yb == 0).sum().item())
            tot_real += int((yb == 1).sum().item())

    # ---- aggregate ----
    rec_fake = (correct_fake / max(1, tot_fake)).cpu()
    rec_real = (correct_real / max(1, tot_real)).cpu()
    bacc = 0.5 * (rec_fake + rec_real)
    acc = (correct_fake + correct_real).cpu() / float(tot_fake + tot_real + 1e-12)

    best_idx = int(torch.argmax(bacc))
    best_thr = float(thrs[best_idx].cpu())
    res = {
        "acc": float(acc[best_idx]),
        "bacc": float(bacc[best_idx]),
        "best_thr": best_thr,
        "rec_fake": float(rec_fake[best_idx]),
        "rec_real": float(rec_real[best_idx]),
        "method_acc_fake": (method_correct / method_total if method_total > 0 else 0.0),
        "per_method_acc": {
            m: (per_m_correct[m] / per_m_total[m] if per_m_total[m] > 0 else None) for m in method_names
        },
    }

    # chọn thêm "conservative threshold" nếu cần
    mask = (bacc >= cons_bacc_min) & (rec_real >= cons_rec_real_min)
    if mask.any():
        idxs = torch.nonzero(mask).squeeze(1)
        best = idxs[torch.argmax(rec_fake[idxs])]
        res.update(
            {
                "cons_thr": float(thrs[best].cpu()),
                "cons_acc": float(acc[best]),
                "cons_bacc": float(bacc[best]),
                "cons_rec_fake": float(rec_fake[best]),
                "cons_rec_real": float(rec_real[best]),
            }
        )

    print(
        f"[KQ {phase_name}] acc={res['acc']:.4f} | bacc={res['bacc']:.4f} | "
        f"rec_fake={res['rec_fake']:.4f} | rec_real={res['rec_real']:.4f} | "
        f"thr*={res['best_thr']:.3f} | m_acc(fake)={res['method_acc_fake']:.4f}"
    )
    if "cons_thr" in res:
        print(
            f"[KQ {phase_name}] CONS thr={res['cons_thr']:.3f} | bacc={res['cons_bacc']:.4f} | "
            f"rec_fake={res['cons_rec_fake']:.4f} | rec_real={res['cons_rec_real']:.4f}"
        )
    return res


# ----------------- MAIN -----------------
def main():
    ap = argparse.ArgumentParser("Sweep ensemble threshold trên VAL (spatial, face-only)")
    ap.add_argument("--data_root", type=str, required=True, help="VD: data/processed_multi")
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        nargs="+",
        help="Một hoặc nhiều detector_best.pt (ensemble khi >=2)",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--img_size", type=int, default=0, help="0 = lấy từ ckpt.meta.img_size (ckpt đầu)")
    ap.add_argument("--model_name", type=str, default="", help="Rỗng = lấy từ ckpt.meta.model_name/backbone_model")
    ap.add_argument("--val_batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)

    # threshold sweep
    ap.add_argument("--thr_min", type=float, default=0.0)
    ap.add_argument("--thr_max", type=float, default=1.0)
    ap.add_argument("--thr_steps", type=int, default=201)

    # eval control
    ap.add_argument("--val_tta", choices=["none", "hflip", "scale"], default="none")
    ap.add_argument("--val_repeat", type=int, default=1)
    ap.add_argument("--cons_bacc_min", type=float, default=0.90)
    ap.add_argument("--cons_rec_real_min", type=float, default=0.90)
    ap.add_argument("--fake_index", type=int, default=0, help="Chỉ số lớp FAKE trong head nhị phân (0 hoặc 1)")

    ap.add_argument("--out_json", type=str, default="", help="Nếu set, lưu kết quả res ra file JSON")
    args = ap.parse_args()

    # device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # dataset VAL
    img_size = args.img_size if args.img_size > 0 else None
    tfm_eval = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)) if img_size is not None else transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    ds_val = MultiBranchDataset(args.data_root, "val", tfm_eval)
    method_names = ds_val.method_names
    print(f"📊 Method union (VAL): {method_names}")
    print(f"🔎 VAL size: {len(ds_val)}")

    dl_val = DataLoader(
        ds_val,
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if args.workers > 0 else None,
        persistent_workers=(args.workers > 0),
    )

    # load models
    ckpt_paths: List[str] = args.ckpt
    models: List[nn.Module] = []
    metas: List[Dict] = []

    img_size_arg = args.img_size if args.img_size > 0 else None
    model_name_arg = args.model_name if args.model_name else None

    for p in ckpt_paths:
        m, meta = load_checkpoint_build_model(
            ckpt_path=p,
            device=device,
            img_size_arg=img_size_arg,
            model_name_arg=model_name_arg,
        )
        models.append(m)
        metas.append(meta)
        print(
            f"[LOAD] {os.path.basename(p)} | model={meta.get('model_name')} | "
            f"img_size={meta.get('img_size')} | best_thr_single={meta.get('best_thr')}"
        )

    print(f"[INFO] Tổng số model: {len(models)}")

    # chạy sweep ensemble
    res = evaluate_ensemble(
        models=models,
        loader=dl_val,
        device=device,
        method_names=method_names,
        thr_min=args.thr_min,
        thr_max=args.thr_max,
        thr_steps=args.thr_steps,
        val_tta=args.val_tta,
        val_repeat=args.val_repeat,
        cons_bacc_min=args.cons_bacc_min,
        cons_rec_real_min=args.cons_rec_real_min,
        fake_index=args.fake_index,
        phase_name="VAL_ENSEMBLE",
    )

    # in kết quả ngắn gọn
    print("\n========== ENSEMBLE THRESHOLD RESULT ==========")
    print(f"- num_models       : {len(models)}")
    print(f"- best_thr         : {res['best_thr']:.6f}")
    print(f"- acc              : {res['acc']:.4f}")
    print(f"- bacc             : {res['bacc']:.4f}")
    print(f"- rec_fake         : {res['rec_fake']:.4f}")
    print(f"- rec_real         : {res['rec_real']:.4f}")
    if "cons_thr" in res:
        print(f"- cons_thr (safe)  : {res['cons_thr']:.6f}")
        print(f"- cons_bacc        : {res['cons_bacc']:.4f}")
        print(f"- cons_rec_fake    : {res['cons_rec_fake']:.4f}")
        print(f"- cons_rec_real    : {res['cons_rec_real']:.4f}")
    print("===============================================")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Kết quả đã được lưu vào {args.out_json}")


if __name__ == "__main__":
    main()
