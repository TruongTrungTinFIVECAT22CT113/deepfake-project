# -*- coding: utf-8 -*-
import argparse
import importlib
import sys

def main():
    parser = argparse.ArgumentParser("backend CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Batch eval (mirror backend_eval.py)
    p_eval = sub.add_parser("eval-batch", help="Batch evaluate dataset (mirror backend_eval.py)")
    p_eval.add_argument("--root", type=str, required=True)
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--device", type=str, default="cuda")
    p_eval.add_argument("--img_size", type=int, default=0)
    p_eval.add_argument("--model_name", type=str, default="")
    p_eval.add_argument("--batch", type=int, default=64)
    p_eval.add_argument("--threshold", type=float, default=-1.0)
    p_eval.add_argument("--bbox_scale", type=float, default=1.10)
    p_eval.add_argument("--det_thr", type=float, default=0.5)
    p_eval.add_argument("--fake_index", type=int, default=0)
    p_eval.add_argument("--detector_backend", type=str, default="retinaface", choices=["retinaface","mediapipe"])
    p_eval.add_argument("--out_real", type=str, default="results_real.csv")
    p_eval.add_argument("--out_fake", type=str, default="results_fake.csv")
    p_eval.add_argument("--skip_real", action="store_true")
    p_eval.add_argument("--skip_fake", action="store_true")

    args = parser.parse_args()

    if args.cmd == "eval-batch":
        be = importlib.import_module("backend_eval")
        argv = ["--root", args.root, "--ckpt", args.ckpt,
                "--device", args.device,
                "--batch", str(args.batch),
                "--bbox_scale", str(args.bbox_scale),
                "--det_thr", str(args.det_thr),
                "--fake_index", str(args.fake_index),
                "--detector_backend", args.detector_backend,
                "--out_real", args.out_real,
                "--out_fake", args.out_fake]
        if args.img_size > 0:
            argv += ["--img_size", str(args.img_size)]
        if args.model_name:
            argv += ["--model_name", args.model_name]
        if args.threshold >= 0:
            argv += ["--threshold", str(args.threshold)]
        if args.skip_real:
            argv += ["--skip_real"]
        if args.skip_fake:
            argv += ["--skip_fake"]

        bak = sys.argv[:]
        try:
            sys.argv = ["backend_eval.py"] + argv
            be.main()
        finally:
            sys.argv = bak

if __name__ == "__main__":
    main()