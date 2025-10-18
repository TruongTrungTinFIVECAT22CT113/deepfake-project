import argparse

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Deepfake Detect CLI")
    parser.add_argument('--ckpt_paths', type=str, nargs='+', default=["deepfake-project/deepfake_detector/models/vitb384_512_calibrated.pt"],
                        help="Paths to model checkpoints (multiple for ensemble)")
    parser.add_argument('--thr', type=float, default=None,
                        help="Override global threshold (default: từ checkpoint)")
    parser.add_argument('--per_method_thr', type=str, default="",
                        help="Per-method thresholds (format: Deepfakes=0.72,Face2Face=0.78,...)")
    parser.add_argument('--method_gate', type=float, default=0.55,
                        help="Method confidence gate (>= gate thì dùng per-method threshold & filter)")
    parser.add_argument('--enable_filters', type=int, default=1, choices=[0, 1],
                        help="Bật/tắt filter theo method (1=On, 0=Off)")
    args, _ = parser.parse_known_args()
    return {
        'ckpt_paths': args.ckpt_paths,
        'thr': args.thr,
        'per_method_thr': args.per_method_thr,
        'method_gate': float(args.method_gate),
        'enable_filters': bool(args.enable_filters)
    }