import argparse
import os
import sys

# --- Silence Win WS reset noise on Windows (harmless ConnectionResetError) ---
if sys.platform.startswith("win"):
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost
        def _safe_call_connection_lost(self, exc):
            try:
                return _orig_call_connection_lost(self, exc)
            except ConnectionResetError:
                # Swallow noisy "WinError 10054" when browser closes websocket
                return
        _ProactorBasePipeTransport._call_connection_lost = _safe_call_connection_lost
    except Exception:
        pass
# ---------------------------------------------------------------------------

from components.model import load_multiple_detectors, discover_checkpoints
from components.ui import create_ui

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="", help="Danh sách checkpoint, phân tách bằng dấu phẩy. Nếu bỏ trống sẽ tự dò.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--thr", type=float, default=0.5)           # fallback nếu meta thiếu
    parser.add_argument("--method_gate", type=float, default=0.55)
    parser.add_argument("--per_method_thr", type=str, default="")
    parser.add_argument("--enable_filters", action="store_true", default=True)
    args = vars(parser.parse_args())

    # 1) lấy danh sách ckpt
    if args["models"]:
        ckpt_paths = [os.path.normpath(p.strip()) for p in args["models"].split(",") if p.strip()]
    else:
        ckpt_paths = discover_checkpoints()
        if not ckpt_paths:
            raise SystemExit("Không tìm thấy checkpoint nào. Hãy đặt file .pt trong 'deepfake_detector/models/**' hoặc 'models/**' (ưu tiên 'detector_best.pt').")

    # 2) nạp
    detectors_info = load_multiple_detectors(ckpt_paths)

    # 3) label hiển thị: lấy folder cha của 'checkpoints'
    model_labels = []
    for p in ckpt_paths:
        parent = os.path.basename(os.path.dirname(p))           # thường là 'checkpoints'
        grand = os.path.basename(os.path.dirname(os.path.dirname(p)))  # vitb384_512
        label = grand if parent.lower() == "checkpoints" else parent
        model_labels.append(label or os.path.basename(p))

    classes   = detectors_info[0][3]
    method_ns = detectors_info[0][4]
    img_size  = detectors_info[0][5]
    det_thr   = detectors_info[0][6] if detectors_info[0][6] is not None else args["thr"]

    cli_args = {
        "thr": args["thr"],
        "method_gate": args["method_gate"],
        "per_method_thr": args["per_method_thr"],
        "enable_filters": args["enable_filters"],
    }

    demo = create_ui(detectors_info, classes, method_ns, img_size, det_thr, cli_args, model_labels)
    demo.queue().launch(server_name=args["host"], server_port=args["port"], share=False)

if __name__ == "__main__":
    main()
