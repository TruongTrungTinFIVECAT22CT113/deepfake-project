import argparse
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def build_cmd(src: Path, dst: Path, fps: int, gop: int, crf: int, preset: str,
              profile: str, level: str, audio_bitrate: str, audio_rate: int, keep_audio: bool,
              extra_x264: str, overwrite: bool):
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-hide_banner", "-loglevel", "error",
        "-i", str(src),

        # --- Chuẩn hoá thời gian ---
        "-vsync", "cfr",
        "-r", str(fps),

        # --- Video H.264 + GOP cố định ---
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", profile,
        "-level", level,
        "-preset", preset,
        "-crf", str(crf),
        "-g", str(gop),
        "-x264-params", f"keyint={gop}:min-keyint={gop}:scenecut=0" + (f":{extra_x264}" if extra_x264 else ""),
    ]

    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-ar", str(audio_rate)]
    else:
        cmd += ["-an"]

    cmd += [str(dst)]
    return cmd

def convert_one(src: Path, dst: Path, **kwargs):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(src, dst, **kwargs)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[WARN] Failed: {src}")

def main():
    ap = argparse.ArgumentParser("Convert & normalize videos (CFR/GOP) to H.264")
    ap.add_argument("--in_root",  required=True, help="Input folder (source videos)")
    ap.add_argument("--out_root", required=True, help="Output folder (normalized)")
    ap.add_argument("--fps", type=int, default=25, help="Target FPS (CFR)")
    ap.add_argument("--gop", type=int, default=250, help="GOP size (keyint/min-keyint)")
    ap.add_argument("--crf", type=int, default=20, help="x264 CRF (lower=better quality)")
    ap.add_argument("--preset", default="medium", choices=["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"])
    ap.add_argument("--profile", default="high")
    ap.add_argument("--level",   default="4.1")
    ap.add_argument("--audio_bitrate", default="128k")
    ap.add_argument("--audio_rate", type=int, default=48000)
    ap.add_argument("--no-audio", action="store_true", help="Strip audio")
    ap.add_argument("--ext", default=".mp4", help="Output extension (.mp4 recommended)")
    ap.add_argument("--extra_x264", default="", help="Extra x264 params (e.g. 'aq-mode=3')")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    if not has_ffmpeg():
        raise SystemExit("ffmpeg not found in PATH. Please install ffmpeg and try again.")

    in_root  = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    keep_audio = not args.no_audio

    videos = [p for p in in_root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        print(f"No videos found under: {in_root}")
        return

    print(f"Found {len(videos)} videos")
    for src in videos:
        rel = src.relative_to(in_root)
        dst = (out_root / rel).with_suffix(args.ext)
        convert_one(
            src, dst,
            fps=args.fps, gop=args.gop, crf=args.crf, preset=args.preset,
            profile=args.profile, level=args.level,
            audio_bitrate=args.audio_bitrate, audio_rate=args.audio_rate,
            keep_audio=keep_audio, extra_x264=args.extra_x264, overwrite=args.overwrite,
        )

if __name__ == "__main__":
    main()
