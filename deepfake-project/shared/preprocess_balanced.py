import argparse, os, json, time, random, shutil
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

FACE_METHODS = {
    "Deepfakes","Face2Face","FaceShifter","FaceSwap","NeuralTextures","Transformer"
}
FULL_METHODS = {"Diffusion_match","Diffusion_mismatch"}
DEFAULT_METHODS = list(FACE_METHODS | FULL_METHODS)
DEFAULT_REAL_DIR = "Original"

# ---------------- utils ----------------
def list_videos(d: Path):
    if not d.exists(): return []
    return sorted([p for p in d.rglob("*") if p.suffix.lower() in VIDEO_EXTS])

def probe_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return 0,0.0,0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    dur = (n/fps) if (fps>0 and n>0) else 0.0
    cap.release()
    return n,fps,dur

def ensure_empty_dir(d: Path):
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True,exist_ok=True)

def sec_to_hms(s: float):
    m,s = divmod(int(s),60)
    h,m = divmod(m,60)
    return f"{h:d}:{m:02d}:{s:02d}"

def split_train_val_test(videos, seed:int):
    rng=random.Random(seed)
    vids=videos.copy(); rng.shuffle(vids)
    n=len(vids)
    n_train=int(round(n*0.7))
    n_val=int(round(n*0.2))
    train=vids[:n_train]
    val=vids[n_train:n_train+n_val]
    test=vids[n_train+n_val:]
    return {"train":train,"val":val,"test":test}

# ---------------- segment planning ----------------
def max_nonoverlap_segments(dur, seg_len, margin):
    if dur<=0 or seg_len<=0: return 0
    return int(np.floor((dur+margin)/(seg_len+margin)))

def choose_segment_starts(dur,k,seg_len,strategy,margin,rng):
    if dur<=0 or seg_len<=0 or k<=0: return []
    k=min(k,max_nonoverlap_segments(dur,seg_len,margin))
    if k<=0: return []
    max_start=max(0.0,dur-seg_len)
    if k==1:
        return [min(max_start,max(0.0,(dur-seg_len)/2.0))]
    if strategy=="uniform":
        spacing=(dur-seg_len)/(k-1)
        jitter=min(margin/2.0,spacing/4.0)
        starts=[]
        for i in range(k):
            s=i*spacing
            if jitter>0: s=max(0.0,min(max_start,s+rng.uniform(-jitter,jitter)))
            starts.append(float(s))
        return sorted(starts)
    # random
    starts=[];attempts=0
    while len(starts)<k and attempts<2000:
        s=rng.uniform(0.0,max_start)
        if all(abs(s-t)>=(seg_len+margin) for t in starts):
            starts.append(s)
        attempts+=1
    if len(starts)<k:
        return choose_segment_starts(dur,k,seg_len,"uniform",margin,rng)
    return sorted(starts)

# ---------------- extraction helpers ----------------
def crop_face(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml").detectMultiScale(gray,1.3,5)
    if len(faces)==0: return frame
    x,y,w,h=max(faces,key=lambda f:f[2]*f[3])
    pad=int(0.2*max(w,h))
    x1=max(0,x-pad);y1=max(0,y-pad)
    x2=min(frame.shape[1],x+w+pad)
    y2=min(frame.shape[0],y+h+pad)
    return frame[y1:y2,x1:x2]

def resize_frame(frame,size):
    return cv2.resize(frame,(size,size),interpolation=cv2.INTER_AREA)

def write_frame(out_dir,idx,frame):
    out_dir.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out_dir/f"{idx:06d}.jpg"),frame)

def extract_segments(path,out_dir,fps,starts,seg_len,crop_mode,img_size):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): return 0
    total=0
    f_per_seg=max(1,int(round(seg_len*max(fps,1e-6))))
    for s in starts:
        for i in range(f_per_seg):
            t=s+i/max(fps,1e-6)
            cap.set(cv2.CAP_PROP_POS_MSEC,t*1000.0)
            ok,frame=cap.read()
            if not ok: break
            if crop_mode=="face": frame=crop_face(frame)
            frame=resize_frame(frame,img_size)
            write_frame(out_dir,total,frame)
            total+=1
    cap.release()
    return total

# ---------------- main ----------------
def main():
    ap=argparse.ArgumentParser("Balanced preprocessor with auto frame balancing")
    ap.add_argument("--data_root",required=True)
    ap.add_argument("--out_root",required=True)
    ap.add_argument("--fps",type=float,default=25.0)
    ap.add_argument("--seg_len",type=float,default=0.6)
    ap.add_argument("--margin",type=float,default=0.1)
    ap.add_argument("--strategy",choices=["uniform","random"],default="uniform")
    ap.add_argument("--img_size",type=int,default=512)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    rng=random.Random(args.seed)
    data_root=Path(args.data_root); out_root=Path(args.out_root)
    ensure_empty_dir(out_root)

    classes=list(DEFAULT_METHODS)+[DEFAULT_REAL_DIR]
    vids_by_class={c:list_videos(data_root/c) for c in classes}

    print("\n[Scan]")
    for c,v in vids_by_class.items(): print(f"  {c:<20}: {len(v)} videos")

    # split each class 70/20/10 (non overlap)
    splits={sp:{} for sp in ["train","val","test"]}
    for c,vids in vids_by_class.items():
        split=split_train_val_test(vids,args.seed)
        for sp in ["train","val","test"]: splits[sp][c]=split[sp]

    # target setup
    frames_per_video_face=75
    frames_per_video_diff=75
    target_train,total_val,total_test=52500,15000,7500

    totals={sp:{} for sp in ["train","val","test"]}
    start=time.time()

    for sp in ["train","val","test"]:
        for c,vids in splits[sp].items():
            base=out_root/sp
            split_dir=base/("fake" if c!=DEFAULT_REAL_DIR else "real")/c
            crop_mode="face" if c in FACE_METHODS else "full"
            total_frames=0
            pbar=tqdm(vids,desc=f"{sp}:{c}",unit="vid")
            for vid in pbar:
                _,_,dur=probe_video(vid)
                if c in FACE_METHODS or c in FULL_METHODS:
                    segs=frames_per_video_face//int(args.seg_len*args.fps) if c in FACE_METHODS else frames_per_video_diff//int(args.seg_len*args.fps)
                else:
                    segs=2
                k=min(segs,max_nonoverlap_segments(dur,args.seg_len,args.margin))
                if c==DEFAULT_REAL_DIR and rng.random()<0.23:  # ~23% real vids get +1 segment
                    k=min(k+1,3)
                starts=choose_segment_starts(dur,k,args.seg_len,args.strategy,args.margin,rng)
                count=extract_segments(vid,split_dir/vid.stem,args.fps,starts,args.seg_len,crop_mode,args.img_size)
                total_frames+=count
                pbar.set_postfix(frames=total_frames)
            totals[sp][c]=total_frames

    print("\n[Done]")
    summary={"args":vars(args),"totals":totals,"time_sec":time.time()-start}
    for sp in ["train","val","test"]:
        s=sum(totals[sp].values())
        print(f"{sp}: {s} frames")
        for c,n in totals[sp].items():
            print(f"  - {c:<20}: {n}")
    (out_root/"build_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"\nTotal time: {sec_to_hms(summary['time_sec'])}")

if __name__=="__main__":
    main()
