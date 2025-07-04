#!/usr/bin/env python
# journey_events_mser.py  –  CPU-only diary generator
# ---------------------------------------------------
# 1. ffmpeg → PNG frames @ 2 fps
# 2. YOLO-v8s  → objects (cars, lights, stop signs)
# 3. Farneback  → left / right / stopped
# 4. HSV        → red / green light
# 5. MSER + EasyOCR → shop / station names (no EAST .pb needed)
# 6. One sentence every 30 frames – written to journeys/

from __future__ import annotations
import argparse, subprocess as sp
from collections import Counter, deque
from pathlib import Path
from typing import List, Tuple
import cv2, numpy as np, easyocr
from tqdm import tqdm
from ultralytics import YOLO

# ───────────────────────── user defaults ─────────────────────────
VIDEO = Path(r"C:/Users/hanis/OneDrive/Desktop/Cranfield/IRP/Dataset_Videos/2min_clips/Video-Sample/clip_4.mp4")
FPS    = 2.0     # frames-per-second to extract
CHUNK  = 30      # = one diary sentence
INTRO  = deque(["After a while","Soon","Shortly","Before long","Moments later"])

# optical-flow thresholds
DIR_THR, STOP_THR = 1.5, 0.6    # px median, mag mean

# ───────────────────────── CLI ─────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--video",  type=Path,  default=VIDEO)
ap.add_argument("--fps",    type=float, default=FPS)
ap.add_argument("--chunk",  type=int,   default=CHUNK)
ap.add_argument("--work",   type=Path,  default=Path("dataset"))
ap.add_argument("--ollama")                     # optional local LLM
args = ap.parse_args()

FRAMES = args.work / "frames"; FRAMES.mkdir(parents=True, exist_ok=True)
JRN    = Path("journeys");       JRN.mkdir(exist_ok=True)

# ───────────────────────── step-0 extract png ─────────
print("↳ extracting frames …")
sp.run(["ffmpeg","-loglevel","error","-y","-i",str(args.video),
        "-vf",f"fps={args.fps}", str(FRAMES/"%06d.png")])
frames = sorted(FRAMES.glob("*.png"))
print(f"   {len(frames)} frames → {FRAMES}")

# ───────────────────────── models ────────────────────
print("↳ loading YOLO-v8s …")
yolo = YOLO("yolov8s.pt")
ocr  = easyocr.Reader(["en"], gpu=False, verbose=False)

# ─────────── helpers – flow / HSV / MSER / OCR ───────
_prev = None; _turn_buf,_stop_buf = deque(maxlen=15),deque(maxlen=15)

def flow_cues(img: np.ndarray)->Tuple[str|None,bool]:
    global _prev
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); dir_,stop=None,False
    if _prev is not None:
        f=cv2.calcOpticalFlowFarneback(_prev,g,None,.5,3,15,3,5,1.2,0)
        mag,_=cv2.cartToPolar(f[...,0],f[...,1])
        dx=np.median(f[int(g.shape[0]*0.6):,:,0])
        if dx> DIR_THR:  dir_="left"
        elif dx<-DIR_THR: dir_="right"
        stop = mag[int(g.shape[0]*0.6):].mean() < STOP_THR
    _prev=g; _turn_buf.append(dir_ or "straight"); _stop_buf.append(stop)
    maj = Counter(_turn_buf).most_common(1)[0][0]
    return (None if maj=="straight" else maj), any(_stop_buf)

# HSV masks
HSV_RED1,HSV_RED2 = ((0,70,50),(10,255,255)),((170,70,50),(180,255,255))
HSV_GREEN         = ((40,50,50),(90,255,255))
def light_colour(crop):
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    r=cv2.inRange(hsv,*HSV_RED1)|cv2.inRange(hsv,*HSV_RED2)
    g=cv2.inRange(hsv,*HSV_GREEN)
    if r.mean()>40 and r.mean()>g.mean():   return "red"
    if g.mean()>40 and g.mean()>r.mean():   return "green"
    return None

def mser_boxes(img)->List[Tuple[int,int,int,int]]:
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    try: mser=cv2.MSER_create(min_area=800,max_area=50000,delta=5)
    except TypeError: mser=cv2.MSER_create(5,8,200,0.7,0.05,800,50000,1.01,0.003,5)
    regions,_=mser.detectRegions(gray)
    boxes=[]
    for p in regions:
        x,y,w,h=cv2.boundingRect(p)
        if 4000<w*h<60000 and 1<=w/h<=8: boxes.append((x,y,x+w,y+h))
    return boxes

def ocr_words(crop):
    return [t.strip() for _,t,conf in ocr.readtext(crop) if conf>0.45
            and t.isalpha() and len(t)>3]

# ─────────── main loop ───────────────────────────────
chunks=[frames[i:i+args.chunk] for i in range(0,len(frames),args.chunk)]
GLOBAL_POI=set()

for idx,chunk in enumerate(chunks):
    dir_,stopped,lights=None,False,set()
    counter=Counter()

    for fp in tqdm(chunk,leave=False):
        img=cv2.imread(str(fp))

        # YOLO
        det=yolo(fp,device="cpu",verbose=False)[0]
        for cls,bb in zip(det.boxes.cls.cpu(),det.boxes.xyxy.cpu()):
            cls=int(cls)
            if cls==9:          # traffic light
                x1,y1,x2,y2=map(int,bb)
                col=light_colour(img[y1:y2,x1:x2]);  lights.add(col or "")
            if cls==11: lights.add("stop sign")

        d,s=flow_cues(img); dir_=dir_ or d; stopped=stopped or s

        for x1,y1,x2,y2 in mser_boxes(img):
            counter.update(ocr_words(img[y1:y2,x1:x2]))

    # filter OCR noise
    pois={p for p,c in counter.items() if c>=2}
    fresh=[p for p in pois if p not in GLOBAL_POI]
    GLOBAL_POI.update(pois)

    # compose sentence
    parts=[]
    if dir_: parts.append(f"took a {dir_} turn")
    if stopped and ("red" in lights or "stop sign" in lights):
        parts.append("stopped at the red light")
    elif "red" in lights and "green" in lights:
        parts.append("waited at the signal until it turned green")
    if fresh: parts.append("passed "+", ".join(sorted(fresh)))

    if not parts: parts=["drove straight on"]
    intro=INTRO[0]; INTRO.rotate(-1)
    paragraph=f"{intro} the car "+", then ".join(parts)+"."

    # optional Ollama re-wording
    if args.ollama:
        try:
            res=sp.run(["ollama","run",args.ollama],input=paragraph.encode(),
                       capture_output=True,timeout=25)
            if res.returncode==0: paragraph=res.stdout.decode().strip()
        except Exception: pass

    out=JRN/f"journey_{idx:03d}.txt"
    out.write_text(paragraph,"utf-8")
    print(out.name,"→",paragraph)

print("✓ finished – journey paragraphs in",JRN)
