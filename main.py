import math, io, base64
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from ultralytics import YOLO as _YOLO
    YOLO_OK = True
except:
    YOLO_OK = False

try:
    from pythermalcomfort.utilities import v_relative as _vr
    from pythermalcomfort.models import pmv_ppd_iso as _pmv
    THERMAL_OK = True
except:
    THERMAL_OK = False

# ── 클래스 설정 ──
CLO_MAP   = {0:0.30, 1:0.57, 2:0.74, 3:0.96}
CLS_LABEL = {0:"반팔", 1:"얇은 긴팔", 2:"니트/맨투맨", 3:"외투"}
CLS_COLOR = {0:(239,68,68), 1:(245,158,11), 2:(16,185,129), 3:(59,130,246)}

MET=1.1; RH=50; V_AIR=0.1
PMV_HIGH=0.5; PMV_LOW=-0.5
AC_MIN=22.0; AC_MAX=28.0; BASE_TEMP=24.0
WEIGHT_PATH = "best.pt"

# ── 서버 상태 ──
state = {
    "ac_temp":BASE_TEMP, "base_energy":0.0, "smart_energy":0.0,
    "frame":0, "logs":[], "last_pmv":0.0, "last_avg_clo":0.55,
    "last_counts":{0:0,1:0,2:0,3:0}
}
yolo_model = None

def get_model():
    global yolo_model
    if yolo_model is None and YOLO_OK and Path(WEIGHT_PATH).exists():
        yolo_model = _YOLO(WEIGHT_PATH)
        add_log(f"✅ YOLOv8 로드 완료: {WEIGHT_PATH}")
    return yolo_model

def add_log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    state["logs"].insert(0, f"[{ts}] {msg}")
    if len(state["logs"]) > 20: state["logs"].pop()

def calc_pmv(tdb, clo):
    if THERMAL_OK:
        try:
            vr = _vr(v=V_AIR, met=MET)
            return float(_pmv(tdb=tdb, tr=tdb, vr=vr, rh=RH, met=MET, clo=clo).pmv)
        except: pass
    pa=RH*10*math.exp(16.6536-4030.183/(tdb+235))
    icl=0.155*clo; m=MET*58.15
    fcl=1.0+1.29*icl if icl<=0.078 else 1.05+0.645*icl
    vr=max(0.05,V_AIR+0.3*(MET-1.0)); hcf=12.1*math.sqrt(vr)
    tcl=max(tdb,min(40.0,35.7-0.028*m-icl*3.5*tdb))
    for _ in range(300):
        hca=2.38*abs(100*tcl-100*tdb)**0.25; hc=max(hcf,hca)
        tclN=35.7-0.028*m-icl*(3.96e-8*fcl*((tcl+273)**4-(tdb+273)**4)+fcl*hc*(tcl-tdb))
        if abs(tclN-tcl)<1e-5: tcl=tclN; break
        tcl=tcl*0.6+tclN*0.4
    hca=2.38*abs(100*tcl-100*tdb)**0.25; hc=max(hcf,hca)
    L=(m-3.05e-3*(5733-6.99*m-pa)-(0.42*(m-58.15) if m>58.15 else 0)
       -1.7e-5*m*(5867-pa)-0.0014*m*(34-tdb)
       -3.96e-8*fcl*((tcl+273)**4-(tdb+273)**4)-fcl*hc*(tcl-tdb))
    return max(-3.0,min(3.0,(0.303*math.exp(-0.036*m)+0.028)*L))

def cooling_load(t_out, t_set):
    return max(0.5,1.8+(t_out-28)*0.22)*(0.91**(t_set-BASE_TEMP))

def draw_boxes(img, boxes):
    draw=ImageDraw.Draw(img)
    try: font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",16)
    except: font=ImageFont.load_default()
    for (x1,y1,x2,y2,cls_id,conf) in boxes:
        c=CLS_COLOR.get(cls_id,(150,150,150))
        draw.rectangle([x1,y1,x2,y2],outline=c,width=3)
        label=f"cls{cls_id} {CLS_LABEL[cls_id]} clo={CLO_MAP[cls_id]:.2f} {conf:.2f}"
        try:
            bb=draw.textbbox((x1,y1-22),label,font=font)
            draw.rectangle([bb[0]-3,bb[1]-2,bb[2]+3,bb[3]+2],fill=c)
            draw.text((x1,y1-22),label,fill=(255,255,255),font=font)
        except: draw.text((x1,y1-22),label,fill=c)
    return img

def img_to_b64(img):
    buf=io.BytesIO(); img.save(buf,format="JPEG",quality=90)
    return base64.b64encode(buf.getvalue()).decode()

app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def root():
    p=Path("index.html")
    return HTMLResponse(p.read_text(encoding="utf-8") if p.exists()
                        else "<h2>index.html을 smartroom 폴더에 넣어주세요</h2>")

@app.post("/analyze")
async def analyze(file:UploadFile=File(...),
                  indoor_temp:float=Form(24.0),
                  outdoor_temp:float=Form(32.0)):
    state["frame"]+=1
    raw=await file.read()
    img=Image.open(io.BytesIO(raw)).convert("RGB")
    boxes=[]; m=get_model()
    if m:
        res=m(np.array(img),verbose=False,conf=0.25)[0]
        for box in res.boxes:
            c=int(box.cls[0].item())
            if c in CLO_MAP:
                cf=float(box.conf[0].item())
                x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                boxes.append((x1,y1,x2,y2,c,cf))
        add_log(f"YOLO 감지: {len(boxes)}명")
    else:
        W,H=img.size
        boxes=[(int(W*.05),int(H*.1),int(W*.45),int(H*.9),1,0.91),
               (int(W*.55),int(H*.1),int(W*.95),int(H*.9),2,0.87)]
        add_log("⚠️ 더미 감지 (모델 없음)")
    counts={0:0,1:0,2:0,3:0}
    for(*_,c,_cf) in boxes:
        if c in counts: counts[c]+=1
    state["last_counts"]=counts
    total=sum(counts.values())
    avg_clo=(sum(CLO_MAP[c]*counts[c] for c in counts)/total
             if total>0 else state["last_avg_clo"])
    state["last_avg_clo"]=avg_clo
    tdb=state["ac_temp"] if state["frame"]>1 else indoor_temp
    pmv=calc_pmv(tdb,avg_clo); state["last_pmv"]=pmv
    cur=state["ac_temp"]
    if pmv>PMV_HIGH:
        new=max(AC_MIN,cur-1.0); add_log(f"PMV {pmv:+.2f} > +0.5 → {new:.1f}°C 하향")
    elif pmv<PMV_LOW:
        new=min(AC_MAX,cur+1.0); add_log(f"PMV {pmv:+.2f} < -0.5 → {new:.1f}°C 상향")
    else:
        new=cur; add_log(f"PMV {pmv:+.2f} 쾌적 → {new:.1f}°C 유지")
    state["ac_temp"]=new
    state["base_energy"]+=cooling_load(outdoor_temp,BASE_TEMP)
    state["smart_energy"]+=cooling_load(outdoor_temp,new)
    saving=((state["base_energy"]-state["smart_energy"])/state["base_energy"]*100
            if state["base_energy"]>0 else 0.0)
    img_b64=img_to_b64(draw_boxes(img.copy(),boxes))
    return JSONResponse({"ok":True,"frame":state["frame"],"annotated_b64":img_b64,
        "cls_counts":counts,"total_persons":total,"avg_clo":round(avg_clo,3),
        "pmv":round(pmv,3),"ac_temp":round(new,1),"ac_baseline":BASE_TEMP,
        "base_energy":round(state["base_energy"],2),"smart_energy":round(state["smart_energy"],2),
        "saving_pct":round(saving,1),"saving_kwh":round(state["base_energy"]-state["smart_energy"],2),
        "outdoor_temp":outdoor_temp,"indoor_temp":indoor_temp,"logs":state["logs"][:8]})

@app.get("/state")
async def get_state():
    saving=((state["base_energy"]-state["smart_energy"])/state["base_energy"]*100
            if state["base_energy"]>0 else 0.0)
    return {**state,"saving_pct":round(saving,1)}

@app.post("/reset")
async def reset():
    state.update({"ac_temp":BASE_TEMP,"base_energy":0.0,"smart_energy":0.0,
                  "frame":0,"logs":[],"last_pmv":0.0,"last_avg_clo":0.55,
                  "last_counts":{0:0,1:0,2:0,3:0}})
    return {"ok":True}