from __future__ import annotations
import re, cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR
from roi import unit_rois
FIELDS=["atk","hp","def","spd","cri","crid","efh","efr"]
class PanelOCR:
    def __init__(self): self.ocr=RapidOCR()
    def _run(self,img):
        result,_=self.ocr(img)
        if not result:return "",0.0
        parts=[]; conf=[]
        for item in result:
            if len(item)>=3: parts.append(str(item[1])); conf.append(float(item[2]))
        return "".join(parts).strip(),sum(conf)/len(conf) if conf else 0.0
    def name(self,img):
        ims=[]
        up=cv2.resize(img,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC); ims.append(up)
        g=cv2.cvtColor(up,cv2.COLOR_BGR2GRAY); ims.extend([g,cv2.convertScaleAbs(g,alpha=1.5,beta=10)])
        res=[(*self._run(x),) for x in ims]; return max(res,key=lambda x:x[1])
    def number(self,img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); gray=cv2.resize(gray,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
        clahe=cv2.createCLAHE(2.0,(8,8)).apply(gray); _,bw=cv2.threshold(clahe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        candidates=[]
        for im in (gray,clahe,bw,255-bw):
            txt,conf=self._run(im); nums=re.findall(r"\d+(?:\.\d+)?",txt.replace(",",""))
            for n in nums:
                try:candidates.append((float(n),conf,txt))
                except:pass
        if not candidates:return None,0.0,""
        value,conf,txt=max(candidates,key=lambda x:x[1]); return int(value) if value.is_integer() else value,conf,txt
    def parse_side(self,img,side,panel_db):
        rows=[]
        for slot in range(1,6):
            rois=unit_rois(img,slot); raw_name,nconf=self.name(rois["name"]); name,nmatch,ncands=panel_db.correct_name(raw_name)
            r={"side":side,"slot":slot,"name_detected":name,"name":name,"raw_name":raw_name,"name_ocr_conf":round(nconf,3),"name_match_conf":round(nmatch,3),"name_candidates":ncands}
            confs=[]
            for f in FIELDS:
                v,c,raw=self.number(rois[f]); r[f]=v; r[f"raw_{f}"]=raw; confs.append(c)
            r["panel_ocr_conf"]=round(sum(confs)/len(confs),3)
            ev=panel_db.evidence(name,r); r.update(soul_inferred=ev["soul"],soul=ev["soul"],soul_match_score=ev["score"],soul_margin=ev["margin"],soul_level=ev["level"],reference_record_id=ev["reference_record_id"],suggestions=ev["suggestions"],soul_candidates=" | ".join(f'{x["soul"]}:{x["score"]:.3f}' for x in ev["candidates"]),confirmed=False)
            rows.append(r)
        return rows
