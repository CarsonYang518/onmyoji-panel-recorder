from __future__ import annotations
import re, cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR
from roi import BASE_W, BASE_H, COLS, NAME_Y, ROWS, normalize_panel

FIELDS=["atk","hp","def","spd","cri","crid","efh","efr"]

class PanelOCR:
    """Exactly one RapidOCR inference per side in the normal path."""
    def __init__(self): self.ocr=RapidOCR()

    @staticmethod
    def _center(box):
        a=np.asarray(box,dtype=float); return float(a[:,0].mean()),float(a[:,1].mean())

    @staticmethod
    def _prepare_full(img, loc, scale=1.75):
        img,_=normalize_panel(img,loc)
        lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        l=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(l)
        out=cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR)
        if scale != 1.0: out=cv2.resize(out,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        return out,scale

    def recognize_full(self,img,loc):
        prepared,scale=self._prepare_full(img,loc); result,_=self.ocr(prepared); items=[]
        for item in result or []:
            if len(item)<3: continue
            try:
                x,y=self._center(item[0]); items.append({"x":x/scale,"y":y/scale,"text":str(item[1]).strip(),"conf":float(item[2])})
            except Exception: continue
        return items

    @staticmethod
    def _in_cell(item,x1,x2,y1,y2): return x1 <= item["x"] <= x2 and y1 <= item["y"] <= y2

    @staticmethod
    def _best_number(items):
        candidates=[]
        for it in items:
            txt=it["text"].replace(",","").replace("％","%").strip()
            for n in re.findall(r"\d+(?:\.\d+)?",txt):
                try: candidates.append((float(n),it["conf"],txt))
                except Exception: pass
        if not candidates:return None,0.0,""
        value,conf,txt=max(candidates,key=lambda z:z[1])
        return (int(value) if value.is_integer() else value),conf,txt

    def parse_items(self,items,side,panel_db):
        rows=[]
        for slot,(x1,x2) in enumerate(COLS,1):
            nh=[it for it in items if self._in_cell(it,x1+4,x2-4,NAME_Y[0]-5,NAME_Y[1]+5)]
            nh=sorted(nh,key=lambda z:z["x"]); raw_name="".join(it["text"] for it in nh).replace(" ","")
            nconf=max((it["conf"] for it in nh),default=0.0)
            # Parse stats BEFORE final name resolution so stats can disambiguate partial OCR.
            r={"side":side,"slot":slot,"raw_name":raw_name,"name_ocr_conf":round(nconf,3)}; confs=[]
            for f,(y1,y2) in ROWS.items():
                hits=[it for it in items if self._in_cell(it,x1+10,x2-10,y1-3,y2+3)]
                v,c,raw=self._best_number(hits); r[f]=v; r[f"raw_{f}"]=raw; confs.append(c)
            resolved=panel_db.resolve_name(raw_name,r)
            r.update(name_detected=resolved["name"],name=resolved["name"],
                     name_match_conf=round(float(resolved.get("score",0)),3),
                     name_candidates=resolved.get("candidates",[]),
                     name_warning=resolved.get("warning","") or "",
                     name_suggested=resolved.get("suggested_name","") or "")
            r["panel_ocr_conf"]=round(sum(confs)/len(confs),3)
            r["ocr_missing_fields"]=[f for f in FIELDS if r.get(f) is None]
            rows.append(r)
        return rows

    @staticmethod
    def attach_soul_evidence(rows,panel_db):
        for r in rows:
            ev=panel_db.evidence(r.get("name",""),r)
            r.update(soul_inferred=ev["soul"],soul=ev["soul"],soul_match_score=ev["score"],
                     soul_margin=ev["margin"],soul_level=ev["level"],reference_record_id=ev["reference_record_id"],
                     suggestions=ev["suggestions"],
                     soul_candidates=" | ".join(f'{x["soul"]}:{x["score"]:.3f}' for x in ev["candidates"]),
                     confirmed=False)
            r["_match_signature"]=(r.get("name","") or "",)+tuple(r.get(f) for f in FIELDS)
        return rows
