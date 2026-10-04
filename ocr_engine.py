from __future__ import annotations
import re, cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR
from roi import unit_rois

FIELDS=["atk","hp","def","spd","cri","crid","efh","efr"]

class PanelOCR:
    """Fast fixed-layout OCR.

    V3 used 3 OCR passes/name + 4 passes/number: ~350 engine calls/match.
    V3.1 packs fixed ROIs into contact sheets, normally requiring only four
    engine calls for a two-sided match. Missing cells alone use a cheap fallback.
    """
    def __init__(self):
        self.ocr=RapidOCR()

    def _run_items(self,img):
        result,_=self.ocr(img)
        return result or []

    @staticmethod
    def _prep(img, scale=2.2, numeric=False):
        if img is None or img.size == 0:
            return np.full((80,240,3),255,np.uint8)
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        g=cv2.resize(g,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        g=cv2.createCLAHE(clipLimit=1.8,tileGridSize=(8,8)).apply(g)
        if numeric:
            # Keep anti-aliased glyphs; RapidOCR is usually better on grayscale than hard thresholding.
            g=cv2.convertScaleAbs(g,alpha=1.25,beta=4)
        return cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _sheet(cells, ncols, cell_w=420, cell_h=120, pad=12):
        nrows=(len(cells)+ncols-1)//ncols
        sheet=np.full((nrows*cell_h,ncols*cell_w,3),255,np.uint8)
        for i,im in enumerate(cells):
            r,c=divmod(i,ncols)
            h,w=im.shape[:2]
            scale=min((cell_w-2*pad)/max(w,1),(cell_h-2*pad)/max(h,1),1.0)
            if scale<1: im=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA); h,w=im.shape[:2]
            y=r*cell_h+(cell_h-h)//2; x=c*cell_w+(cell_w-w)//2
            sheet[y:y+h,x:x+w]=im
        return sheet,cell_w,cell_h

    @staticmethod
    def _center(box):
        a=np.asarray(box,dtype=float)
        return float(a[:,0].mean()),float(a[:,1].mean())

    def _recognize_sheet(self,cells,ncols,numeric=False):
        prepared=[self._prep(x,numeric=numeric) for x in cells]
        sheet,cw,ch=self._sheet(prepared,ncols)
        out=[[] for _ in cells]
        for item in self._run_items(sheet):
            if len(item)<3: continue
            try:
                x,y=self._center(item[0]); conf=float(item[2]); txt=str(item[1]).strip()
                c=min(ncols-1,max(0,int(x//cw))); r=max(0,int(y//ch)); idx=r*ncols+c
                if idx<len(out): out[idx].append((txt,conf))
            except Exception:
                continue
        return out

    @staticmethod
    def _best_number(items):
        candidates=[]
        for txt,conf in items:
            for n in re.findall(r"\d+(?:\.\d+)?",txt.replace(",","")):
                try: candidates.append((float(n),conf,txt))
                except Exception: pass
        if not candidates:return None,0.0,""
        value,conf,txt=max(candidates,key=lambda x:x[1])
        return int(value) if value.is_integer() else value,conf,txt

    def _fallback_number(self,img):
        # One engine call only, and only for cells missed by the sheet pass.
        items=[]
        for item in self._run_items(self._prep(img,scale=3.0,numeric=True)):
            if len(item)>=3: items.append((str(item[1]),float(item[2])))
        return self._best_number(items)

    def parse_side(self,img,side,panel_db):
        all_rois=[unit_rois(img,slot) for slot in range(1,6)]

        # One OCR call for all five names.
        name_hits=self._recognize_sheet([r["name"] for r in all_rois],ncols=5,numeric=False)
        # One OCR call for all 40 panel cells. Ordering is slot-major, 8 fields each.
        num_cells=[r[f] for r in all_rois for f in FIELDS]
        num_hits=self._recognize_sheet(num_cells,ncols=8,numeric=True)

        rows=[]
        for slot in range(1,6):
            nh=name_hits[slot-1]
            raw_name="".join(t for t,_ in sorted(nh,key=lambda x:-x[1])) if nh else ""
            nconf=max((c for _,c in nh),default=0.0)
            name,nmatch,ncands=panel_db.correct_name(raw_name)
            r={"side":side,"slot":slot,"name_detected":name,"name":name,"raw_name":raw_name,
               "name_ocr_conf":round(nconf,3),"name_match_conf":round(nmatch,3),"name_candidates":ncands}
            confs=[]
            for fi,f in enumerate(FIELDS):
                idx=(slot-1)*len(FIELDS)+fi
                v,c,raw=self._best_number(num_hits[idx])
                if v is None:
                    v,c,raw=self._fallback_number(all_rois[slot-1][f])
                r[f]=v; r[f"raw_{f}"]=raw; confs.append(c)
            r["panel_ocr_conf"]=round(sum(confs)/len(confs),3)
            ev=panel_db.evidence(name,r)
            r.update(soul_inferred=ev["soul"],soul=ev["soul"],soul_match_score=ev["score"],
                     soul_margin=ev["margin"],soul_level=ev["level"],reference_record_id=ev["reference_record_id"],
                     suggestions=ev["suggestions"],
                     soul_candidates=" | ".join(f'{x["soul"]}:{x["score"]:.3f}' for x in ev["candidates"]),
                     confirmed=False)
            r["_match_signature"]=(r.get("name",""),)+tuple(r.get(f) for f in FIELDS)
            rows.append(r)
        return rows
