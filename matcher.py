from __future__ import annotations
import json, math, hashlib
from pathlib import Path
from rapidfuzz import process, fuzz

FIELDS = ["atk","hp","def","spd","cri","crid","efh","efr"]
SCALES = {"atk":100,"hp":750,"def":30,"spd":4,"cri":4,"crid":7,"efh":5,"efr":5}
WEIGHTS = {"atk":1.0,"hp":1.0,"def":1.0,"spd":1.55,"cri":1.25,"crid":1.25,"efh":1.25,"efr":1.25}

class PanelDB:
    def __init__(self, path):
        self.path=Path(path); raw=self.path.read_bytes(); self.version=hashlib.sha256(raw).hexdigest()[:12]
        self.rows=json.loads(raw.decode("utf-8"))
        self.names=sorted({str(r.get("name","")).strip() for r in self.rows if r.get("name")})
        self.souls=sorted({str(r.get("build","")).strip() for r in self.rows if r.get("build")})
        self.by_name={}
        for r in self.rows:
            n=str(r.get("name","")).strip()
            if n: self.by_name.setdefault(n,[]).append(r)

    def correct_name(self, raw: str, topk=5):
        raw=(raw or "").strip().replace(" ","")
        if not raw: return "",0.0,[]
        if raw in self.by_name: return raw,1.0,[(raw,1.0)]
        hits=process.extract(raw,self.names,scorer=fuzz.WRatio,limit=topk)
        c=[(h[0],h[1]/100.0) for h in hits]
        return (c[0][0],c[0][1],c) if c else (raw,0.0,[])

    @staticmethod
    def ref_values(v, field):
        vals=v if isinstance(v,list) else [v]; out=[]
        for x in vals:
            if x is None: continue
            try: x=float(x)
            except Exception: continue
            if field in {"cri","crid","efh","efr"} and abs(x)<=5: x*=100.0
            out.append(x)
        return out

    def compare_record(self,panel,rec):
        details={}; weighted=totalw=0.0; exact=used=0
        for f in FIELDS:
            x=panel.get(f)
            if x is None: continue
            vals=self.ref_values(rec.get(f),f)
            if not vals: continue
            best=min(vals,key=lambda z:abs(float(x)-z)); err=abs(float(x)-best)
            sim=math.exp(-err/SCALES[f]); w=WEIGHTS[f]
            weighted+=w*sim; totalw+=w; used+=1
            tol=.5 if f in {"cri","crid","efh","efr"} else (1 if f in {"spd","def"} else 2)
            if err<=tol: exact+=1
            details[f]={"observed":x,"reference":best,"error":err,"similarity":sim}
        if not totalw:return 0.0,0,0,details
        coverage=used/len(FIELDS)
        score=(.78*(weighted/totalw)+.22*(exact/max(1,used)))*(.72+.28*coverage)
        return min(1.0,score),exact,used,details

    def infer_soul(self,name,panel,topk=5):
        best={}
        for r in self.by_name.get(name,[]):
            score,exact,used,details=self.compare_record(panel,r)
            x={"soul":str(r.get("build","")).strip(),"score":score,"exact":exact,"used":used,
               "recordId":r.get("recordId",""),"details":details,"record":r}
            if x["soul"] and (x["soul"] not in best or score>best[x["soul"]]["score"]): best[x["soul"]]=x
        return sorted(best.values(),key=lambda x:(x["score"],x["exact"]),reverse=True)[:topk]

    def evidence(self,name,panel):
        cand=self.infer_soul(name,panel,5)
        if not cand:return {"soul":"","score":0.0,"margin":0.0,"candidates":[],"reference_record_id":"","suggestions":{},"level":"无参考"}
        b=cand[0]; second=cand[1]["score"] if len(cand)>1 else 0.0; margin=b["score"]-second
        if b["score"]>=.92 and margin>=.08: level="高"
        elif b["score"]>=.78 and margin>=.04: level="中"
        else: level="低"
        return {"soul":b["soul"],"score":round(b["score"],3),"margin":round(margin,3),"candidates":cand,
                "reference_record_id":b["recordId"],"suggestions":self.correction_suggestions(b),"level":level}

    def correction_suggestions(self,best,min_other_exact=5):
        if not best or best["used"]<6:return {}
        ds=best["details"]; out={}
        for f,d in ds.items():
            other=sum(1 for k,v in ds.items() if k!=f and v["error"] <= (.5 if k in {"cri","crid","efh","efr"} else (1 if k in {"spd","def"} else 2)))
            if other>=min_other_exact and d["similarity"]<.30: out[f]=d["reference"]
        return out
