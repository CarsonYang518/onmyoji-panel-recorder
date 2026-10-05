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
        # Automatically discover names such as 酒吞童子 < 鬼王酒吞童子.
        self.related_names={n:[] for n in self.names}
        for a in self.names:
            for b in self.names:
                if a != b and (a in b or b in a): self.related_names[a].append(b)

    @staticmethod
    def _norm_name(raw):
        return (raw or "").strip().replace(" ","").replace("·","")

    def correct_name(self, raw: str, topk=5):
        """Text-only correction. Exact legal names always win."""
        raw=self._norm_name(raw)
        if not raw: return "",0.0,[]
        if raw in self.by_name: return raw,1.0,[(raw,1.0)]
        hits=process.extract(raw,self.names,scorer=fuzz.WRatio,limit=topk)
        c=[(h[0],h[1]/100.0) for h in hits]
        return (c[0][0],c[0][1],c) if c else (raw,0.0,[])

    def name_panel_score(self,name,panel):
        """Best 8-stat agreement for a shikigami across all reference builds."""
        best=0.0
        for rec in self.by_name.get(name,[]):
            score,_,used,_=self.compare_record(panel,rec)
            if used >= 4: best=max(best,score)
        return best

    def resolve_name(self,raw,panel,topk=5):
        """Resolve OCR name with stats while protecting exact legal short names.

        Exact OCR is never silently replaced. If a related longer/shorter legal name
        fits the stats much better, it is returned as a warning suggestion only.
        Non-exact OCR uses text + panel agreement to choose among candidates.
        """
        raw=self._norm_name(raw)
        text_name,text_score,text_candidates=self.correct_name(raw,topk=max(topk,8))
        if not raw:
            return {"name":"","score":0.0,"candidates":[],"warning":"","suggested_name":""}

        if raw in self.by_name:
            base_panel=self.name_panel_score(raw,panel)
            alternatives=[]
            for n in self.related_names.get(raw,[]):
                ps=self.name_panel_score(n,panel)
                alternatives.append((n,ps))
            alternatives.sort(key=lambda x:x[1],reverse=True)
            suggested=""; warning=""
            if alternatives:
                n,ps=alternatives[0]
                # Deliberately conservative: exact OCR remains authoritative.
                if ps >= .88 and ps-base_panel >= .16:
                    suggested=n
                    warning=f'OCR 精确识别为「{raw}」，但属性明显更符合「{n}」({ps:.3f} vs {base_panel:.3f})，请人工确认。'
            return {"name":raw,"score":1.0,"candidates":[(raw,1.0)],
                    "warning":warning,"suggested_name":suggested,"panel_score":base_panel}

        # Candidate pool: fuzzy hits plus substring-related names. No hard-coded names.
        pool={n:s for n,s in text_candidates}
        for n in self.names:
            if raw in n or n in raw:
                pool[n]=max(pool.get(n,0.0),fuzz.WRatio(raw,n)/100.0)
        ranked=[]
        for n,ts in pool.items():
            ps=self.name_panel_score(n,panel)
            # Text remains primary; stats break ambiguous/partial OCR cases.
            combined=.68*ts+.32*ps
            ranked.append({"name":n,"text":ts,"panel":ps,"score":combined})
        ranked.sort(key=lambda x:(x["score"],x["text"],x["panel"]),reverse=True)
        if not ranked:
            return {"name":text_name,"score":text_score,"candidates":text_candidates,"warning":"","suggested_name":""}
        best=ranked[0]
        return {"name":best["name"],"score":best["score"],
                "candidates":[(x["name"],x["score"]) for x in ranked[:topk]],
                "warning":"","suggested_name":"","panel_score":best["panel"]}

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
