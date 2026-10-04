from __future__ import annotations
import sqlite3,json,uuid,shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
FIELDS=["atk","hp","def","spd","cri","crid","efh","efr"]
class Store:
    def __init__(self,root="storage"):
        self.root=Path(root); self.root.mkdir(exist_ok=True); (self.root/"screenshots").mkdir(exist_ok=True)
        self.db=self.root/"matches.sqlite3"; self.init()
    def con(self):
        c=sqlite3.connect(self.db,timeout=20); c.row_factory=sqlite3.Row; c.execute("PRAGMA foreign_keys=ON"); return c
    def init(self):
        with self.con() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS matches(match_id TEXT PRIMARY KEY,created_at TEXT,updated_at TEXT,match_date TEXT,match_time TEXT,winner TEXT,status TEXT,red_image TEXT,blue_image TEXT,notes TEXT,reference_version TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS units(match_id TEXT,side TEXT,slot INTEGER,name_detected TEXT,name TEXT,name_manually_changed INTEGER,soul_inferred TEXT,soul TEXT,soul_manually_changed INTEGER,atk REAL,hp REAL,def REAL,spd REAL,cri REAL,crid REAL,efh REAL,efr REAL,name_ocr_conf REAL,panel_ocr_conf REAL,soul_match_score REAL,soul_margin REAL,reference_record_id TEXT,soul_candidates TEXT,confirmed INTEGER,PRIMARY KEY(match_id,side,slot),FOREIGN KEY(match_id) REFERENCES matches(match_id) ON DELETE CASCADE)""")
    def _backup(self):
        if self.db.exists() and self.db.stat().st_size:
            d=self.root/"backups"; d.mkdir(exist_ok=True); shutil.copy2(self.db,d/f"matches-{datetime.now():%Y%m%d-%H%M%S}.sqlite3")
            backups=sorted(d.glob("*.sqlite3")); [p.unlink() for p in backups[:-20]]
    def save_match(self,meta,rows,red_bytes=None,blue_bytes=None):
        self._backup(); mid=meta.get("match_id") or datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6]; now=datetime.now().isoformat(timespec="seconds")
        paths={"RED":"","BLUE":""}
        for side,b in (("RED",red_bytes),("BLUE",blue_bytes)):
            if b:
                p=self.root/"screenshots"/f"{mid}_{side}.png"; p.write_bytes(b); paths[side]=str(p)
        with self.con() as c:
            c.execute("INSERT INTO matches VALUES(?,?,?,?,?,?,?,?,?,?,?)",(mid,now,now,meta.get("match_date",""),meta.get("match_time",""),meta.get("winner","") or "","COMPLETE" if meta.get("winner") else "PENDING",paths["RED"],paths["BLUE"],meta.get("notes",""),meta.get("reference_version","")))
            self._write_units(c,mid,rows)
        self.export(); return mid
    def _write_units(self,c,mid,rows):
        for r in rows:
            vals=[mid,r["side"],int(r["slot"]),r.get("name_detected",""),r.get("name",""),int(r.get("name","")!=r.get("name_detected","")),r.get("soul_inferred",""),r.get("soul",""),int(r.get("soul","")!=r.get("soul_inferred",""))]+[r.get(f) for f in FIELDS]+[r.get("name_ocr_conf"),r.get("panel_ocr_conf"),r.get("soul_match_score"),r.get("soul_margin"),r.get("reference_record_id",""),r.get("soul_candidates",""),int(bool(r.get("confirmed",False)))]
            c.execute("INSERT OR REPLACE INTO units VALUES("+",".join("?"*len(vals))+")",vals)
    def update_winner(self,mid,winner):
        self._backup();
        with self.con() as c:c.execute("UPDATE matches SET winner=?,status=?,updated_at=? WHERE match_id=?",(winner,"COMPLETE" if winner else "PENDING",datetime.now().isoformat(timespec="seconds"),mid))
        self.export()
    def update_match(self,mid,meta,rows):
        self._backup();
        with self.con() as c:
            c.execute("UPDATE matches SET match_date=?,match_time=?,winner=?,status=?,notes=?,updated_at=? WHERE match_id=?",(meta.get("match_date",""),meta.get("match_time",""),meta.get("winner","") or "","COMPLETE" if meta.get("winner") else "PENDING",meta.get("notes",""),datetime.now().isoformat(timespec="seconds"),mid))
            c.execute("DELETE FROM units WHERE match_id=?",(mid,)); self._write_units(c,mid,rows)
        self.export()
    def delete_match(self,mid):
        self._backup();
        with self.con() as c:
            imgs=c.execute("SELECT red_image,blue_image FROM matches WHERE match_id=?",(mid,)).fetchone(); c.execute("DELETE FROM matches WHERE match_id=?",(mid,))
        if imgs:
            for x in imgs:
                try:
                    if x: Path(x).unlink(missing_ok=True)
                except:pass
        self.export()
    def matches(self):
        with self.con() as c:return pd.read_sql_query("SELECT * FROM matches ORDER BY created_at DESC",c)
    def units(self,mid=None):
        q="SELECT u.*,m.match_date,m.match_time,m.winner,m.status FROM units u JOIN matches m USING(match_id)"; args=[]
        if mid:q+=" WHERE u.match_id=?"; args=[mid]
        q+=" ORDER BY m.created_at DESC,u.side,u.slot"
        with self.con() as c:return pd.read_sql_query(q,c,params=args)
    def export(self):
        m=self.matches(); u=self.units(); m.to_csv(self.root/"matches.csv",index=False,encoding="utf-8-sig"); u.to_csv(self.root/"units.csv",index=False,encoding="utf-8-sig")
        tmp=self.root/"matches.jsonl.tmp"; final=self.root/"matches.jsonl"
        with open(tmp,"w",encoding="utf8") as f:
            for _,mr in m.iterrows():
                d=mr.to_dict(); d["units"]=u[u.match_id==mr.match_id].to_dict("records"); f.write(json.dumps(d,ensure_ascii=False,default=str)+"\n")
        tmp.replace(final)
