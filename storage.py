from __future__ import annotations
import os, sqlite3, json, uuid, shutil
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from pathlib import Path
from datetime import datetime
import pandas as pd

FIELDS=["atk","hp","def","spd","cri","crid","efh","efr"]

MATCH_COLS=["match_id","created_at","updated_at","match_date","match_time","winner","status","red_image","blue_image","notes","reference_version"]
UNIT_COLS=["match_id","side","slot","name_detected","name","name_manually_changed","soul_inferred","soul","soul_manually_changed",
           "atk","hp","def","spd","cri","crid","efh","efr","name_ocr_conf","panel_ocr_conf","soul_match_score","soul_margin",
           "reference_record_id","soul_candidates","confirmed"]

class Store:
    def __init__(self,root="storage"):
        self.root=Path(root); self.root.mkdir(exist_ok=True); (self.root/"screenshots").mkdir(exist_ok=True)
        self.database_url=self._database_url()
        self.backend="postgres" if self.database_url else "sqlite"
        self.db=self.root/"matches.sqlite3"
        if self.backend=="sqlite":
            self.init()

    def _database_url(self):
        # Streamlit Cloud: st.secrets["DATABASE_URL"]; local fallback: env var.
        try:
            import streamlit as st
            v=st.secrets.get("DATABASE_URL","")
            if v: return str(v).strip()
        except Exception:
            pass
        return os.environ.get("DATABASE_URL","").strip()

    def con(self):
        if self.backend=="postgres":
            import psycopg
            return psycopg.connect(self.database_url,connect_timeout=15,prepare_threshold=None,)
        c=sqlite3.connect(self.db,timeout=20); c.row_factory=sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def init(self):
        with self.con() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS matches(match_id TEXT PRIMARY KEY,created_at TEXT,updated_at TEXT,match_date TEXT,match_time TEXT,winner TEXT,status TEXT,red_image TEXT,blue_image TEXT,notes TEXT,reference_version TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS units(match_id TEXT,side TEXT,slot INTEGER,name_detected TEXT,name TEXT,name_manually_changed INTEGER,soul_inferred TEXT,soul TEXT,soul_manually_changed INTEGER,atk REAL,hp REAL,def REAL,spd REAL,cri REAL,crid REAL,efh REAL,efr REAL,name_ocr_conf REAL,panel_ocr_conf REAL,soul_match_score REAL,soul_margin REAL,reference_record_id TEXT,soul_candidates TEXT,confirmed INTEGER,PRIMARY KEY(match_id,side,slot),FOREIGN KEY(match_id) REFERENCES matches(match_id) ON DELETE CASCADE)""")

    def _backup(self):
        # PostgreSQL is persistent cloud storage; local file backups apply only to SQLite fallback.
        if self.backend!="sqlite": return
        if self.db.exists() and self.db.stat().st_size:
            d=self.root/"backups"; d.mkdir(exist_ok=True)
            shutil.copy2(self.db,d/f"matches-{datetime.now():%Y%m%d-%H%M%S}.sqlite3")
            backups=sorted(d.glob("*.sqlite3"))
            [p.unlink() for p in backups[:-20]]

    def _ph(self,n):
        return ",".join(["%s" if self.backend=="postgres" else "?"]*n)

    def _read(self,q,args=()):
        if self.backend=="postgres":
            with self.con() as c:
                with c.cursor() as cur:
                    cur.execute(q,list(args))
                    rows=cur.fetchall()
                    cols=[d.name for d in cur.description]
                    return pd.DataFrame(rows,columns=cols)
        with self.con() as c:
            return pd.read_sql_query(q,c,params=list(args))

    def _secret(self,name):
        try:
            import streamlit as st
            v=st.secrets.get(name,"")
            if v: return str(v).strip()
        except Exception:
            pass
        return os.environ.get(name,"").strip()

    def storage_configured(self):
        return bool(self._secret("SUPABASE_URL") and self._secret("SUPABASE_SECRET_KEY"))

    def _upload_storage(self,path,data,content_type="image/webp"):
        base=self._secret("SUPABASE_URL").rstrip("/")
        key=self._secret("SUPABASE_SECRET_KEY")
        if not base or not key:
            raise RuntimeError("未配置 SUPABASE_URL / SUPABASE_SECRET_KEY")
        encoded="/".join(quote(x,safe="") for x in path.split("/"))
        url=f"{base}/storage/v1/object/soul-icons/{encoded}"
        req=Request(url,data=data,method="POST",headers={
            "apikey":key,
            "Content-Type":content_type,"x-upsert":"true",
            "User-Agent":"onmyoji-panel-recorder/1.0"})
        try:
            with urlopen(req,timeout=30) as resp:
                if not (200 <= resp.status < 300):
                    body=resp.read().decode("utf-8",errors="replace")
                    raise RuntimeError(f"Storage HTTP {resp.status}: {body}")
        except HTTPError as e:
            body=e.read().decode("utf-8",errors="replace")
            raise RuntimeError(f"Storage HTTP {e.code}: {body}") from e

    def _download_storage(self,path):
        """Download one object from the private soul-icons bucket using the server secret."""
        base=self._secret("SUPABASE_URL").rstrip("/")
        key=self._secret("SUPABASE_SECRET_KEY")
        if not base or not key:
            raise RuntimeError("未配置 SUPABASE_URL / SUPABASE_SECRET_KEY")
        encoded="/".join(quote(x,safe="") for x in path.split("/"))
        url=f"{base}/storage/v1/object/authenticated/soul-icons/{encoded}"
        req=Request(url,method="GET",headers={"apikey":key,"User-Agent":"onmyoji-panel-recorder/1.0"})
        try:
            with urlopen(req,timeout=30) as resp:
                return resp.read()
        except HTTPError as e:
            body=e.read().decode("utf-8",errors="replace")
            raise RuntimeError(f"Storage HTTP {e.code}: {body}") from e

    def soul_samples(self,limit=600):
        """Return recent human-labelled soul image metadata."""
        if self.backend!="postgres": return pd.DataFrame()
        limit=max(1,min(int(limit),3000))
        return self._read(f"""SELECT sample_id,match_id,side,slot,shikigami_name,soul_confirmed,soul_auto,
                              image_path,panel_confidence,created_at
                       FROM soul_samples
                       WHERE soul_confirmed IS NOT NULL AND soul_confirmed <> ''
                       ORDER BY created_at DESC LIMIT {limit}""")

    def load_soul_gallery(self,limit=600):
        """Load labelled private-bucket crops. Bad/missing objects are skipped."""
        meta=self.soul_samples(limit); out=[]
        if meta.empty: return out
        for _,r in meta.iterrows():
            try:
                out.append({"soul":str(r.soul_confirmed),"path":str(r.image_path),
                            "shikigami":str(r.shikigami_name or ""),
                            "image_bytes":self._download_storage(str(r.image_path))})
            except Exception:
                continue
        return out

    def save_soul_samples(self,mid,samples):
        """Upload confirmed soul crops, then index successful uploads in soul_samples.

        This is deliberately separate from save_match(): image-storage failure must never
        roll back an otherwise valid match. Returns (saved_count, errors).
        """
        if self.backend!="postgres":
            return 0,["当前不是 PostgreSQL 后端，未上传御魂样本"]
        saved=0; errors=[]
        sql="""INSERT INTO soul_samples
               (match_id,side,slot,shikigami_name,soul_confirmed,soul_auto,image_path,
                crop_x1,crop_y1,crop_x2,crop_y2,panel_confidence)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (match_id,side,slot) DO UPDATE SET
               shikigami_name=EXCLUDED.shikigami_name,
               soul_confirmed=EXCLUDED.soul_confirmed,soul_auto=EXCLUDED.soul_auto,
               image_path=EXCLUDED.image_path,crop_x1=EXCLUDED.crop_x1,crop_y1=EXCLUDED.crop_y1,
               crop_x2=EXCLUDED.crop_x2,crop_y2=EXCLUDED.crop_y2,
               panel_confidence=EXCLUDED.panel_confidence"""
        for x in samples:
            try:
                self._upload_storage(x["image_path"],x["image_bytes"])
                vals=(mid,x["side"],int(x["slot"]),x.get("shikigami_name","") or "",
                      x.get("soul_confirmed","") or "",x.get("soul_auto","") or "",x["image_path"],
                      int(x["crop_x1"]),int(x["crop_y1"]),int(x["crop_x2"]),int(x["crop_y2"]),
                      x.get("panel_confidence"))
                with self.con() as c:
                    with c.cursor() as cur: cur.execute(sql,vals)
                saved+=1
            except Exception as e:
                errors.append(f'{x.get("side")} {x.get("slot")}: {e}')
        return saved,errors

    def sync_soul_sample_labels(self,mid,rows):
        """Keep an existing sample label aligned with later human history edits."""
        if self.backend!="postgres": return
        with self.con() as c:
            with c.cursor() as cur:
                for r in rows:
                    cur.execute("""UPDATE soul_samples SET shikigami_name=%s,soul_confirmed=%s,soul_auto=%s
                                   WHERE match_id=%s AND side=%s AND slot=%s""",
                                (r.get("name","") or "",r.get("soul","") or "",r.get("soul_inferred","") or "",
                                 mid,r["side"],int(r["slot"])))

    def save_match(self,meta,rows,red_bytes=None,blue_bytes=None):
        self._backup()
        mid=meta.get("match_id") or datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6]
        now=datetime.now().astimezone().isoformat(timespec="seconds")
        # Screenshots remain optional local artifacts. The canonical structured data is in PostgreSQL.
        paths={"RED":"","BLUE":""}
        if self.backend=="sqlite":
            for side,b in (("RED",red_bytes),("BLUE",blue_bytes)):
                if b:
                    p=self.root/"screenshots"/f"{mid}_{side}.png"; p.write_bytes(b); paths[side]=str(p)
        vals=(mid,now,now,meta.get("match_date",""),meta.get("match_time",""),meta.get("winner","") or "",
              "COMPLETE" if meta.get("winner") else "PENDING",paths["RED"],paths["BLUE"],meta.get("notes",""),meta.get("reference_version",""))
        with self.con() as c:
            cur=c.cursor() if self.backend=="postgres" else c
            if self.backend=="postgres":
                cur.execute(f"""INSERT INTO matches ({','.join(MATCH_COLS)}) VALUES ({self._ph(len(MATCH_COLS))})
                               ON CONFLICT (match_id) DO UPDATE SET
                               updated_at=EXCLUDED.updated_at,match_date=EXCLUDED.match_date,match_time=EXCLUDED.match_time,
                               winner=EXCLUDED.winner,status=EXCLUDED.status,notes=EXCLUDED.notes,
                               reference_version=EXCLUDED.reference_version""",vals)
            else:
                cur.execute(f"INSERT OR REPLACE INTO matches ({','.join(MATCH_COLS)}) VALUES ({self._ph(len(MATCH_COLS))})",vals)
            self._write_units(cur,mid,rows)
        self.export(); return mid

    def _write_units(self,c,mid,rows):
        for r in rows:
            vals=[mid,r["side"],int(r["slot"]),r.get("name_detected",""),r.get("name",""),
                  int(r.get("name","")!=r.get("name_detected","")),r.get("soul_inferred",""),r.get("soul",""),
                  int(r.get("soul","")!=r.get("soul_inferred",""))]+[r.get(f) for f in FIELDS]+[
                  r.get("name_ocr_conf"),r.get("panel_ocr_conf"),r.get("soul_match_score"),r.get("soul_margin"),
                  r.get("reference_record_id",""),r.get("soul_candidates",""),int(bool(r.get("confirmed",False)))]
            if self.backend=="postgres":
                c.execute(f"""INSERT INTO units ({','.join(UNIT_COLS)}) VALUES ({self._ph(len(vals))})
                              ON CONFLICT (match_id,side,slot) DO UPDATE SET
                              {','.join(f"{x}=EXCLUDED.{x}" for x in UNIT_COLS[3:])}""",vals)
            else:
                c.execute("INSERT OR REPLACE INTO units VALUES("+self._ph(len(vals))+")",vals)

    def update_winner(self,mid,winner):
        self._backup(); now=datetime.now().astimezone().isoformat(timespec="seconds")
        p="%s" if self.backend=="postgres" else "?"
        with self.con() as c:
            cur=c.cursor() if self.backend=="postgres" else c
            cur.execute(f"UPDATE matches SET winner={p},status={p},updated_at={p} WHERE match_id={p}",
                        (winner,"COMPLETE" if winner else "PENDING",now,mid))
        self.export()

    def update_match(self,mid,meta,rows):
        self._backup(); now=datetime.now().astimezone().isoformat(timespec="seconds")
        p="%s" if self.backend=="postgres" else "?"
        with self.con() as c:
            cur=c.cursor() if self.backend=="postgres" else c
            cur.execute(f"UPDATE matches SET match_date={p},match_time={p},winner={p},status={p},notes={p},updated_at={p} WHERE match_id={p}",
                        (meta.get("match_date",""),meta.get("match_time",""),meta.get("winner","") or "",
                         "COMPLETE" if meta.get("winner") else "PENDING",meta.get("notes",""),now,mid))
            cur.execute(f"DELETE FROM units WHERE match_id={p}",(mid,))
            self._write_units(cur,mid,rows)
        self.sync_soul_sample_labels(mid,rows)
        self.export()

    def delete_match(self,mid):
        self._backup(); p="%s" if self.backend=="postgres" else "?"
        imgs=None
        with self.con() as c:
            cur=c.cursor() if self.backend=="postgres" else c
            cur.execute(f"SELECT red_image,blue_image FROM matches WHERE match_id={p}",(mid,))
            imgs=cur.fetchone()
            cur.execute(f"DELETE FROM matches WHERE match_id={p}",(mid,))
        if self.backend=="sqlite" and imgs:
            for x in imgs:
                try:
                    if x: Path(x).unlink(missing_ok=True)
                except Exception: pass
        self.export()

    def matches(self):
        return self._read("SELECT * FROM matches ORDER BY created_at DESC")

    def units(self,mid=None):
        p="%s" if self.backend=="postgres" else "?"
        q="SELECT u.*,m.match_date,m.match_time,m.winner,m.status FROM units u JOIN matches m USING(match_id)"
        args=[]
        if mid: q+=f" WHERE u.match_id={p}"; args=[mid]
        q+=" ORDER BY m.created_at DESC,u.side,u.slot"
        return self._read(q,args)

    def export(self):
        m=self.matches(); u=self.units()
        m.to_csv(self.root/"matches.csv",index=False,encoding="utf-8-sig")
        u.to_csv(self.root/"units.csv",index=False,encoding="utf-8-sig")
        tmp=self.root/"matches.jsonl.tmp"; final=self.root/"matches.jsonl"
        with open(tmp,"w",encoding="utf8") as f:
            for _,mr in m.iterrows():
                d=mr.to_dict()
                d["units"]=u[u.match_id==mr.match_id].to_dict("records")
                f.write(json.dumps(d,ensure_ascii=False,default=str)+"\n")
        tmp.replace(final)
