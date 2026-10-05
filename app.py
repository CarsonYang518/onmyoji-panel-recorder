from __future__ import annotations
from pathlib import Path
from datetime import date
import time, hashlib
import cv2,numpy as np,pandas as pd,streamlit as st
from matcher import PanelDB,FIELDS
from ocr_engine import PanelOCR
from storage import Store
from security import gate,configured
from roi import locate_panel, draw_panel_preview, normalize_panel, COLS, SOUL_Y

ROOT=Path(__file__).parent; DATA=ROOT/"data/duel-panels.json"
st.set_page_config(page_title="阴阳师 · 对弈竞猜数据台",page_icon="⚔️",layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1500px}.hero{padding:18px 22px;border:1px solid rgba(128,128,128,.22);border-radius:18px;margin-bottom:12px}.muted{opacity:.72}.stTabs [data-baseweb=tab-list]{gap:8px}.stTabs [data-baseweb=tab]{border-radius:10px;padding:8px 14px}.unit-title{font-size:1.08rem;font-weight:700;margin-top:.15rem}.confidence{font-size:.82rem;opacity:.72}.redtag{color:#d9534f}.bluetag{color:#4285f4}
</style>""",unsafe_allow_html=True)

def file_sig(p):
    s=p.stat(); return (str(p),s.st_mtime_ns,s.st_size)
@st.cache_resource
def get_db(sig): return PanelDB(sig[0])
@st.cache_resource
def get_ocr(): return PanelOCR()
DB=get_db(file_sig(DATA)); STORE=Store(ROOT/"storage")

# Flash messages survive st.rerun(), then disappear after being shown once.
if "flash_success" in st.session_state:
    st.success(st.session_state.pop("flash_success"))
if "flash_warning" in st.session_state:
    st.warning(st.session_state.pop("flash_warning"))

def decode(b):return cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)


@st.cache_data(ttl=300,show_spinner=False)
def cached_soul_gallery_features():
    """Download/decode each historical crop once, then cache its feature matrix."""
    gallery=STORE.load_soul_gallery(limit=600)
    labels=[]; feats=[]
    for x in gallery:
        try:
            arr=np.frombuffer(x["image_bytes"],np.uint8)
            im=cv2.imdecode(arr,cv2.IMREAD_COLOR)
            f=_soul_feature(im)
            if f is not None:
                labels.append(x["soul"]); feats.append(f)
        except Exception:
            continue
    if not feats:
        return {"labels":[],"features":np.empty((0,0),dtype=np.float32)}
    return {"labels":labels,"features":np.ascontiguousarray(np.vstack(feats),dtype=np.float32)}

def _soul_feature(img):
    if img is None or img.size==0:return None
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img.copy()
    g=cv2.resize(g,(64,48),interpolation=cv2.INTER_AREA)
    g=cv2.equalizeHist(g)
    edge=cv2.Canny(g,45,130)
    a=g.astype(np.float32).reshape(-1); e=edge.astype(np.float32).reshape(-1)/255.0
    a=(a-a.mean())/(a.std()+1e-6)
    v=np.concatenate([a,e*1.5]).astype(np.float32); n=np.linalg.norm(v)
    return v/n if n>0 else v

def _image_soul_prediction(crop,gallery,topk=7):
    q=_soul_feature(crop); feats=gallery.get("features") if gallery else None; labels=gallery.get("labels",[]) if gallery else []
    if q is None or feats is None or feats.size==0:
        return {"soul":"","score":0.0,"n":0,"votes":0,"top_k_n":0,"candidates":[]}
    # All historical similarities in one vectorized matrix-vector operation.
    sims=np.clip((feats @ q + 1.0)/2.0,0.0,1.0)
    k=min(int(topk),len(sims)); idx=np.argpartition(sims,-k)[-k:]; idx=idx[np.argsort(sims[idx])[::-1]]
    agg={}
    for i in idx:
        label=labels[int(i)]; sim=float(sims[int(i)])
        z=agg.setdefault(label,{"sims":[],"votes":0}); z["sims"].append(sim); z["votes"]+=1
    cand=[]
    for label,z in agg.items():
        ss=sorted(z["sims"],reverse=True)[:3]; mean=float(np.mean(ss)); vote=z["votes"]/k
        score=mean*(.68+.32*vote)
        cand.append({"soul":label,"score":score,"similarity":mean,"votes":z["votes"]})
    cand.sort(key=lambda x:(x["score"],x["votes"]),reverse=True); b=cand[0]
    return {"soul":b["soul"],"score":round(b["score"],3),"n":len(labels),"votes":b["votes"],"top_k_n":k,"candidates":cand[:5]}

def _combine_soul_evidence(r, attr_soul, attr_score, attr_margin):
    """Conservative decision: a high match score without margin is not discriminative evidence."""
    img_soul=r.get("soul_image_pred",""); img_score=float(r.get("soul_image_score",0) or 0)
    img_votes=int(r.get("soul_image_votes",0) or 0); img_n=int(r.get("soul_image_n",0) or 0)
    attr_ready=bool(attr_soul) and attr_score>=.80 and attr_margin>=.05
    image_ready=bool(img_soul) and img_n>=3 and img_votes>=2 and img_score>=.72
    conflict=attr_ready and image_ready and attr_soul!=img_soul
    if attr_ready and image_ready and not conflict:
        return attr_soul, round(.55*attr_score+.45*img_score,3), "高", False, attr_ready, image_ready
    if conflict:
        return "", 0.0, "低", True, attr_ready, image_ready
    if attr_ready:
        return attr_soul, round(attr_score,3), "中", False, attr_ready, image_ready
    if image_ready:
        return img_soul, round(img_score,3), "中", False, attr_ready, image_ready
    return "", 0.0, "低", False, attr_ready, image_ready

def attach_image_soul_evidence(rows,red_img,blue_img,panel_locs):
    """Add cached image evidence, then make a conservative joint decision."""
    try: gallery=cached_soul_gallery_features()
    except Exception as e:
        gallery={"labels":[],"features":np.empty((0,0),dtype=np.float32)}
        for r in rows:r["soul_image_error"]=str(e)
    # Normalize each side once, not once per unit.
    panels={}
    for side,img in (("RED",red_img),("BLUE",blue_img)):
        try: panels[side],_=normalize_panel(img,panel_locs[side])
        except Exception: panels[side]=None
    for r in rows:
        attr_soul=r.get("soul_inferred",""); attr_score=float(r.get("soul_match_score",0) or 0); attr_margin=float(r.get("soul_margin",0) or 0)
        r["soul_attribute_pred"]=attr_soul; r["soul_attribute_score"]=attr_score
        try:
            panel=panels.get(r["side"])
            x1,x2=COLS[int(r["slot"])-1]; y1,y2=SOUL_Y; crop=panel[y1:y2,x1+38:x2-38]
            ip=_image_soul_prediction(crop,gallery)
        except Exception as e:
            ip={"soul":"","score":0.0,"n":0,"votes":0,"top_k_n":0,"candidates":[]}; r["soul_image_error"]=str(e)
        r["soul_image_pred"]=ip["soul"]; r["soul_image_score"]=ip["score"]; r["soul_image_votes"]=ip["votes"]; r["soul_image_n"]=ip["n"]; r["soul_image_top_k_n"]=ip["top_k_n"]
        r["soul_image_candidates"]=" | ".join(f'{x["soul"]}:{x["score"]:.3f}' for x in ip["candidates"])
        combined,score,level,conflict,ar,ir=_combine_soul_evidence(r,attr_soul,attr_score,attr_margin)
        r["soul_attribute_reliable"]=ar; r["soul_image_reliable"]=ir; r["soul_evidence_conflict"]=conflict
        r["soul_inferred"]=combined; r["soul"]=combined; r["soul_match_score"]=score; r["soul_level"]=level
    return rows

def recalc(r):
    old_inferred=r.get("soul_inferred",""); old_soul=r.get("soul","")
    ev=DB.evidence(r.get("name",""),r)
    attr_soul=ev["soul"]; attr_score=float(ev["score"] or 0); attr_margin=float(ev["margin"] or 0)
    r["soul_attribute_pred"]=attr_soul; r["soul_attribute_score"]=attr_score
    combined,score,level,conflict,ar,ir=_combine_soul_evidence(r,attr_soul,attr_score,attr_margin)
    r["soul_attribute_reliable"]=ar; r["soul_image_reliable"]=ir; r["soul_evidence_conflict"]=conflict
    r["soul_inferred"]=combined; r["soul_match_score"]=score; r["soul_margin"]=attr_margin; r["soul_level"]=level
    r["reference_record_id"]=ev["reference_record_id"]; r["suggestions"]=ev["suggestions"]
    r["soul_candidates"]=" | ".join(f'{x["soul"]}:{x["score"]:.3f}' for x in ev["candidates"])
    auto_changed=False
    if not old_soul or old_soul==old_inferred:
        r["soul"]=combined; auto_changed=(old_soul!=r["soul"])
    r["_match_signature"]=(r.get("name","")+"",)+tuple(r.get(f) for f in FIELDS)
    return auto_changed

def recognize_pair_with_progress(red_bytes: bytes, blue_bytes: bytes):
    """Run exactly two full-image OCR inferences and expose real stage progress."""
    progress=st.progress(0, text="0% · 准备识别")
    status=st.empty()
    t0=time.perf_counter(); timing={}

    def step(pct,msg):
        elapsed=time.perf_counter()-t0
        progress.progress(pct,text=f"{pct}% · {msg}")
        status.caption(f"当前：{msg}　·　已用时 {elapsed:.1f} 秒")

    step(5,"初始化 OCR 引擎")
    t=time.perf_counter(); ocr=get_ocr(); timing["OCR模型初始化"]=time.perf_counter()-t
    step(15,"OCR 引擎就绪")

    red_img,blue_img=decode(red_bytes),decode(blue_bytes)
    step(18,"定位红方『阵容详情』主面板")
    t=time.perf_counter(); red_loc=locate_panel(red_img); timing["红方面板定位"]=time.perf_counter()-t
    if not red_loc.get("ok"):
        progress.empty(); status.empty(); raise ValueError("无法可靠定位红方『阵容详情』主面板，请检查截图是否完整。")
    step(25,"定位蓝方『阵容详情』主面板")
    t=time.perf_counter(); blue_loc=locate_panel(blue_img); timing["蓝方面板定位"]=time.perf_counter()-t
    if not blue_loc.get("ok"):
        progress.empty(); status.empty(); raise ValueError("无法可靠定位蓝方『阵容详情』主面板，请检查截图是否完整。")

    step(32,"识别红方主面板（第 1/2 次 OCR）")
    t=time.perf_counter(); red_items=ocr.recognize_full(red_img,red_loc); timing["红方整图OCR"]=time.perf_counter()-t
    step(55,"红方 OCR 完成")

    step(58,"识别蓝方主面板（第 2/2 次 OCR）")
    t=time.perf_counter(); blue_items=ocr.recognize_full(blue_img,blue_loc); timing["蓝方整图OCR"]=time.perf_counter()-t
    step(80,"蓝方 OCR 完成")

    step(84,"在主面板归一化坐标中解析 10 个式神与 80 项面板")
    t=time.perf_counter(); rows=ocr.parse_items(red_items,"RED",DB)+ocr.parse_items(blue_items,"BLUE",DB); timing["坐标解析"]=time.perf_counter()-t
    step(91,"计算御魂候选")
    t=time.perf_counter(); rows=ocr.attach_soul_evidence(rows,DB)
    rows=attach_image_soul_evidence(rows,red_img,blue_img,{"RED":red_loc,"BLUE":blue_loc})
    timing["御魂候选"]=time.perf_counter()-t
    step(98,"整理识别结果")
    timing["总耗时"]=time.perf_counter()-t0
    progress.progress(100,text="100% · 识别完成")
    status.caption(f"完成 · 总耗时 {timing['总耗时']:.1f} 秒 · 本场固定 2 次整图 OCR")
    return rows,timing,{"RED":red_loc,"BLUE":blue_loc}



def build_soul_samples(mid, rows, red_bytes, blue_bytes, panel_locs):
    """Create 10 compact WebP crops from the already-located canonical panels."""
    by_key={(r["side"],int(r["slot"])):r for r in rows}
    samples=[]
    for side,bts in (("RED",red_bytes),("BLUE",blue_bytes)):
        img=decode(bts); loc=panel_locs.get(side) or {}
        panel,_=normalize_panel(img,loc)
        conf=float(loc.get("confidence",0) or 0)
        for slot in range(1,6):
            r=by_key[(side,slot)]
            x1,x2=COLS[slot-1]; y1,y2=SOUL_Y
            # Match roi.unit_rois(): remove side padding so text/borders contribute less.
            cx1,cx2=x1+38,x2-38
            crop=panel[y1:y2,cx1:cx2]
            ok,buf=cv2.imencode(".webp",crop,[cv2.IMWRITE_WEBP_QUALITY,90])
            if not ok: raise ValueError(f"{side} {slot} 御魂图片编码失败")
            samples.append({
                "side":side,"slot":slot,"shikigami_name":r.get("name","") or "",
                "soul_confirmed":r.get("soul","") or "","soul_auto":r.get("soul_inferred","") or "",
                "image_path":f"{mid}/{side}_{slot}.webp","image_bytes":buf.tobytes(),
                "crop_x1":cx1,"crop_y1":y1,"crop_x2":cx2,"crop_y2":y2,
                "panel_confidence":conf})
    return samples

def clear_editor_widget_state(prefix="new_"):
    for k in list(st.session_state.keys()):
        if str(k).startswith(prefix):
            del st.session_state[k]

def edit_unit(r,key):
    side=r["side"]; accent="🔴" if side=="RED" else "🔵"
    st.markdown(f'<div class="unit-title">{accent} {side} · {r["slot"]}</div>',unsafe_allow_html=True)

    old_name=r.get("name","")
    name_options=[""]+DB.names
    name=st.selectbox("式神",name_options,index=name_options.index(old_name) if old_name in name_options else 0,key=f"{key}_name")
    r["name"]=name; r["name_manually_changed"]=bool(name!=r.get("name_detected",""))

    # Editing stats never invokes OCR. Only this unit's cheap reference match is recomputed.
    a=st.columns(4); b=st.columns(4)
    for i,f in enumerate(FIELDS):
        col=(a+b)[i]; val=r.get(f)
        r[f]=col.number_input(f.upper(),value=float(val or 0),step=1.0,key=f"{key}_{f}")

    sig=(r.get("name","")+"",)+tuple(r.get(f) for f in FIELDS)
    auto_changed=False
    if sig != r.get("_match_signature"):
        auto_changed=recalc(r)

    souls=[""]+DB.souls; soul_key=f"{key}_soul"
    desired=r.get("soul","") if r.get("soul","") in souls else ""
    if soul_key not in st.session_state:
        st.session_state[soul_key]=desired
    elif auto_changed:
        st.session_state[soul_key]=desired
    soul=st.selectbox("御魂",souls,key=soul_key)
    r["soul"]=soul; r["soul_manually_changed"]=bool(soul!=r.get("soul_inferred",""))

    if st.button("重新匹配本式神御魂",key=f"{key}_recalc",use_container_width=True):
        recalc(r); st.rerun()
    level=r.get("soul_level","低")
    st.caption(f'推断：{r.get("soul_inferred") or "无法确定"} · 证据 {level} · score {r.get("soul_match_score",0):.3f} · 属性 margin {r.get("soul_margin",0):.3f}')
    def _candidate_buttons(title, candidate_text, prefix):
        """Render unique soul candidates as one-click choices without rerunning OCR/matching."""
        if not candidate_text:
            return
        parsed=[]
        seen=set()
        for part in str(candidate_text).split("|"):
            part=part.strip()
            if not part:
                continue
            name, sep, score_text = part.rpartition(":")
            name=name.strip() if sep else part
            score_text=score_text.strip() if sep else ""
            if not name or name in seen or name not in DB.souls:
                continue
            seen.add(name)
            parsed.append((name,score_text))
        if not parsed:
            return
        st.caption(title)
        cols=st.columns(min(len(parsed),4))
        def choose_soul(chosen):
            # Callback runs before the next script rerun, so changing the selectbox
            # session-state value here is safe. No OCR or image matching is invoked.
            st.session_state[soul_key]=chosen
        for i,(candidate,score_text) in enumerate(parsed):
            label=f"{candidate} {score_text}".strip()
            cols[i % len(cols)].button(
                label,
                key=f"{key}_{prefix}_{i}_{candidate}",
                use_container_width=True,
                on_click=choose_soul,
                args=(candidate,),
            )

    if r.get("soul_candidates"):
        _candidate_buttons("属性候选（点击即可填入）：",r["soul_candidates"],"attrcand")
    if r.get("soul_attribute_pred"):
        st.caption(f'属性预测：{r.get("soul_attribute_pred")} · {r.get("soul_attribute_score",0):.3f}')
        if not r.get("soul_attribute_reliable"):
            st.caption("↳ 属性证据不足：高 score 但候选区分度（margin）不足时不会自动定案。")
    if r.get("soul_image_pred"):
        st.caption(f'图片预测：{r.get("soul_image_pred")} · {r.get("soul_image_score",0):.3f} · Top-K {r.get("soul_image_votes",0)}/{r.get("soul_image_top_k_n",0)} 票 · 历史图库总样本 {r.get("soul_image_n",0)}')
        if r.get("soul_image_candidates"):
            _candidate_buttons("图片候选（点击即可填入）：",r["soul_image_candidates"],"imgcand")
        if not r.get("soul_image_reliable"):
            st.caption("↳ 图片证据不足：当前不会单独用于自动定案。")
    if r.get("soul_evidence_conflict"):
        st.warning(f'御魂证据冲突：属性→{r.get("soul_attribute_pred") or "—"}；图片→{r.get("soul_image_pred") or "—"}。请人工确认。')
    if r.get("name_warning"):
        st.warning(r["name_warning"])
        suggested=r.get("name_suggested","")
        if suggested and st.button(f"改为 {suggested}",key=f"{key}_namefix",use_container_width=True):
            r["name"]=suggested; st.session_state[f"{key}_name"]=suggested; recalc(r); st.rerun()
    sug=r.get("suggestions") or {}
    if sug:
        st.warning("疑似 OCR 异常："+"；".join(f"{f.upper()} {r.get(f)} → {v:g}" for f,v in sug.items()))
        if st.button("接受建议修正",key=f"{key}_fix",use_container_width=True):
            for f,v in sug.items():
                r[f]=v; st.session_state[f"{key}_{f}"]=float(v)
            recalc(r); st.rerun()
    r["confirmed"]=st.checkbox("已人工核验",value=bool(r.get("confirmed")),key=f"{key}_ok")
    return r

def clean_row(r):
    out={k:v for k,v in r.items() if not k.startswith("_")}; return out

st.markdown(f'<div class="hero"><h2 style="margin:0">⚔️ 阴阳师 · 对弈竞猜数据台</h2><div class="muted">Carson Yang</div><div class="confidence">参考库 {len(DB.rows)} 条 · {len(DB.names)} 式神 · {len(DB.souls)} 御魂 · version {DB.version}</div></div>',unsafe_allow_html=True)
if not configured():st.warning("当前没有配置 SAVE_PASSWORD：可以识别和浏览，但所有写入、修改、删除均被锁定。")

t1,t2,t3,t4=st.tabs(["✨ 新比赛","🏁 补录结果","🗂️ 历史管理","⬇️ 导出"])
with t1:
    c1,c2=st.columns(2); red=c1.file_uploader("红方阵容详情",["png","jpg","jpeg"],key="red"); blue=c2.file_uploader("蓝方阵容详情",["png","jpg","jpeg"],key="blue")
    if red:c1.image(red,width="stretch")
    if blue:c2.image(blue,width="stretch")
    if red and blue and st.button("✨ 智能识别双方",type="primary",use_container_width=True):
        rb,bb=red.getvalue(),blue.getvalue()
        image_pair_id=hashlib.sha256(rb+bb).hexdigest()[:16]
        # A normal widget rerun never enters this block. OCR only runs on this explicit button click.
        try:
            rows,timing,panel_locs=recognize_pair_with_progress(rb,bb)
        except ValueError as e:
            st.error(str(e)); st.stop()
        clear_editor_widget_state("new_")
        st.session_state.rows=rows; st.session_state.rb=rb; st.session_state.bb=bb
        st.session_state.ocr_done=True; st.session_state.ocr_timing=timing; st.session_state.panel_locs=panel_locs; st.session_state.image_pair_id=image_pair_id
        # Stable per image pair: repeated clicks cannot create another match row.
        st.session_state.current_match_id=f"img-{image_pair_id}"
        st.rerun()
    if "rows" in st.session_state:
        st.success("OCR 已完成。之后修改式神、属性、御魂、核验状态、备注或密码都不会再次运行 OCR；属性变化只重算对应式神的御魂候选。")
        timing=st.session_state.get("ocr_timing",{})
        if timing:
            with st.expander("⏱️ 本次识别性能诊断",expanded=False):
                cols=st.columns(min(5,len(timing)))
                for i,(k,v) in enumerate(timing.items()): cols[i%len(cols)].metric(k,f"{v:.2f}s")
                st.caption("V3.3 先自动定位『阵容详情』主面板，再在面板内部归一化坐标；正常路径每张截图仅 1 次 OCR。")
            locs=st.session_state.get("panel_locs",{})
            if locs:
                with st.expander("🎯 查看主面板定位结果",expanded=False):
                    pc1,pc2=st.columns(2)
                    for side,col,bts in (("RED",pc1,st.session_state.rb),("BLUE",pc2,st.session_state.bb)):
                        loc=locs.get(side,{})
                        with col:
                            st.caption(f'{side} · 定位置信度 {loc.get("confidence",0):.1%} · bbox {loc.get("bbox")}')
                            st.image(cv2.cvtColor(draw_panel_preview(decode(bts),loc),cv2.COLOR_BGR2RGB),width="stretch")
        missing=sum(len(r.get("ocr_missing_fields",[])) for r in st.session_state.rows)
        if missing: st.warning(f"整图 OCR 有 {missing}/80 个属性未识别。V3.3 为保证速度不会自动逐格重试，请在下方人工补充这些值。")
        st.divider(); left,right=st.columns(2)
        for side,col in (("RED",left),("BLUE",right)):
            with col:
                st.subheader("🔴 红方" if side=="RED" else "🔵 蓝方")
                for r in [x for x in st.session_state.rows if x["side"]==side]:
                    with st.container(border=True): edit_unit(r,f'new_{side}_{r["slot"]}')
        st.divider(); a,b,c=st.columns([1,1,2]); md=a.date_input("日期",date.today()); mt=b.text_input("场次/时间"); notes=c.text_input("备注")
        winner=st.selectbox("真实结果（未知可留空）",["","RED","BLUE","DRAW","INVALID"])
        ok=gate("保存密码","save_password")
        if st.button("💾 保存本场",type="primary",use_container_width=True,disabled=not ok):
            rows=[clean_row(x) for x in st.session_state.rows]
            if len(rows)!=10:st.error("必须有 10 个式神。")
            elif not all(x.get("confirmed") for x in rows):st.error("请人工核验并勾选全部 10 个式神。")
            else:
                mid=st.session_state.get("current_match_id") or f'img-{st.session_state.get("image_pair_id", hashlib.sha256(st.session_state.rb+st.session_state.bb).hexdigest()[:16])}'
                meta={"match_id":mid,"match_date":str(md),"match_time":mt,"winner":winner,"notes":notes,"reference_version":DB.version}
                mid=STORE.save_match(meta,rows,st.session_state.rb,st.session_state.bb)
                sample_msg=""
                try:
                    samples=build_soul_samples(mid,rows,st.session_state.rb,st.session_state.bb,st.session_state.get("panel_locs",{}))
                    n,errors=STORE.save_soul_samples(mid,samples)
                    if n>0:
                        cached_soul_gallery_features.clear()
                    if n==10:
                        sample_msg="；御魂图片样本 10/10 已保存"
                    else:
                        detail=("；"+"；".join(errors[:3])) if errors else ""
                        st.session_state["flash_warning"]=f"比赛 {mid} 已保存，但御魂图片样本仅保存 {n}/10{detail}"
                except Exception as e:
                    st.session_state["flash_warning"]=f"比赛 {mid} 已保存，但御魂图片样本未保存：{e}"
                if "flash_warning" not in st.session_state:
                    st.session_state["flash_success"]=f"比赛 {mid} 保存成功{sample_msg}。"
                # Clear all new-match state so the refreshed page is ready for the next match.
                clear_editor_widget_state("new_")
                for k in ["rows","rb","bb","ocr_done","ocr_timing","panel_locs","image_pair_id","current_match_id","red","blue","save_password"]:
                    st.session_state.pop(k,None)
                st.rerun()
with t2:
    m=STORE.matches(); pending=m[m.status=="PENDING"] if not m.empty else m
    if pending.empty:st.success("没有待补录结果的比赛。")
    else:
        st.dataframe(pending[["match_id","match_date","match_time","notes"]],hide_index=True,width="stretch")
        mid=st.selectbox("比赛",pending.match_id.tolist()); w=st.radio("真实结果",["RED","BLUE","DRAW","INVALID"],horizontal=True); ok=gate("管理密码","winner_password")
        if st.button("写入结果",type="primary",disabled=not ok):STORE.update_winner(mid,w);st.success("已更新");st.rerun()
with t3:
    m=STORE.matches()
    if m.empty:st.info("还没有历史数据。")
    else:
        q=st.text_input("搜索 match_id / 日期 / 备注")
        view=m.copy()
        if q:view=view[view.astype(str).apply(lambda x:x.str.contains(q,case=False,na=False)).any(axis=1)]
        st.dataframe(view[["match_id","match_date","match_time","winner","status","notes","updated_at"]],hide_index=True,width="stretch")
        mid=st.selectbox("选择历史比赛",view.match_id.tolist()); units=STORE.units(mid); meta=m[m.match_id==mid].iloc[0]
        st.subheader("比赛详情")
        edited=[]; L,R=st.columns(2)
        for side,col in (("RED",L),("BLUE",R)):
            with col:
                for _,x in units[units.side==side].iterrows():
                    r=x.to_dict(); r["confirmed"]=bool(r.get("confirmed")); r.setdefault("soul_level","")
                    with st.container(border=True): edited.append(edit_unit(r,f'hist_{mid}_{side}_{int(r["slot"])}'))
        ea,eb,ec=st.columns([1,1,2]); ed=ea.text_input("日期",str(meta.match_date),key="edit_date"); et=eb.text_input("时间",str(meta.match_time),key="edit_time"); en=ec.text_input("备注",str(meta.notes or ""),key="edit_notes"); ew=st.selectbox("结果",["","RED","BLUE","DRAW","INVALID"],index=["","RED","BLUE","DRAW","INVALID"].index(str(meta.winner or "")),key="edit_winner")
        ok=gate("管理密码","history_password"); x,y=st.columns(2)
        if x.button("保存历史修改",type="primary",use_container_width=True,disabled=not ok):
            if not all(r.get("confirmed") for r in edited):st.error("10 个式神都必须保持人工核验状态。")
            else:
                STORE.update_match(mid,{"match_date":ed,"match_time":et,"notes":en,"winner":ew},[clean_row(r) for r in edited])
                st.session_state["flash_success"]=f"比赛 {mid} 的修改已保存。"
                st.rerun()
        confirm_delete=y.checkbox("我确认删除整场比赛",key="confirm_delete")
        if y.button("🗑️ 删除比赛",use_container_width=True,disabled=not(ok and confirm_delete)):
            STORE.delete_match(mid)
            st.session_state["flash_success"]=f"比赛 {mid} 已删除。"
            st.rerun()
with t4:
    m=STORE.matches();u=STORE.units(); a,b,c=st.columns(3);a.metric("比赛",len(m));b.metric("式神记录",len(u));c.metric("待补结果",int((m.status=="PENDING").sum()) if not m.empty else 0)

    st.caption("所有数据导出均需要管理密码；导出文件不包含密码。")
    export_ok=gate("管理密码","export_password")

    for fn,label,mime in [
        ("matches.csv","matches.csv","text/csv"),
        ("units.csv","units.csv","text/csv"),
        ("matches.jsonl","matches.jsonl","application/json"),
    ]:
        p=ROOT/"storage"/fn
        if p.exists():
            st.download_button(
                f"下载 {label}",
                p.read_bytes(),
                fn,
                mime,
                use_container_width=True,
                disabled=not export_ok,
            )

    st.divider()
    st.subheader("🛡️ 数据库完整备份")
    db_path=ROOT/"storage"/"matches.sqlite3"

    if db_path.exists():
        st.caption(f"SQLite 数据库大小：{db_path.stat().st_size/1024:.1f} KB")
        st.download_button(
            "⬇️ 下载 matches.sqlite3 完整备份",
            data=db_path.read_bytes(),
            file_name="matches.sqlite3",
            mime="application/x-sqlite3",
            use_container_width=True,
            disabled=not export_ok,
        )
    else:
        st.warning("当前未找到 SQLite 数据库文件。")
