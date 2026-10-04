from __future__ import annotations
from pathlib import Path
from datetime import date
import time, hashlib
import cv2,numpy as np,pandas as pd,streamlit as st
from matcher import PanelDB,FIELDS
from ocr_engine import PanelOCR
from storage import Store
from security import gate,configured
from roi import locate_panel, draw_panel_preview

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
def decode(b):return cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)

def recalc(r):
    old_inferred=r.get("soul_inferred","")
    old_soul=r.get("soul","")
    ev=DB.evidence(r.get("name",""),r)
    r["soul_inferred"]=ev["soul"]; r["soul_match_score"]=ev["score"]; r["soul_margin"]=ev["margin"]
    r["soul_level"]=ev["level"]; r["reference_record_id"]=ev["reference_record_id"]; r["suggestions"]=ev["suggestions"]
    r["soul_candidates"]=" | ".join(f'{x["soul"]}:{x["score"]:.3f}' for x in ev["candidates"])
    auto_changed=False
    if not old_soul or old_soul==old_inferred:
        r["soul"]=ev["soul"]; auto_changed=(old_soul!=r["soul"])
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
    t=time.perf_counter(); rows=ocr.attach_soul_evidence(rows,DB); timing["御魂候选"]=time.perf_counter()-t
    step(98,"整理识别结果")
    timing["总耗时"]=time.perf_counter()-t0
    progress.progress(100,text="100% · 识别完成")
    status.caption(f"完成 · 总耗时 {timing['总耗时']:.1f} 秒 · 本场固定 2 次整图 OCR")
    return rows,timing,{"RED":red_loc,"BLUE":blue_loc}

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
    if auto_changed and soul_key in st.session_state:
        st.session_state[soul_key]=r.get("soul","")
    old_soul=r.get("soul","")
    soul=st.selectbox("御魂",souls,index=souls.index(old_soul) if old_soul in souls else 0,key=soul_key)
    r["soul"]=soul; r["soul_manually_changed"]=bool(soul!=r.get("soul_inferred",""))

    if st.button("重新匹配本式神御魂",key=f"{key}_recalc",use_container_width=True):
        recalc(r); st.rerun()
    level=r.get("soul_level","低")
    st.caption(f'推断：{r.get("soul_inferred") or "—"} · 证据 {level} · score {r.get("soul_match_score",0):.3f} · margin {r.get("soul_margin",0):.3f}')
    if r.get("soul_candidates"):st.caption("候选："+r["soul_candidates"])
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

st.markdown(f'<div class="hero"><h2 style="margin:0">⚔️ 阴阳师 · 对弈竞猜数据台</h2><div class="muted">CarsonYang</div><div class="confidence">参考库 {len(DB.rows)} 条 · {len(DB.names)} 式神 · {len(DB.souls)} 御魂 · version {DB.version}</div></div>',unsafe_allow_html=True)
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
                mid=STORE.save_match({"match_date":str(md),"match_time":mt,"winner":winner,"notes":notes,"reference_version":DB.version},rows,st.session_state.rb,st.session_state.bb); st.success(f"已保存 {mid}"); del st.session_state["rows"]
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
            else:STORE.update_match(mid,{"match_date":ed,"match_time":et,"notes":en,"winner":ew},[clean_row(r) for r in edited]);st.success("修改已保存");st.rerun()
        confirm_delete=y.checkbox("我确认删除整场比赛",key="confirm_delete")
        if y.button("🗑️ 删除比赛",use_container_width=True,disabled=not(ok and confirm_delete)):STORE.delete_match(mid);st.success("已删除");st.rerun()
with t4:
    m=STORE.matches();u=STORE.units(); a,b,c=st.columns(3);a.metric("比赛",len(m));b.metric("式神记录",len(u));c.metric("待补结果",int((m.status=="PENDING").sum()) if not m.empty else 0)
    st.caption("导出为只读操作，不需要管理密码；导出文件不包含密码。")
    for fn,label,mime in [("matches.csv","matches.csv","text/csv"),("units.csv","units.csv","text/csv"),("matches.jsonl","matches.jsonl","application/json")]:
        p=ROOT/"storage"/fn
        if p.exists():st.download_button(f"下载 {label}",p.read_bytes(),fn,mime,use_container_width=True)
