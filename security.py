from __future__ import annotations
import os,hmac,hashlib
import streamlit as st

def _secret():
    try: return str(st.secrets.get("SAVE_PASSWORD","") or "")
    except Exception: return str(os.getenv("SAVE_PASSWORD","") or "")

def configured(): return bool(_secret())
def verify(value:str)->bool:
    s=_secret()
    return bool(s) and hmac.compare_digest(hashlib.sha256((value or "").encode()).digest(),hashlib.sha256(s.encode()).digest())

def gate(label="管理密码",key="admin_password"):
    if not configured():
        st.error("尚未配置 SAVE_PASSWORD。为避免无认证写入，所有保存/修改/删除操作已禁用。")
        return False
    value=st.text_input(label,type="password",key=key,autocomplete="off")
    return verify(value)
