from __future__ import annotations
import cv2
import numpy as np

# Canonical coordinate system is the OUTER beige lineup table, not the screenshot.
# It was measured from the original 1026x542 reference screenshot after locating
# the panel at approximately x=9..1009, y=28..519.
BASE_W, BASE_H = 1000, 491
COLS = [(140,309),(309,480),(480,651),(651,821),(821,998)]
NAME_Y = (60,101)
ROWS = {
    "atk": (101,141), "hp": (141,181), "def": (181,221), "spd": (221,261),
    "cri": (261,301), "crid": (301,341), "efh": (341,381), "efr": (381,421),
}
SOUL_Y = (421,488)


def _cluster(vals, tol):
    vals=sorted(float(v) for v in vals)
    groups=[]
    for v in vals:
        if not groups or v-groups[-1][-1] > tol: groups.append([v])
        else: groups[-1].append(v)
    return [float(np.median(g)) for g in groups]


def _best_six_equal(xs):
    """Find six near-equally-spaced vertical grid lines = 5 shikigami columns."""
    if len(xs) < 6: return None
    best=None
    for i in range(len(xs)-5):
        seq=np.asarray(xs[i:i+6],dtype=float)
        d=np.diff(seq); med=float(np.median(d))
        if med <= 0: continue
        cv=float(np.std(d)/med)
        # Five data columns are visually very regular. Small penalty favors wider tables.
        score=cv + 0.04/(med+1e-6)
        if best is None or score < best[0]: best=(score,seq,med,cv)
    if best and best[3] < 0.18: return best
    return None


def locate_panel(img: np.ndarray):
    """Locate the central 阵容详情 table by its grid geometry.

    Returns dict with bbox=(x1,y1,x2,y2), confidence, method, diagnostics.
    It does not depend on screenshot resolution/aspect ratio.
    """
    h,w=img.shape[:2]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    edges=cv2.Canny(gray,45,140)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(70,int(min(w,h)*0.13)),
                          minLineLength=max(100,int(min(w,h)*0.34)),maxLineGap=max(10,int(min(w,h)*0.025)))
    vertical=[]; horizontal=[]
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            dx=abs(int(x2)-int(x1)); dy=abs(int(y2)-int(y1))
            if dy > 0.42*h and dx < max(9,0.012*w): vertical.append((x1+x2)/2)
            if dx > 0.48*w and dy < max(9,0.012*h): horizontal.append((y1+y2)/2)
    xs=_cluster(vertical,max(5,w*0.006))
    ys=_cluster(horizontal,max(5,h*0.009))
    best=_best_six_equal(xs)
    if not best or len(ys)<2:
        return {"ok":False,"confidence":0.0,"method":"grid_hough","bbox":None,
                "diagnostics":{"vertical_clusters":xs,"horizontal_clusters":ys}}
    _,seq,colw,cv=best
    # seq[0] is the left edge of the first shikigami data column. The label column
    # is ~0.82 of a data column in the reference UI.
    px1=float(seq[0]-0.82*colw); px2=float(seq[-1])
    # Outer panel horizontal borders are the extreme long table lines around the grid.
    # Limit to plausible lines around the vertical table span to reject title decorations.
    y1=float(min(ys)); y2=float(max(ys))
    # Small geometry padding makes OCR tolerant to Hough choosing an inner border pixel.
    pad_x=max(2,int(0.004*w)); pad_y=max(2,int(0.006*h))
    x1=max(0,int(round(px1-pad_x))); x2=min(w,int(round(px2+pad_x)))
    yy1=max(0,int(round(y1-pad_y))); yy2=min(h,int(round(y2+pad_y)))
    bw=x2-x1; bh=yy2-yy1
    ar=bw/max(bh,1); area=(bw*bh)/(w*h)
    # Expected panel aspect is about 2.04; allow broad tolerance for borders/crops.
    geom=max(0.0,1.0-abs(ar-2.04)/0.85)
    regular=max(0.0,1.0-cv/0.18)
    coverage=min(1.0,max(0.0,(area-0.25)/0.35))
    conf=float(np.clip(0.45*regular+0.35*geom+0.20*coverage,0,1))
    ok=bool(1.45 < ar < 2.9 and area > 0.22 and conf >= 0.48)
    return {"ok":ok,"confidence":conf,"method":"grid_hough","bbox":(x1,yy1,x2,yy2) if ok else None,
            "diagnostics":{"vertical_clusters":xs,"horizontal_clusters":ys,
                           "data_grid_x":[round(float(x),1) for x in seq],"column_width":round(colw,1),
                           "aspect_ratio":round(ar,3),"area_ratio":round(area,3),"regularity_cv":round(cv,4)}}


def normalize_panel(img: np.ndarray, loc=None):
    loc=loc or locate_panel(img)
    if not loc.get("ok") or not loc.get("bbox"):
        raise ValueError("无法可靠定位阵容详情主面板")
    x1,y1,x2,y2=loc["bbox"]
    crop=img[y1:y2,x1:x2]
    return cv2.resize(crop,(BASE_W,BASE_H),interpolation=cv2.INTER_AREA),loc


def draw_panel_preview(img: np.ndarray, loc):
    out=img.copy()
    if loc.get("ok") and loc.get("bbox"):
        x1,y1,x2,y2=loc["bbox"]
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),max(2,int(min(img.shape[:2])*0.004)))
    return out


def crop(img: np.ndarray, box):
    x1,y1,x2,y2=box
    return img[y1:y2,x1:x2]


def unit_rois(panel_img: np.ndarray, slot: int):
    # panel_img is already canonicalized by normalize_panel().
    x1,x2=COLS[slot-1]; pad=8
    out={"name":crop(panel_img,(x1+pad,NAME_Y[0],x2-pad,NAME_Y[1]))}
    for key,(y1,y2) in ROWS.items(): out[key]=crop(panel_img,(x1+20,y1,x2-20,y2))
    out["soul_icon"]=crop(panel_img,(x1+38,SOUL_Y[0],x2-38,SOUL_Y[1]))
    return out
