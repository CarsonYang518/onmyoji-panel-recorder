from __future__ import annotations
import cv2
import numpy as np

# Coordinates measured from the supplied 1026x542 Duel Guessing panel screenshots.
BASE_W, BASE_H = 1026, 542
COLS = [(149,318),(318,489),(489,660),(660,830),(830,1007)]
NAME_Y = (88,129)
ROWS = {
    "atk": (129,169), "hp": (169,209), "def": (209,249), "spd": (249,289),
    "cri": (289,329), "crid": (329,369), "efh": (369,409), "efr": (409,449),
}
SOUL_Y = (449,516)


def normalize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (BASE_W, BASE_H), interpolation=cv2.INTER_AREA)


def crop(img: np.ndarray, box):
    x1,y1,x2,y2 = box
    return img[y1:y2, x1:x2]


def unit_rois(img: np.ndarray, slot: int):
    img = normalize(img)
    x1,x2 = COLS[slot-1]
    pad = 8
    out = {"name": crop(img,(x1+pad,NAME_Y[0],x2-pad,NAME_Y[1]))}
    for key,(y1,y2) in ROWS.items():
        out[key] = crop(img,(x1+20,y1,x2-20,y2))
    out["soul_icon"] = crop(img,(x1+38,SOUL_Y[0],x2-38,SOUL_Y[1]))
    return out
