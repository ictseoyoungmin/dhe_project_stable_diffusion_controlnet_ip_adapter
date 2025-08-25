import os, json, argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw

# -----------------------------
# Utils
# -----------------------------
def to_gray_rgba(path):
    img = Image.open(path).convert("RGBA")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return img, gray  # PIL RGBA, float32 HxW

def norm01(x, eps=1e-8):
    x = x.astype(np.float32)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)

def gaussian_kernel(size, sigma):
    k = cv2.getGaussianKernel(size, sigma)
    return (k @ k.T).astype(np.float32)

# -----------------------------
# Robust FG mask (white/flat BG)
# -----------------------------
def estimate_fg_mask(img_rgba, gray, flood_tol=12, min_fg_ratio=0.002):
    H, W = gray.shape
    u8 = gray.astype(np.uint8)
    corners = [(1,1), (1,W-2), (H-2,1), (H-2,W-2)]
    masks = []
    for (sy, sx) in corners:
        mask = np.zeros((H+2, W+2), np.uint8)
        _img = u8.copy()
        cv2.floodFill(_img, mask, (sx, sy), 255, loDiff=flood_tol, upDiff=flood_tol,
                      flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        masks.append(mask[1:-1,1:-1])
    bg_mask = np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, k, iterations=2)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN,  k, iterations=1)

    fg_mask = (1 - bg_mask).astype(np.uint8)
    if fg_mask.sum() < min_fg_ratio * H * W:
        _, th = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fg_mask = (th > 0).astype(np.uint8)

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, k, iterations=1)
    return fg_mask.astype(np.float32)

# -----------------------------
# Score components
# -----------------------------
def flatness_map(gray, win=25):
    g = gray / 255.0
    g2 = g * g
    k = gaussian_kernel(win, max(1, win//6))
    mu  = cv2.filter2D(g,  -1, k, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.filter2D(g2, -1, k, borderType=cv2.BORDER_REFLECT)
    var = np.clip(mu2 - mu * mu, 0, None)
    F = 1.0 - norm01(var)
    return F

def edge_energy(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    return norm01(mag)

def center_bias(shape, sigma_ratio=0.30):
    H, W = shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W/2.0, H/2.0
    d2 = (x - cx)**2 + (y - cy)**2
    sigma = (min(H, W) * sigma_ratio)**2
    C = np.exp(-d2 / (2*sigma))
    return norm01(C)

def symmetry_map(gray):
    g = gray / 255.0
    g_flipped = np.fliplr(g)
    diff = np.abs(g - g_flipped)
    SYM = 1.0 - norm01(diff)
    SYM = cv2.GaussianBlur(SYM, (7,7), 0)
    return SYM

def structure_penalty(gray):
    kx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    ky = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ax = np.abs(kx); ay = np.abs(ky)
    horiz = cv2.GaussianBlur(ay, (9,9), 0)
    vert  = cv2.GaussianBlur(ax, (9,9), 0)
    O = norm01(0.6*horiz + 0.4*vert)
    return O

def masked_edge_distance(gray, fg_mask):
    u8 = gray.astype(np.uint8)
    edges = cv2.Canny(u8, 80, 160, L2gradient=True)
    fg_boundary = (cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))) > 0)
    zeros = ((edges > 0) | fg_boundary).astype(np.uint8)
    inside = (fg_mask > 0).astype(np.uint8)
    inside[zeros > 0] = 0
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3) if inside.max() else np.zeros_like(gray, np.float32)
    D = norm01(dist) * (fg_mask > 0).astype(np.float32)
    return D, edges

# -----------------------------
# Support (plane) builder
# -----------------------------
def build_support_mask(gray, fg_mask, flat_thr=0.65, edge_thr=0.35, k_close=7, k_open=5):
    F = flatness_map(gray, win=31)
    E = edge_energy(gray)
    P = (fg_mask > 0).astype(np.uint8)
    sup = ((F >= float(flat_thr)) & (E <= float(edge_thr)) & (P > 0)).astype(np.uint8)
    if k_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        sup = cv2.morphologyEx(sup, cv2.MORPH_CLOSE, k, iterations=1)
    if k_open > 1:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        sup = cv2.morphologyEx(sup, cv2.MORPH_OPEN, k2, iterations=1)
    cnts, _ = cv2.findContours(sup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return P
    c = max(cnts, key=cv2.contourArea)
    sup2 = np.zeros_like(sup)
    cv2.drawContours(sup2, [c], -1, 1, thickness=-1)
    return sup2

# -----------------------------
# Score aggregation (support-aware)
# -----------------------------
def compute_score(gray, fg_mask, support=None):
    F = flatness_map(gray, win=31)
    D, edges = masked_edge_distance(gray, fg_mask)
    E = edge_energy(gray)
    C = center_bias(gray.shape, sigma_ratio=0.30)
    SYM = symmetry_map(gray)
    O = structure_penalty(gray)
    bg = 1.0 - fg_mask
    # weights
    w_f, w_d, w_c, w_sym, w_e, w_o, w_bg, w_sup = 0.18, 0.42, 0.14, 0.06, 0.05, 0.03, 0.35, 0.20
    S = (w_f*(F*fg_mask) + w_d*D + w_c*(C*fg_mask) + w_sym*(SYM*fg_mask)
         - w_e*(E*fg_mask) - w_o*(O*fg_mask) - w_bg*bg)
    if support is not None:
        S = S + w_sup * support.astype(np.float32)
        S = S * (0.8*support + 0.2*fg_mask)
    S = cv2.GaussianBlur(S, (9,9), 0)
    return norm01(S), dict(F=F, D=D, E=E, C=C, SYM=SYM, O=O, edges=edges, FG=fg_mask, SUP=support)

# -----------------------------
# BBox search
# -----------------------------
def window_score(S, fg_mask, cx, cy, w, h, fg_min=0.35, fg_gamma=2.0):
    H, W = S.shape
    x0 = int(np.clip(cx - w//2, 0, W-1))
    y0 = int(np.clip(cy - h//2, 0, H-1))
    x1 = int(np.clip(x0 + w, 0, W))
    y1 = int(np.clip(y0 + h, 0, H))
    if x1 <= x0 or y1 <= y0:
        return -1.0
    sc = float(np.mean(S[y0:y1, x0:x1]))
    fg_ratio = float(np.mean(fg_mask[y0:y1, x0:x1] > 0.5))
    if fg_ratio < 1e-6:
        return -1.0
    weight = max(fg_ratio / max(fg_min, 1e-6), 0.0) ** fg_gamma
    return sc * weight

def pick_bbox(gray, S, fg_mask,
              ratio_list=(1.2, 2.0, 3.0, 4.0),
              k_sizes=24, min_scale=0.10, max_scale=0.60):
    H, W = S.shape
    cy, cx = np.unravel_index(int(np.argmax(S)), S.shape)
    short = min(H, W)
    ws = np.linspace(int(short*min_scale), int(short*max_scale), k_sizes).astype(int)
    best = (-1, (cx, cy, ws[0], int(ws[0]*ratio_list[0])))
    for r in ratio_list:
        for w in ws:
            h = int(max(1, round(w * r)))
            sc = window_score(S, fg_mask, cx, cy, w, h, fg_min=0.35, fg_gamma=2.0)
            if sc > best[0]:
                best = (sc, (cx, cy, w, h))
    cx, cy, w, h = best[1]
    x = int(np.clip(cx - w//2, 0, W-1))
    y = int(np.clip(cy - h//2, 0, H-1))
    w = int(np.clip(w, 1, W - x))
    h = int(np.clip(h, 1, H - y))
    return (x, y, w, h)

def refine_bbox_by_score(S, fg_mask, bbox, iters=80, step=3, shrink=0.95,
                         min_fg=0.65, prefer_center=True):
    H, W = S.shape
    x, y, w, h = map(int, bbox)
    best = (-1.0, (x, y, w, h))
    C = center_bias(S.shape, sigma_ratio=0.35) if prefer_center else None
    def score_box(b):
        x0, y0, bw, bh = map(int, b)
        if bw < 8 or bh < 8: return -1e9
        x1, y1 = x0 + bw, y0 + bh
        subS = S[y0:y1, x0:x1]; subF = fg_mask[y0:y1, x0:x1]
        fg_ratio = float(np.mean(subF > 0.5))
        if fg_ratio < min_fg: return -1e9
        sc = float(np.mean(subS))
        if C is not None: sc = sc*0.8 + float(np.mean(C[y0:y1, x0:x1]))*0.2
        return sc
    cur_sc = score_box((x, y, w, h))
    if cur_sc > best[0]: best = (cur_sc, (x, y, w, h))
    for _ in range(iters):
        x0, y0, bw, bh = best[1]
        cand = [
            (x0-step, y0, bw, bh), (x0+step, y0, bw, bh),
            (x0, y0-step, bw, bh), (x0, y0+step, bw, bh),
            (x0, y0, max(8, int(bw*shrink)), max(8, int(bh*shrink))),
        ]
        improved = False
        for b in cand:
            bx, by, bw, bh = b
            bx = max(0, min(bx, W-8)); by = max(0, min(by, H-8))
            bw = max(8, min(bw, W - bx)); bh = max(8, min(bh, H - by))
            sc = score_box((bx, by, bw, bh))
            if sc > best[0]:
                best = (sc, (bx, by, bw, bh)); improved = True
        if not improved: break
    return best[1]

# -----------------------------
# Landmarks (lite20 / ext32 / std68 / ibug68)
# -----------------------------
def canonical_landmarks_lite20(size=256):
    s = size
    browL  = [[int(0.28*s), int(0.30*s)],
              [int(0.35*s), int(0.26*s)],
              [int(0.42*s), int(0.25*s)],
              [int(0.49*s), int(0.26*s)]]
    browR  = [[int(0.51*s), int(0.26*s)],
              [int(0.58*s), int(0.25*s)],
              [int(0.65*s), int(0.26*s)],
              [int(0.72*s), int(0.30*s)]]
    eyeL   = [[int(0.34*s), int(0.36*s)],
              [int(0.40*s), int(0.34*s)],
              [int(0.46*s), int(0.36*s)],
              [int(0.40*s), int(0.38*s)]]
    eyeR   = [[int(0.54*s), int(0.36*s)],
              [int(0.60*s), int(0.34*s)],
              [int(0.66*s), int(0.36*s)],
              [int(0.60*s), int(0.38*s)]]
    nose   = [[int(0.50*s), int(0.46*s)],
              [int(0.50*s), int(0.58*s)]]
    mouth  = [[int(0.42*s), int(0.68*s)],
              [int(0.58*s), int(0.68*s)]]
    pts = np.array(browL + browR + eyeL + eyeR + nose + mouth, dtype=np.float32)
    return pts

def canonical_landmarks_ext32(size=256):
    s = size
    browL  = [[int(0.28*s), int(0.30*s)],
              [int(0.35*s), int(0.26*s)],
              [int(0.42*s), int(0.25*s)],
              [int(0.49*s), int(0.26*s)]]
    browR  = [[int(0.51*s), int(0.26*s)],
              [int(0.58*s), int(0.25*s)],
              [int(0.65*s), int(0.26*s)],
              [int(0.72*s), int(0.30*s)]]
    eyeL = [[int(0.31*s), int(0.36*s)],
            [int(0.36*s), int(0.33*s)],
            [int(0.42*s), int(0.36*s)],
            [int(0.36*s), int(0.39*s)],
            [int(0.36*s), int(0.36*s)]]
    eyeR = [[int(0.58*s), int(0.36*s)],
            [int(0.64*s), int(0.33*s)],
            [int(0.69*s), int(0.36*s)],
            [int(0.64*s), int(0.39*s)],
            [int(0.64*s), int(0.36*s)]]
    nose = [[int(0.50*s), int(0.44*s)],
            [int(0.46*s), int(0.57*s)],
            [int(0.54*s), int(0.57*s)],
            [int(0.50*s), int(0.60*s)]]
    mouth = [
        [int(0.40*s), int(0.68*s)],
        [int(0.46*s), int(0.66*s)],
        [int(0.50*s), int(0.65*s)],
        [int(0.54*s), int(0.66*s)],
        [int(0.60*s), int(0.68*s)],
        [int(0.54*s), int(0.70*s)],
        [int(0.50*s), int(0.71*s)],
        [int(0.46*s), int(0.70*s)],
        [int(0.50*s), int(0.67*s)],
        [int(0.50*s), int(0.69*s)]
    ]
    pts = np.array(browL + browR + eyeL + eyeR + nose + mouth, dtype=np.float32)
    assert pts.shape[0] == 32
    return pts

def canonical_landmarks_std68(size=256):
    s = float(size)
    cx, cy = 0.50*s, 0.55*s
    face_rx, face_ry = 0.32*s, 0.36*s
    brow_y = 0.30*s
    eye_y  = 0.36*s
    nose_top_y, nose_bot_y = 0.42*s, 0.60*s
    mouth_cy = 0.70*s
    mouth_rx, mouth_ry = 0.14*s, 0.07*s
    inner_rx, inner_ry = 0.09*s, 0.045*s
    eye_rx, eye_ry = 0.055*s, 0.030*s
    eye_off = 0.105*s

    def jaw_arc():
        ang = np.linspace(20, 160, 17)
        pts = []
        for a in ang:
            r = np.deg2rad(a)
            x = cx + face_rx*np.cos(r)
            y = cy + face_ry*np.sin(r)
            pts.append([int(round(x)), int(round(y))])
        return pts

    def brows():
        lbx = np.linspace(cx-0.18*s, cx-0.02*s, 5)
        rbx = np.linspace(cx+0.02*s, cx+0.18*s, 5)
        def row(xs):
            return [[int(round(x)), int(round(brow_y - 0.03*s*np.cos(np.pi*(i/4))))] for i, x in enumerate(xs)]
        return row(lbx) + row(rbx)

    def nose():
        bridge_y = np.linspace(nose_top_y, (nose_top_y+nose_bot_y)*0.55, 4)
        bridge = [[int(round(cx)), int(round(y))] for y in bridge_y]
        xs = np.linspace(cx-0.07*s, cx+0.07*s, 5)
        low = []
        for x in xs:
            t = (x - cx) / (0.07*s + 1e-6)
            y = nose_bot_y - 0.01*s + 0.05*s*(1 - t*t)**0.5
            low.append([int(round(x)), int(round(y))])
        return bridge + low

    def eye_left():
        cxL = cx - eye_off
        xs = [cxL - eye_rx, cxL - 0.35*eye_rx, cxL + 0.35*eye_rx, cxL + eye_rx]
        return [
            [int(round(xs[0])), int(round(eye_y))],
            [int(round(xs[1])), int(round(eye_y - 0.8*eye_ry))],
            [int(round(xs[2])), int(round(eye_y - 0.8*eye_ry))],
            [int(round(xs[3])), int(round(eye_y))],
            [int(round(xs[2])), int(round(eye_y + 0.8*eye_ry))],
            [int(round(xs[1])), int(round(eye_y + 0.8*eye_ry))],
        ]

    def eye_right():
        cxR = cx + eye_off
        xs = [cxR - eye_rx, cxR - 0.35*eye_rx, cxR + 0.35*eye_rx, cxR + eye_rx]
        return [
            [int(round(xs[0])), int(round(eye_y))],
            [int(round(xs[1])), int(round(eye_y - 0.8*eye_ry))],
            [int(round(xs[2])), int(round(eye_y - 0.8*eye_ry))],
            [int(round(xs[3])), int(round(eye_y))],
            [int(round(xs[2])), int(round(eye_y + 0.8*eye_ry))],
            [int(round(xs[1])), int(round(eye_y + 0.8*eye_ry))],
        ]

    def mouth():
        L = cx - mouth_rx; R = cx + mouth_rx
        def y_upper(x):
            t = (x - cx) / (mouth_rx + 1e-6)
            return mouth_cy - mouth_ry * np.sqrt(max(0.0, 1 - t*t))
        def y_lower(x):
            t = (x - cx) / (mouth_rx + 1e-6)
            return mouth_cy + mouth_ry * np.sqrt(max(0.0, 1 - t*t))
        xs_up = [L + (R-L)/6, L + 2*(R-L)/6, cx, R - 2*(R-L)/6, R - (R-L)/6]
        outer = []
        outer.append([int(round(L)), int(round(mouth_cy))])
        for x in xs_up: outer.append([int(round(x)), int(round(y_upper(x)))])
        outer.append([int(round(R)), int(round(mouth_cy))])
        xs_dn = [R - (R-L)/6, R - 2*(R-L)/6, cx, L + 2*(R-L)/6, L + (R-L)/6]
        for x in xs_dn: outer.append([int(round(x)), int(round(y_lower(x)))])
        L2 = cx - inner_rx; R2 = cx + inner_rx
        inner = []
        inner.append([int(round(L2)), int(round(mouth_cy))])
        for x in [L2 + (R2-L2)/4, cx, R2 - (R2-L2)/4]:
            t = (x - cx) / (inner_rx + 1e-6)
            y = mouth_cy - inner_ry * np.sqrt(max(0.0, 1 - t*t))
            inner.append([int(round(x)), int(round(y))])
        inner.append([int(round(R2)), int(round(mouth_cy))])
        for x in [R2 - (R2-L2)/4, cx, L2 + (R2-L2)/4]:
            t = (x - cx) / (inner_rx + 1e-6)
            y = mouth_cy + inner_ry * np.sqrt(max(0.0, 1 - t*t))
            inner.append([int(round(x)), int(round(y))])
        return outer + inner

    pts = np.array(jaw_arc() + brows() + nose() + eye_left() + eye_right() + mouth(), dtype=np.float32)
    return pts

def canonical_landmarks_ibug68(size=256):
    # 여기서는 std68 형태를 그대로 사용 (필요시 별도 조정 가능)
    return canonical_landmarks_std68(size)

def get_canonical_landmarks(landmarks_type="ext32", size=256):
    lt = landmarks_type.lower()
    if lt == "lite20":
        return canonical_landmarks_lite20(size), "lite20"
    elif lt == "ext32":
        return canonical_landmarks_ext32(size), "ext32"
    elif lt in ["std68", "dlib68"]:
        return canonical_landmarks_std68(size), "std68"
    elif lt in ["ibug68", "ibug", "68ibug"]:
        return canonical_landmarks_ibug68(size), "ibug68"
    else:
        raise ValueError(f"Unsupported landmarks_type: {landmarks_type}")

# -----------------------------
# Expression library (요약: 기존 로직 반영)
# -----------------------------
def apply_expression(canon_pts, lm_type, emotion="normal", intensity=0.6, size=256):
    if emotion == "normal" or intensity <= 1e-6:
        return canon_pts.copy()
    pts = canon_pts.copy()
    s = float(size); k = float(np.clip(intensity, 0.0, 1.0))

    if lm_type == "lite20":
        browL = slice(0,4); browR = slice(4,8)
        eyeL  = slice(8,12); eyeR  = slice(12,16)
        mouth = slice(18,20)
        if emotion == "happy":
            pts[mouth, 0] += np.array([-1, +1]) * (0.015*s)*k
            pts[mouth, 1] += np.array([-1, -1]) * (0.020*s)*k
            pts[eyeL, 1]  += (+0.006*s)*k; pts[eyeR, 1]  += (+0.006*s)*k
            pts[browL, 1] -= (0.006*s)*k;  pts[browR, 1] -= (0.006*s)*k
        elif emotion == "sad":
            pts[mouth, 0] += np.array([+1, -1]) * (0.010*s)*k
            pts[mouth, 1] += (+0.020*s)*k
            pts[browL, 1] += (0.004*s)*k; pts[browL.start, 1] -= (0.010*s)*k
            pts[browR, 1] += (0.004*s)*k; pts[browR.stop-1, 1] -= (0.010*s)*k
        elif emotion == "angry":
            pts[browL, 1] += (0.010*s)*k; pts[browL, 0] += (+0.006*s)*k
            pts[browR, 1] += (0.010*s)*k; pts[browR, 0] += (-0.006*s)*k
            pts[eyeL, 1]  += (+0.008*s)*k; pts[eyeR, 1]  += (+0.008*s)*k
            pts[mouth, 1] += (+0.010*s)*k
        elif emotion == "surprised":
            pts[mouth, 1] += (+0.025*s)*k
            pts[browL, 1] -= (0.012*s)*k; pts[browR, 1] -= (0.012*s)*k
            pts[eyeL, 1]  -= (0.006*s)*k; pts[eyeR, 1]  -= (0.006*s)*k

    elif lm_type == "ext32":
        browL = slice(0,4); browR = slice(4,8)
        eyeL  = slice(8,13); eyeR = slice(13,18)
        mouth = slice(22,32)
        m_left, m_right = 22, 26
        m_upL, m_upM, m_upR = 23, 24, 25
        m_dnR, m_dnM, m_dnL = 27, 28, 29
        m_in_up, m_in_dn = 30, 31
        if emotion == "happy":
            pts[[m_left, m_right], 0] += np.array([-1, +1]) * (0.015*s)*k
            pts[[m_left, m_right], 1] += (-0.022*s)*k
            pts[[m_upL, m_upM, m_upR], 1] -= (0.010*s)*k
            pts[[m_dnL, m_dnM, m_dnR], 1] -= (0.004*s)*k
            pts[[m_in_up], 1] -= (0.006*s)*k; pts[[m_in_dn], 1] -= (0.002*s)*k
            pts[eyeL, 1] += (+0.004*s)*k; pts[eyeR, 1] += (+0.004*s)*k
            pts[browL,1] -= (0.006*s)*k;  pts[browR,1] -= (0.006*s)*k
        elif emotion == "sad":
            pts[[m_left, m_right], 0] += np.array([+1, -1]) * (0.010*s)*k
            pts[[m_left, m_right], 1] += (+0.020*s)*k
            pts[[m_upL, m_upM, m_upR], 1] += (+0.006*s)*k
            pts[[m_dnL, m_dnM, m_dnR], 1] += (+0.012*s)*k
            pts[browL,1] += (+0.004*s)*k; pts[browL.start,1] -= (0.010*s)*k
            pts[browR,1] += (+0.004*s)*k; pts[browR.stop-1,1] -= (0.010*s)*k
        elif emotion == "angry":
            pts[browL,1] += (+0.012*s)*k; pts[browL,0] += (+0.006*s)*k
            pts[browR,1] += (+0.012*s)*k; pts[browR,0] += (-0.006*s)*k
            pts[eyeL,1]  += (+0.010*s)*k; pts[eyeR,1]  += (+0.010*s)*k
            pts[[m_upM, m_dnM], 1] += (+0.006*s)*k
        elif emotion == "surprised":
            pts[[m_upM], 1] -= (0.018*s)*k
            pts[[m_dnM], 1] += (0.022*s)*k
            pts[[m_in_up], 1] -= (0.015*s)*k; pts[[m_in_dn], 1] += (0.018*s)*k
            pts[[m_left, m_right], 1] += (+0.008*s)*k
            pts[browL,1] -= (0.014*s)*k; pts[browR,1] -= (0.014*s)*k
            pts[eyeL,1]  -= (0.006*s)*k; pts[eyeR,1]  -= (0.006*s)*k

    elif lm_type in ["std68", "ibug68"]:
        browL   = slice(17,22)   # 17..21
        browR   = slice(22,27)   # 22..26
        eyeL    = slice(36,42)   # 36..41
        eyeR    = slice(42,48)   # 42..47
        m_out   = slice(48,60)   # 48..59
        m_in    = slice(60,68)   # 60..67
        mL, mR = 48, 54
        m_up_mid, m_dn_mid = 51, 57

        if emotion == "happy":
            pts[[mL, mR], 0] += np.array([-1, +1]) * (0.016*s)*k
            pts[[mL, mR], 1] += (-0.020*s)*k
            pts[m_out, 1]    += np.linspace(-0.010*s, -0.004*s, pts[m_out].shape[0]) * k
            pts[m_in,  1]    += np.linspace(-0.006*s, -0.002*s, pts[m_in].shape[0]) * k
            pts[eyeL, 1]     += (+0.004*s)*k; pts[eyeR, 1] += (+0.004*s)*k
            pts[browL, 1]    -= (0.006*s)*k; pts[browR, 1] -= (0.006*s)*k

        elif emotion == "sad":
            pts[[mL, mR], 0] += np.array([+1, -1]) * (0.010*s)*k
            pts[[mL, mR], 1] += (+0.020*s)*k
            pts[m_out, 1]    += (+0.008*s)*k
            pts[m_in,  1]    += (+0.006*s)*k
            pts[[21,22], 1]  -= (0.010*s)*k
            pts[browL, 1]    += (+0.004*s)*k; pts[browR, 1] += (+0.004*s)*k

        elif emotion == "angry":
            pts[browL, 1]    += (+0.012*s)*k; pts[browL, 0] += (+0.006*s)*k
            pts[browR, 1]    += (+0.012*s)*k; pts[browR, 0] += (-0.006*s)*k
            pts[eyeL, 1]     += (+0.010*s)*k; pts[eyeR, 1] += (+0.010*s)*k
            pts[[m_up_mid, m_dn_mid], 1] += (+0.006*s)*k

        elif emotion == "surprised":
            pts[[m_up_mid], 1] -= (0.018*s)*k
            pts[[m_dn_mid], 1] += (0.022*s)*k
            pts[browL, 1]     -= (0.014*s)*k; pts[browR, 1] -= (0.014*s)*k
            pts[eyeL, 1]      -= (0.006*s)*k; pts[eyeR, 1]  -= (0.006*s)*k

    return pts

# -----------------------------
# Aspect-aware scaling helpers
# -----------------------------
def estimate_aspect_from_mask(fg_mask, fallback_aspect=1.2):
    m = (fg_mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return float(fallback_aspect)
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    w, h = rect[1]
    if w < 1 or h < 1:
        return float(fallback_aspect)
    a = float(max(w, h) / max(1.0, min(w, h)))
    return a

def anisotropic_factors(aspect, alpha=0.35, limit=1.35):
    sx = float(aspect ** float(alpha))
    sx = float(np.clip(sx, 1.0/limit, limit))
    sy = 1.0 / sx
    return sx, sy

def grow_bbox_anisotropic(bbox, face_scale, sx, sy, image_shape):
    H, W = image_shape
    x, y, w, h = map(float, bbox)
    cx, cy = x + w/2.0, y + h/2.0
    req_w = w * float(face_scale) * float(sx)
    req_h = h * float(face_scale) * float(sy)
    max_w = 2.0 * min(cx, (W-1) - cx)
    max_h = 2.0 * min(cy, (H-1) - cy)
    k_w = (max_w / max(1e-6, req_w))
    k_h = (max_h / max(1e-6, req_h))
    k = min(1.0, k_w, k_h)
    final_w = max(1.0, req_w * k)
    final_h = max(1.0, req_h * k)
    x2 = int(round(cx - final_w/2.0))
    y2 = int(round(cy - final_h/2.0))
    w2 = int(round(final_w)); h2 = int(round(final_h))
    x2 = max(0, min(x2, W-1))
    y2 = max(0, min(y2, H-1))
    w2 = max(1, min(w2, W - x2))
    h2 = max(1, min(h2, H - y2))
    return (x2, y2, w2, h2)

# -----------------------------
# Warp & Draw
# -----------------------------
def warp_canonical_to_bbox(canon_pts, bbox, canonical_size=256):
    x, y, w, h = bbox
    scale_x = w / float(canonical_size)
    scale_y = h / float(canonical_size)
    world = np.empty_like(canon_pts, dtype=np.float32)
    world[:, 0] = x + canon_pts[:, 0] * scale_x
    world[:, 1] = y + canon_pts[:, 1] * scale_y
    return world

def draw_landmarks_overlay(img_rgba, landmarks, radius=3, color=(0,180,255,255), bbox=None):
    out = img_rgba.copy()
    draw = ImageDraw.Draw(out)
    if bbox is not None:
        x, y, w, h = bbox
        draw.rectangle([x, y, x+w, y+h], outline=(0,255,0,255), width=3)
    for (px, py) in landmarks:
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
    return out

def clamp_points_to_mask(pts, fg_mask, margin=2):
    H, W = fg_mask.shape
    safe = (fg_mask > 0).astype(np.uint8)
    if margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin+1, 2*margin+1))
        safe = cv2.erode(safe, k, iterations=1)
    cnts, _ = cv2.findContours(safe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return pts
    cnt = max(cnts, key=cv2.contourArea).reshape(-1,2).astype(np.float32)
    M = cv2.moments(safe, binaryImage=True)
    if M["m00"] > 1e-6:
        cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
    else:
        cx, cy = W/2.0, H/2.0
    center = np.array([cx, cy], dtype=np.float32)
    out = pts.copy().astype(np.float32)
    for i, p in enumerate(out):
        xi, yi = int(round(p[0])), int(round(p[1]))
        xi = np.clip(xi, 0, W-1); yi = np.clip(yi, 0, H-1)
        if safe[yi, xi] == 1:
            continue
        d2 = np.sum((cnt - p[None,:])**2, axis=1)
        j = int(np.argmin(d2))
        closest = cnt[j]
        v = center - closest
        nv = v / (np.linalg.norm(v) + 1e-6)
        out[i] = np.clip(closest + nv * max(1.0, float(margin)), [0,0], [W-1,H-1]).astype(np.float32)
    return out

# -----------------------------
# Plane indices & scale from plane
# -----------------------------
def plane_indices(lm_type):
    lt = lm_type.lower()
    if lt == "lite20":
        return list(range(8, 16)) + [16, 17] + [18, 19]
    elif lt == "ext32":
        return list(range(8, 13)) + list(range(13, 18)) + list(range(18, 22)) + list(range(22, 32))
    elif lt in ["std68","ibug68"]:
        return list(range(36, 48)) + list(range(27, 36)) + list(range(48, 68))
    else:
        return None

def aabb_of_points(pts):
    x0, y0 = float(np.min(pts[:,0])), float(np.min(pts[:,1]))
    x1, y1 = float(np.max(pts[:,0])), float(np.max(pts[:,1]))
    return x0, y0, x1, y1, (x1-x0), (y1-y0)

def extent_from_center(mask, cx, cy):
    H, W = mask.shape
    cx = int(np.clip(round(cx), 0, W-1))
    cy = int(np.clip(round(cy), 0, H-1))
    if mask[cy, cx] == 0:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0, 0
        d2 = (xs - cx)**2 + (ys - cy)**2
        j = int(np.argmin(d2))
        cx, cy = int(xs[j]), int(ys[j])
    L = R = U = D = 0
    for x in range(cx, -1, -1):
        if mask[cy, x] == 0: break
        L += 1
    for x in range(cx, W):
        if mask[cy, x] == 0: break
        R += 1
    for y in range(cy, -1, -1):
        if mask[y, cx] == 0: break
        U += 1
    for y in range(cy, H):
        if mask[y, cx] == 0: break
        D += 1
    avail_w = max(0, L + R - 1)
    avail_h = max(0, U + D - 1)
    return avail_w, avail_h

def compute_face_scale_from_plane(canon_pts, lm_type, bbox, support_mask, canonical_size=256, fill=0.85):
    x, y, w, h = map(float, bbox)
    cx, cy = x + w/2.0, y + h/2.0
    avail_w, avail_h = extent_from_center(support_mask, cx, cy)
    if avail_w < 2 or avail_h < 2:
        return 1.0
    idx = plane_indices(lm_type)
    if not idx:
        return 1.0
    plane = canon_pts[idx]
    _, _, _, _, pw, ph = aabb_of_points(plane)
    rw = max(1e-6, pw / float(canonical_size))
    rh = max(1e-6, ph / float(canonical_size))
    scale_w = (avail_w * float(fill)) / max(1e-6, w * rw)
    scale_h = (avail_h * float(fill)) / max(1e-6, h * rh)
    face_scale = float(np.clip(min(scale_w, scale_h), 0.5, 5.0))
    return face_scale

# -----------------------------
# Auto params
# -----------------------------
def compute_auto_params(fg_mask, bbox, image_shape,
                        target_cover=(0.70, 0.85),
                        alpha_range=(0.30, 0.55),
                        limit_range=(1.20, 1.40)):
    H, W = image_shape
    x,y,w,h = map(float, bbox)
    m = (fg_mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.4, 0.40, 1.30
    c = max(cnts, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    if rw < 1 or rh < 1:
        return 1.4, 0.40, 1.30
    fg_ratio = float(np.mean(m[int(y):int(y+h), int(x):int(x+w)])) if (w>1 and h>1) else 0.5
    rw *= max(0.3, min(1.0, fg_ratio + 0.25))
    rh *= max(0.3, min(1.0, fg_ratio + 0.25))
    s_w = (target_cover[0] * w) / max(1e-6, rw)
    s_h = (target_cover[1] * h) / max(1e-6, rh)
    face_scale = float(np.clip(min(s_w, s_h), 0.8, 3.0))
    aspect = max(rw, rh) / max(1.0, min(rw, rh))
    t = float(np.clip((aspect-1.0)/2.0, 0.0, 1.0))
    stretch_alpha = alpha_range[0] * (1-t) + alpha_range[1] * t
    stretch_limit = limit_range[0] * (1-t) + limit_range[1] * t
    return face_scale, stretch_alpha, stretch_limit

# -----------------------------
# Main build
# -----------------------------
def build_normal_rig_auto(base_image_path, save_dir="out", canonical_size=256,
                          landmarks_type="ext32", emotion="normal", intensity=0.6,
                          face_scale=1.0, stretch_alpha=0.35, stretch_limit=1.35,
                          clamp_to_mask=True, mode=None):
    os.makedirs(save_dir, exist_ok=True)

    img_rgba, gray = to_gray_rgba(base_image_path)

    # 1) robust FG + 2) support 먼저
    fg_mask = estimate_fg_mask(img_rgba, gray, flood_tol=12)
    support = build_support_mask(gray, fg_mask, flat_thr=0.70, edge_thr=0.30)

    # 3) support-aware score
    S, maps = compute_score(gray, fg_mask, support=support)
    # support를 강하게 선호하도록 보정
    S_eff = S * (0.85*support + 0.15*fg_mask) + 0.05*center_bias(gray.shape, 0.28)

    # 4) bbox: pick -> refine (support 기준으로 평가)
    bbox0 = pick_bbox(gray, S_eff, support)
    bbox = refine_bbox_by_score(S_eff, support, bbox0,
                                iters=80, step=3, shrink=0.95, min_fg=0.65, prefer_center=True)

    # 5) auto params (support 기반 추천)
    if mode == "auto" or mode is True:
        face_scale_auto, stretch_alpha, stretch_limit = compute_auto_params(
            support, bbox, image_shape=gray.shape
        )
    else:
        face_scale_auto = face_scale

    # 6) landmarks & expression
    canon, lm_type = get_canonical_landmarks(landmarks_type, canonical_size)
    canon_emotion = apply_expression(canon, lm_type, emotion=emotion, intensity=intensity, size=canonical_size)

    # 7) support 채움 비율로 face_scale 역산 → 최종
    face_scale_plane = compute_face_scale_from_plane(
        canon_pts=canon, lm_type=lm_type, bbox=bbox, support_mask=support,
        canonical_size=canonical_size, fill=0.85
    )
    face_scale_final = float(np.clip(face_scale_auto * face_scale_plane, 1., 3.0))

    # 8) 비등방 stretch
    aspect = estimate_aspect_from_mask(support, fallback_aspect=1.2)
    sx, sy = anisotropic_factors(aspect, alpha=stretch_alpha, limit=stretch_limit)

    # 9) bbox grow + warp + clamp
    H, W = gray.shape
    bbox_scaled = grow_bbox_anisotropic(bbox, face_scale_final, sx, sy, image_shape=(H, W))
    world_neutral = warp_canonical_to_bbox(canon, bbox_scaled, canonical_size)
    world_emotion = warp_canonical_to_bbox(canon_emotion, bbox_scaled, canonical_size)

    if clamp_to_mask:
        world_neutral = clamp_points_to_mask(world_neutral, support, margin=2)
        world_emotion = clamp_points_to_mask(world_emotion, support, margin=2)

    # 10) save rig + 디버그
    rig = {
        "bbox": [int(bbox_scaled[0]), int(bbox_scaled[1]), int(bbox_scaled[2]), int(bbox_scaled[3])],
        "image_size": [int(H), int(W)],
        "landmarks_type": lm_type,
        "canonical_size": canonical_size,
        "num_landmarks": int(canon.shape[0]),
        "emotion": emotion,
        "intensity": float(np.clip(intensity,0,1)),
        "face_scale": float(face_scale_final),
        "aspect": float(aspect),
        "stretch_alpha": float(stretch_alpha),
        "stretch_limit": float(stretch_limit),
        "sx": float(sx), "sy": float(sy),
        "landmarks_world_neutral": world_neutral.astype(float).tolist(),
        "landmarks_world_emotion": world_emotion.astype(float).tolist()
    }
    with open(os.path.join(save_dir, "rig.json"), "w", encoding="utf-8") as f:
        json.dump(rig, f, ensure_ascii=False, indent=2)

    heat = (cv2.applyColorMap((S*255).astype(np.uint8), cv2.COLORMAP_JET))
    heat = Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay_neutral = draw_landmarks_overlay(img_rgba, world_neutral, bbox=bbox_scaled)
    overlay_emotion = draw_landmarks_overlay(img_rgba, world_emotion, color=(255,0,0,255), bbox=bbox_scaled)

    overlay_neutral.save(os.path.join(save_dir, f"overlay_neutral_{lm_type}.png"))
    overlay_emotion.save(os.path.join(save_dir, f"overlay_{emotion}_{lm_type}.png"))
    heat.save(os.path.join(save_dir, "score_heatmap.png"))
    Image.fromarray(cv2.cvtColor(maps["edges"], cv2.COLOR_GRAY2RGB)).save(os.path.join(save_dir, "edges.png"))
    Image.fromarray((fg_mask*255).astype(np.uint8)).save(os.path.join(save_dir, "fg_mask.png"))
    if support is not None:
        Image.fromarray((support*255).astype(np.uint8)).save(os.path.join(save_dir, "support_mask.png"))

    print(f"[OK] Saved in: {save_dir}")
    return rig, overlay_neutral, overlay_emotion, heat

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="/workspace/dhe_project/assets/obj/tissue/object_00.png") #/workspace/dhe_project/assets/sofa_01.png
    parser.add_argument("--out", type=str, default="out_tissue")
    parser.add_argument("--landmarks", type=str, default="ext32", choices=["lite20","ext32","std68","ibug68"])
    parser.add_argument("--canon_size", type=int, default=256)
    parser.add_argument("--emotion", type=str, default="normal",
                        choices=["normal","happy","sad","angry","surprised"])
    parser.add_argument("--intensity", type=float, default=0.6)
    parser.add_argument("--face_scale", type=float, default=1.1)
    parser.add_argument("--stretch_alpha", type=float, default=0.45)
    parser.add_argument("--stretch_limit", type=float, default=1.3)
    parser.add_argument("--auto", action='store_true', help="전역 자동 스케일/스트레치 활성화")
    parser.add_argument("--no_mask_clamp", action="store_true", help="평면 밖으로 튀는 점 보정 끄기")
    args = parser.parse_args()

    build_normal_rig_auto(
        base_image_path=args.image,
        save_dir=args.out,
        canonical_size=args.canon_size,
        landmarks_type=args.landmarks,
        emotion=args.emotion,
        intensity=args.intensity,
        face_scale=args.face_scale,
        stretch_alpha=args.stretch_alpha,
        stretch_limit=args.stretch_limit,
        clamp_to_mask=(not args.no_mask_clamp),
        mode=("auto" if 1 else None),
    )
