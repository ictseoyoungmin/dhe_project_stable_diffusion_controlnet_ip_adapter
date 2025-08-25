import os, json, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, Image
import cv2
from scipy.spatial import Delaunay

import torch
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
)

# ---------------------------
# I/O & Utils
# ---------------------------
def load_rgba(path):
    img = Image.open(path).convert("RGBA")
    return img

def rgba_to_rgb(img_rgba, bg=(255, 255, 255)):
    bg_img = Image.new("RGB", img_rgba.size, bg)
    bg_img.paste(img_rgba, mask=img_rgba.split()[-1])
    return bg_img

def save_img(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def read_rig(path):
    with open(path, "r", encoding="utf-8") as f:
        rig = json.load(f)
    pts = np.array(
        rig["landmarks_world_emotion"]
        if "landmarks_world_emotion" in rig and len(rig["landmarks_world_emotion"]) > 0
        else rig["landmarks_world_neutral"],
        dtype=np.float32,
    )
    bbox = tuple(rig["bbox"])  # (x,y,w,h)
    lm_type = rig.get("landmarks_type", "lite20")
    return rig, bbox, pts, lm_type

# ---------------------------
# Mask utilities
# ---------------------------
def make_face_mask(img_size, bbox, shape="rounded", feather=6, round_radius_ratio=0.22):
    W, H = img_size
    x, y, w, h = map(int, bbox)
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    if shape == "ellipse":
        draw.ellipse([x, y, x + w, y + h], fill=255)
    else:
        r = int(max(2, min(w, h) * round_radius_ratio))
        draw.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=int(feather)))
    return mask

def bbox_from_sketch(sketch_img, expand=12):
    arr = np.array(sketch_img)
    ys, xs = np.where(arr > 0)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - expand); y0 = max(0, y0 - expand)
    x1 = min(arr.shape[1] - 1, x1 + expand); y1 = min(arr.shape[0] - 1, y1 + expand)
    return (int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1))

def union_bbox(a, b):
    if a is None: return b
    if b is None: return a
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = min(ax, bx); y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw); y2 = max(ay + ah, by + bh)
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

# ---------------------------
# Personality prompt builder (<= 77 tokens)
# ---------------------------
PERSONALITY_MAP = {
    "tsundere": {
        "low":  "tsundere, cold look, slight blush, arms crossed",
        "mid":  "tsundere, awkward smile, shy glance",
        "high": "tsundere, warm smile, gentle eyes",
    },
    "polite_junior": {
        "low":  "polite junior, formal tone, neutral face",
        "mid":  "polite junior, small smile, soft eyes",
        "high": "polite junior, warm smile, bright eyes",
    },
    "energetic": {
        "low":  "energetic, curious look, small grin",
        "mid":  "energetic, playful smirk",
        "high": "energetic, bright smile, sparkling eyes",
    },
    "onee_san": {
        "low":  "confident older sister, calm eyes, faint smirk",
        "mid":  "confident older sister, gentle smile",
        "high": "confident older sister, warm embrace mood, tender eyes",
    },
    "yandere": {
        "low":  "yandere, blank stare, slight blush",
        "mid":  "yandere, obsessive gaze, uneasy smile",
        "high": "yandere, intense stare, wide smile",
    },
    "bratty": {
        "low":  "bratty, mocking grin, playful eyes",
        "mid":  "bratty, teasing smirk",
        "high": "bratty, cheeky grin, close face",
    },
  
}

STYLE_SUFFIX = "consistent with furniture texture and lighting, kawaii anime style, flat pastel shading, clean vector-style outline, clean lineart with no noise, soft glossy highlights, poketmonster, cohesive mobile game asset style"

def build_personality_prompt(personality, affinity, base_neg):
    # personality가 주어지면 짧은 문구 + 스타일 꼬리표로 prompt 결정
    p = PERSONALITY_MAP.get(personality, {}).get(affinity, "")
    if p:
        final_prompt = f"{p}, {STYLE_SUFFIX}"
        # 공통 금지어(짧게)a simlpe nose bitween eyes and mouth, a single mouth in the face,
        final_neg = (base_neg.rstrip(", ") + ", angry expression, neutral expression").strip(", ")
        return final_prompt, final_neg
    return None, None

# ---------------------------
# Face Sketch (from landmarks) with triangulation options
# ---------------------------
def _draw_eye(draw, cx, cy, rx, ry, width=2):
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, outline=255, width=width)

def _poly(draw, points, width=2, close=False):
    if len(points) < 2:
        return
    draw.line(points + ([points[0]] if close else []), fill=255, width=width)

def _draw_delaunay(draw, pts, width=2):
    if len(pts) < 3:
        return
    tri = Delaunay(pts)
    for a, b, c in tri.simplices:
        poly = [tuple(map(float, pts[a])), tuple(map(float, pts[b])), tuple(map(float, pts[c]))]
        _poly(draw, poly, width=width, close=True)

def make_face_sketch(size, landmarks, lm_type="ext32", line=2, triangulate="none"):
    """
    triangulate: 'none' | 'mouth' | 'all'
    """
    W, H = size
    canvas = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(canvas)
    pts = landmarks

    if triangulate == "all":
        _draw_delaunay(d, pts, width=line)
        return canvas

    # 눈: 라인(타원) 스케치
    if lm_type == "lite20":
        eyeL = pts[8:12]; eyeR = pts[12:16]
        for eye in [eyeL, eyeR]:
            cx, cy = eye.mean(axis=0)
            ex = max(4, int(np.ptp(eye[:, 0]) * 0.5))
            ey = max(3, int(np.ptp(eye[:, 1]) * 0.8) + 2)
            _draw_eye(d, float(cx), float(cy), ex, ey, width=line)
        mouth_outer = pts[18:20]
        if triangulate == "mouth":
            _draw_delaunay(d, mouth_outer, width=line)
        else:
            if len(mouth_outer) == 2:
                _poly(d, [tuple(map(float, mouth_outer[0])), tuple(map(float, mouth_outer[1]))], width=line)
    else:  # ext32
        eyeL = pts[8:13]; eyeR = pts[13:18]
        for eye in [eyeL, eyeR]:
            cx, cy = eye.mean(axis=0)
            ex = max(4, int(np.ptp(eye[:, 0]) * 0.6) + 2)
            ey = max(3, int(np.ptp(eye[:, 1]) * 1.0) + 2)
            _draw_eye(d, float(cx), float(cy), ex, ey, width=line)

        mouth_outer = pts[22:30]
        mouth_inner = pts[30:32]
        if triangulate == "mouth":
            mp = np.concatenate([mouth_outer, mouth_inner], axis=0)
            _draw_delaunay(d, mp, width=line)
        else:
            _poly(d, [tuple(map(float, p)) for p in mouth_outer], width=line, close=True)
            if len(mouth_inner) == 2:
                _poly(d, [tuple(map(float, mouth_inner[0])), tuple(map(float, mouth_inner[1]))], width=line)

    return canvas

# ---------------------------
# Control images
# ---------------------------
def make_canny_control(rgb_img, low=100, high=200):
    arr = np.array(rgb_img)
    edges = cv2.Canny(arr, low, high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

# ---------------------------
# Diffusion pipeline
# ---------------------------
def build_pipeline(device="cuda", cache_root="/data/huggingface"):
    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(os.path.join(cache_root, "hub"), exist_ok=True)
    os.makedirs(os.path.join(cache_root, "diffusers"), exist_ok=True)
    os.makedirs(os.path.join(cache_root, "transformers"), exist_ok=True)
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_root, "transformers")
    os.environ["DIFFUSERS_CACHE"] = os.path.join(cache_root, "diffusers")

    controlnet_canny = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16,
        cache_dir=cache_root,
    )
    controlnet_scribble = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float16,
        cache_dir=cache_root,
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=[controlnet_canny, controlnet_scribble],
        torch_dtype=torch.float16,
        cache_dir=cache_root,
    )
    if device == "cpu":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.safety_checker = None
    return pipe

# ---------------------------
# Main
# ---------------------------
def main(args):
    os.makedirs(args.out, exist_ok=True)

    # 1) load
    img_rgba = load_rgba(args.image)
    rgb = rgba_to_rgb(img_rgba, bg=(255, 255, 255))
    rig, bbox, landmarks, lm_type = read_rig(args.rig)
    W, H = rgb.size

    # 2) sketch & bbox
    sketch = make_face_sketch(
        (W, H),
        landmarks,
        lm_type=lm_type,
        line=args.sketch_width,
        triangulate=args.triangulate,
    )
    sketch_bbox = bbox_from_sketch(sketch, expand=args.mask_expand)

    use_bbox = bbox
    if args.mask_source == "sketch" and sketch_bbox is not None:
        use_bbox = sketch_bbox
    elif args.mask_source == "union" and sketch_bbox is not None:
        use_bbox = union_bbox(bbox, sketch_bbox)

    mask = make_face_mask((W, H), use_bbox, shape="rounded",
                          feather=args.feather, round_radius_ratio=0.22)

    save_img(mask, os.path.join(args.out, "face_mask.png"))
    save_img(sketch, os.path.join(args.out, "face_sketch.png"))
    debug = Image.merge("RGB", (sketch, mask, mask))
    save_img(debug, os.path.join(args.out, "sketch_over_mask.png"))

    # 3) control images
    canny_img = make_canny_control(rgb, low=args.canny_low, high=args.canny_high)
    scribble_rgb = Image.merge("RGB", (sketch, sketch, sketch))
    save_img(canny_img, os.path.join(args.out, "control_canny.png"))
    save_img(scribble_rgb, os.path.join(args.out, "control_scribble.png"))

    # 3.5) prompt 선택: personality가 있으면 그것을 우선
    final_prompt, final_neg = None, None
    if args.personality:
        final_prompt, final_neg = build_personality_prompt(args.personality, args.affinity, args.neg)

    # personality가 비어있으면 기존 프롬프트 빌더(동공/홍채 추가)를 사용
    if not final_prompt:
        final_prompt = args.prompt
        # 눈/홍채 프롬프트 간단 추가(77 토큰 고려해 짧게)
        if args.add_pupils:
            color_phrase = f"{args.eye_color} iris" if args.eye_color else "colored iris"
            extra = f"anime eyes, black pupils, {color_phrase}, glossy highlights"
            final_prompt = final_prompt.rstrip(", ") + ", " + extra
        final_neg = (args.neg.rstrip(", ") + ", blurry eyes, extra eyes").strip(", ")

    # 4) diffusion
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    pipe = build_pipeline(device=device)

    result = pipe(
        prompt=final_prompt,
        negative_prompt=final_neg,
        image=rgb,
        mask_image=mask,
        control_image=[canny_img, scribble_rgb],
        controlnet_conditioning_scale=[args.canny, args.scribble],
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        strength=args.strength,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    ).images[0]

    out_path = os.path.join(args.out, "generated_face.png")
    save_img(result, out_path)
    print(f"[OK] saved: {out_path}")
    print("[PROMPT]", final_prompt)
    print("[NEG   ]", final_neg)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--rig", type=str, required=True)
    ap.add_argument("--out", type=str, default="out_face")

    # 일반 프롬프트(성격 미사용 시)
    ap.add_argument("--prompt", type=str, default=(
        "kawaii chibi sticker face on the furniture surface, "
        "large round eyes, tiny smiling mouth, flat pastel shading, "
        "thick outline, consistent lighting"
    ))
    ap.add_argument("--neg", type=str, default="photorealistic, complex background, text, watermark, strong shadow")

    # 성격 프롬프트 옵션 (있으면 위 prompt/neg 대신 사용)
    ap.add_argument("--personality", type=str, default="", choices=[
        "", "tsundere", "polite_junior", "energetic", "onee_san", "yandere", "bratty", "shy_housewife"
    ])
    ap.add_argument("--affinity", type=str, default="high", choices=["low", "mid", "high"])

    # 눈 옵션(성격 미사용 시에만 프롬프트에 추가)
    ap.add_argument("--add_pupils", action="store_true", default=True)
    ap.add_argument("--eye_color", type=str, default="brown")
    ap.add_argument("--anime_eyes", action="store_true")

    # 샘플링
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=42)

    # 마스크/스케치
    ap.add_argument("--feather", type=int, default=10)
    ap.add_argument("--sketch_width", type=int, default=2)
    ap.add_argument("--canny_low", type=int, default=100)
    ap.add_argument("--canny_high", type=int, default=200)
    ap.add_argument("--canny", type=float, default=0.5)
    ap.add_argument("--scribble", type=float, default=0.85)
    ap.add_argument("--mask_source", type=str, default="union", choices=["bbox", "sketch", "union"])
    ap.add_argument("--mask_expand", type=int, default=16)
    ap.add_argument("--triangulate", type=str, default="none", choices=["none", "mouth", "all"],
                    help="랜드마크를 들로네 삼각분할로 스케치에 반영")

    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
