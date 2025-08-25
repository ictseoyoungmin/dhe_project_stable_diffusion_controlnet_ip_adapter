import os, json, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from scipy.spatial import Delaunay  # Delaunay 삼각변환을 위해 추가

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

def rgba_to_rgb(img_rgba, bg=(255,255,255)):
    bg_img = Image.new("RGB", img_rgba.size, bg)
    bg_img.paste(img_rgba, mask=img_rgba.split()[-1])
    return bg_img

def save_img(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def read_rig(path):
    with open(path, "r", encoding="utf-8") as f:
        rig = json.load(f)
    # prefer emotion landmarks if present, else neutral
    if "landmarks_world_emotion" in rig and len(rig["landmarks_world_emotion"]) > 0:
        pts = np.array(rig["landmarks_world_emotion"], dtype=np.float32)
    else:
        pts = np.array(rig["landmarks_world_neutral"], dtype=np.float32)
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
        draw.ellipse([x, y, x+w, y+h], fill=255)
    else:
        r = int(max(2, min(w, h) * round_radius_ratio))
        draw.rounded_rectangle([x, y, x+w, y+h], radius=r, fill=255)

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=int(feather)))
    return mask

def bbox_from_sketch(sketch_img, expand=12):
    """Sketch(L mode)에서 선의 바운딩박스를 구하고 margin(expand)을 준다."""
    arr = np.array(sketch_img)
    ys, xs = np.where(arr > 0)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - expand); y0 = max(0, y0 - expand)
    x1 = min(arr.shape[1]-1, x1 + expand); y1 = min(arr.shape[0]-1, y1 + expand)
    return (int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1))

def union_bbox(a, b):
    if a is None: return b
    if b is None: return a
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = min(ax, bx); y1 = min(ay, by)
    x2 = max(ax+aw, bx+bw); y2 = max(ay+ah, by+bh)
    return (int(x1), int(y1), int(x2-x1), int(y2-y1))

# ---------------------------
# Face Sketch (from landmarks)
# ---------------------------
def _draw_eye(draw, cx, cy, rx, ry, width=2):
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, outline=255, width=width)

def _poly(draw, points, width=2, close=False):
    if len(points) < 2:
        return
    draw.line(points + ([points[0]] if close else []), fill=255, width=width)

def make_face_sketch(size, landmarks, lm_type="ext32", line=2):
    """
    size: (W,H)
    landmarks: (N,2) in world coords
    """
    W, H = size
    canvas = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(canvas)

    pts = landmarks

    # 눈 그리기 (기존 코드 유지)
    if lm_type == "lite20":
        eyeL = pts[8:12]; eyeR = pts[12:16]
        for eye in [eyeL, eyeR]:
            cx, cy = eye.mean(axis=0)
            ex = max(4, int(np.ptp(eye[:,0]) * 0.5))
            ey = max(3, int(np.ptp(eye[:,1]) * 0.8) + 2)
            _draw_eye(d, float(cx), float(cy), ex, ey, width=line)
    else:  # ext32
        eyeL = pts[8:13]; eyeR = pts[13:18]
        for eye in [eyeL, eyeR]:
            cx, cy = eye.mean(axis=0)
            ex = max(4, int(np.ptp(eye[:,0]) * 0.6) + 2)
            ey = max(3, int(np.ptp(eye[:,1]) * 1.0) + 2)
            _draw_eye(d, float(cx), float(cy), ex, ey, width=line)

    # 입: 들로네 삼각변환 적용
    if lm_type == "lite20":
        # lite20: 입의 랜드마크는 pts[18:20] (2개 포인트로 제한적)
        # 들로네 삼각변환을 위해 추가 포인트를 생성하거나 단순히 선으로 연결
        mouth_pts = pts[18:20]
        if len(mouth_pts) >= 2:
            _poly(d, list(map(lambda p: (float(p[0]), float(p[1])), mouth_pts)), width=line)
    else:  # ext32
        # ext32: 입의 외곽(pts[22:30])과 내곽(pts[30:32]) 사용
        mouth_pts = np.concatenate([pts[22:30], pts[30:32]], axis=0)  # 외곽 8개 + 내곽 2개
        if len(mouth_pts) >= 3:  # 들로네 삼각변환은 최소 3개 포인트 필요
            # 들로네 삼각변환 수행
            tri = Delaunay(mouth_pts)
            # 삼각형의 외곽선 그리기
            for simplex in tri.simplices:
                points = [tuple(map(float, mouth_pts[i])) for i in simplex]
                _poly(d, points, width=line, close=True)

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
# Prompt builder (pupils/iris)
# ---------------------------
def build_prompts(base_prompt: str, base_neg: str, add_pupils: bool, eye_color: str, anime_eyes: bool):
    if not args.mod_facial:
        extras = []
        if anime_eyes:
            extras.append("anime style eyes, manga-style eye detailing")
        if add_pupils:
            color_phrase = f"{eye_color} iris" if eye_color else "colored iris"
            extras.append(f"large round eyes with black pupils and {color_phrase}, glossy eyes with sparkling highlights, visible pupils and iris details")
        final_prompt = base_prompt
        if extras:
            final_prompt = base_prompt.rstrip(", ") + ", " + ", ".join(extras)
        # negative prompt 보강: 눈 관련 오류 방지
        neg_extras = "closed eyes, extra eyes, deformed pupils, blurry eyes"
        final_neg = base_neg.rstrip(", ") + ", " + neg_extras
    else: # 표정 변화 작업시
        final_prompt, final_neg = base_prompt, base_neg
    return final_prompt, final_neg

# ---------------------------
# Diffusion pipeline
# ---------------------------
def build_pipeline(device="cuda", cache_root="/data/huggingface"):
    # 1) 캐시 경로 보장
    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(os.path.join(cache_root, "hub"), exist_ok=True)
    os.makedirs(os.path.join(cache_root, "diffusers"), exist_ok=True)
    os.makedirs(os.path.join(cache_root, "transformers"), exist_ok=True)

    # 2) Hugging Face 캐시 환경변수
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_root, "transformers")
    os.environ["DIFFUSERS_CACHE"] = os.path.join(cache_root, "diffusers")

    # 3) 모델 로드 (cache_dir 명시)
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
        "runwayml/stable-diffusion-inpainting", # runwayml/stable-diffusion-inpainting, lllyasviel/control_v11p_sd15_inpaint
        controlnet=[controlnet_canny, controlnet_scribble],
        torch_dtype=torch.float16,
        cache_dir=cache_root,
    )

    # 4) 디바이스
    if device == "cpu":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # 5) safety checker 비활성
    pipe.safety_checker = None
    return pipe

# ---------------------------
# Main
# ---------------------------
def main(args):
    os.makedirs(args.out, exist_ok=True)

    # 1) load
    img_rgba = load_rgba(args.image)
    rgb = rgba_to_rgb(img_rgba, bg=(255,255,255))
    rig, bbox, landmarks, lm_type = read_rig(args.rig)
    W, H = rgb.size

    # 2) sketch & bbox
    sketch = make_face_sketch((W, H), landmarks, lm_type=lm_type, line=args.sketch_width)
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
    # 디버그: 스케치가 마스크 안에 들어가는지 시각화(R=sketch, G/B=mask)
    debug = Image.merge("RGB", (sketch, mask, mask))
    save_img(debug, os.path.join(args.out, "sketch_over_mask.png"))

    # 3) control images
    canny_img = make_canny_control(rgb, low=args.canny_low, high=args.canny_high)
    scribble_rgb = Image.merge("RGB", (sketch, sketch, sketch))
    save_img(canny_img, os.path.join(args.out, "control_canny.png"))
    save_img(scribble_rgb, os.path.join(args.out, "control_scribble.png"))

    # 3.5) build prompts with eye options
    final_prompt, final_neg = build_prompts(
        args.prompt, args.neg, args.add_pupils, args.eye_color, args.anime_eyes
    )

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
        generator=torch.Generator(device=device).manual_seed(args.seed)
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
    ap.add_argument("--prompt", type=str, default=(
        "kawaii chibi sticker face on the furniture surface, "
        "large round eyes, tiny smiling mouth, flat pastel shading, "
        "thick outline, consistent lighting, cute character expression"
    ))
    ap.add_argument("--neg", type=str, default=(
        "photorealistic, complex background, text, watermark, strong shadow"
    ))
    # eye options
    ap.add_argument("--add_pupils", action="store_true", default=False,
                    help="프롬프트에 pupils/iris/하이라이트 문구를 자동 추가")
    ap.add_argument("--eye_color", type=str, default=None,
                    help="iris 색상 (예: blue, brown, green, amber, violet, black)")
    ap.add_argument("--anime_eyes", action="store_true",
                    help="anime/manga 스타일 눈 묘사를 프롬프트에 추가")

    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feather", type=int, default=10)
    ap.add_argument("--sketch_width", type=int, default=2)
    ap.add_argument("--canny_low", type=int, default=100)
    ap.add_argument("--canny_high", type=int, default=200)
    ap.add_argument("--canny", type=float, default=0.5, help="Control weight for canny")
    ap.add_argument("--scribble", type=float, default=0.85, help="Control weight for sketch")
    ap.add_argument("--mask_source", type=str, default="union",
                    choices=["bbox", "sketch", "union"],
                    help="마스크 기준 선택: rig bbox / 스케치 bbox / 둘의 합집합")
    ap.add_argument("--mask_expand", type=int, default=16,
                    help="스케치 bbox 주변 margin (px)")
    ap.add_argument("--mod_facial", action="store_true", help="change facial expression")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()
    main(args)


#     main(args)
