import os
import json
import argparse
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import cv2

# ============================================================
# mod_face.py — Object-with-face 전용: rig.json(bbox+landmarks) 기반 표정만 수정
#  - SDXL Inpainting 사용: 마스크(얼굴 영역)만 재생성 → 물건의 나머지 부분 불변
#  - IP-Adapter(선택): 기존 선, 색감, 질감 보존 강화 (style refs)
#  - rig.json: bbox, image_size, landmarks_world_* 를 사용하여 마스크 자동 생성
# ============================================================

DEFAULT_OBJECT_DIRECTIVE = (
    "Modify only the exaggerated facial expression, clean vector-style outline, keep texture and colors consistent with the object"
    # "preserve object materials, textures, surface shading, geometry, pose, camera angle, background, and composition. "
    # "Do not change clothing or body since this is not a person, keep lines and colors consistent with the object."
)

DEFAULT_NEG = (
    "extra limbs, multiple faces, deformed eyes, distorted anatomy, text, watermark, artifacts, "
    "different object shape, different background, change of pose, change of materials, change of design"
)

EMOTION_SNIPPETS = {
    "neutral": "neutral expression, relaxed simple eyes, balanced eyelids",
    "smile": "big smile, upturned mouth corners, slight cheek lift, gentle eye squint, cheerful expression",
    "happy": "smiling mouth, bright eyes, raised cheeks, joyful expression",
    "sad": "downturned mouth corners, soft eyebrows, watery eyes, sad expression",
    "angry": "furrowed brows, narrowed eyes, tense mouth, angry expression",
    "surprised": "O-shaped mouth, widened eyes, raised eyebrows, surprised expression",
    "shy": "gentle closed-mouth smile, slight blush, softened eyes, shy expression",
    "confused": "asymmetrical mouth, tilted eyebrows, puzzled eyes, confused expression",
    "sleepy": "sleepy expression, droopy eyelids, relaxed mouth, calm face, sleeping mood, sleepy eyes",
    "panic": "open mouth, widened eyes, tiny sweat drop detail, panicked expression"
}


# ---------------------------------
# 캐시/프롬프트 유틸
# ---------------------------------

def setup_hf_cache(cache_root: str):
    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(os.path.join(cache_root, "hub"), exist_ok=True)
    os.makedirs(os.path.join(cache_root, "diffusers"), exist_ok=True)
    os.makedirs(os.path.join(cache_root, "transformers"), exist_ok=True)
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_root, "transformers")
    os.environ["DIFFUSERS_CACHE"] = os.path.join(cache_root, "diffusers")


def fuse_prompts(emotion_prompt: str,
                 object_directive: str = DEFAULT_OBJECT_DIRECTIVE,
                 user_prompt: str = "",
                 intensity: Optional[float] = None,
                 max_words: int = 120) -> str:
    # 강도(intensity) 설명어 보정 (0~1): subtle / strong
    intens_phrase = None
    if intensity is not None:
        try:
            t = float(intensity)
            if t <= 0.35:
                intens_phrase = "subtle"
            elif t >= 0.75:
                intens_phrase = "strong"
        except Exception:
            pass
    parts = []
    if intens_phrase:
        parts.append(intens_phrase)
    parts.append(emotion_prompt)
    if user_prompt:
        parts.append(user_prompt)
    parts.append(object_directive)
    base = ", ".join([p for p in parts if p])
    words = base.split()
    if len(words) > max_words:
        base = " ".join(words[:max_words])
    return base


def parse_csv_paths(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out or None


def load_style_refs(paths: List[str], size: int = 512) -> List[Image.Image]:
    imgs = []
    for p in paths:
        if not p:
            continue
        if not os.path.exists(p):
            print(f"[WARN] style_ref not found: {p}")
            continue
        img = load_image(p).convert("RGB")
        imgs.append(img.resize((size, size), Image.LANCZOS))
    return imgs


# ---------------------------------
# rig.json 로더 + 마스크 생성(랜드마크 클러스터)
# ---------------------------------

def load_rig(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 필수 키 체크
    assert "bbox" in data and "image_size" in data and "landmarks_world_neutral" in data, "rig.json must contain bbox, image_size, landmarks_world_neutral"
    return data


def _kmeans(points: np.ndarray, k: int = 3, iters: int = 15, seed: int = 0) -> np.ndarray:
    """간단 k-means (의존성 없이). points: (N,2) -> labels: (N,)"""
    assert points.ndim == 2 and points.shape[1] == 2 and points.shape[0] >= k
    rng = np.random.RandomState(seed)
    # 초기 중심: x+y 순으로 정렬 후 k개 고정 샘플(분산 확보)
    order = np.argsort(points[:, 0] + points[:, 1])
    cent = points[order[np.linspace(0, len(points)-1, k).astype(int)]]
    for _ in range(iters):
        # 할당
        d2 = ((points[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        lab = d2.argmin(axis=1)
        # 업데이트
        for j in range(k):
            sel = points[lab == j]
            if len(sel) > 0:
                cent[j] = sel.mean(axis=0)
    return lab

# --- add this just below make_face_mask_from_rig() ---
def make_face_mask_from_bbox(
    image_size: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    feather: int = 21,
    dilate_px: int = 8,
    expand_px: int = 0,
    shape: str = "ellipse",   # "ellipse" | "rounded" | "rect"
    roundness: float = 0.25,  # for "rounded": 0.0~0.5 권장
) -> Image.Image:
    """
    bbox를 기준으로 얼굴 영역 마스크 생성.
    - expand_px: bbox를 사방으로 확장 (px)
    - shape: 타원(ellipse), 라운드 사각(rounded), 일반 사각(rect)
    - roundness: 라운드 사각일 때 코너 반경 비율
    반환: L 모드 마스크(흰=수정, 검정=보존)
    """
    W, H = image_size
    x, y, w, h = map(int, bbox)

    # 확장 및 클램프
    x0 = max(0, x - expand_px)
    y0 = max(0, y - expand_px)
    x1 = min(W, x + w + expand_px)
    y1 = min(H, y + h + expand_px)

    # 빈 마스크
    m_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(m_img)

    if shape == "ellipse":
        draw.ellipse([x0, y0, x1 - 1, y1 - 1], fill=255)
    elif shape == "rounded":
        rr = int(min(x1 - x0, y1 - y0) * float(max(0.0, min(0.5, roundness))))
        # PIL은 rounded_rectangle을 지원
        draw.rounded_rectangle([x0, y0, x1 - 1, y1 - 1], radius=rr, fill=255)
    else:  # "rect"
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=255)

    m = np.array(m_img, dtype=np.uint8)

    # 확장(dilate)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)

    # feather blur
    if feather and feather > 1 and feather % 2 == 1:
        m = cv2.GaussianBlur(m, (feather, feather), 0)

    return Image.fromarray(m, mode="L")


def make_face_mask_from_rig(
    image_size: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    landmarks: List[List[float]],
    feather: int = 21,
    dilate_px: int = 8,
    cluster_k: int = 3,
) -> Image.Image:
    """rig.json 정보로 얼굴(눈/입) 주변만 선택하는 소프트 마스크 생성.
    - landmarks는 ext32 등 어떤 포맷이든 좌표가 이미지 기준이면 동작
    - k-means로 3클러스터(왼눈/오른눈/입)로 나눠 각 클러스터의 convex hull을 합성
    - bbox 내부로 클램프 후 feather 블러 적용
    반환: L 모드 마스크(흰=수정, 검정=보존)
    """
    W, H = image_size
    x, y, w, h = map(int, bbox)
    x = max(0, min(W - 1, x)); y = max(0, min(H - 1, y))
    w = max(1, min(W - x, w)); h = max(1, min(H - y, h))

    pts = np.array(landmarks, dtype=np.float32)
    # 안전 장치: bbox 범위 밖 좌표 컷
    pts[:, 0] = np.clip(pts[:, 0], x, x + w - 1)
    pts[:, 1] = np.clip(pts[:, 1], y, y + h - 1)

    mask = np.zeros((H, W), dtype=np.uint8)

    if len(pts) >= cluster_k:
        labels = _kmeans(pts, k=cluster_k)
        for kidx in range(cluster_k):
            cluster = pts[labels == kidx]
            if len(cluster) < 3:
                # 점이 적으면 작은 원으로 대체
                for (cx, cy) in cluster:
                    cv2.circle(mask, (int(cx), int(cy)), 6, 255, -1)
                continue
            hull = cv2.convexHull(cluster.astype(np.float32))
            cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    else:
        # 점이 너무 적으면 모든 점 주변 원 합성
        for (cx, cy) in pts:
            cv2.circle(mask, (int(cx), int(cy)), 8, 255, -1)

    # dilate로 약간 확장 후 bbox로 자르기
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
        mask = cv2.dilate(mask, k)

    rect = np.zeros_like(mask)
    rect[y:y+h, x:x+w] = 255
    mask = cv2.bitwise_and(mask, rect)

    # Feathering
    if feather and feather > 1 and feather % 2 == 1:
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)

    return Image.fromarray(mask)


def save_debug_overlay(img: Image.Image, landmarks: List[List[float]], mask: Image.Image, out_path: str):
    """랜드마크/마스크 시각화 저장"""
    vis = img.convert("RGB").copy()
    d = ImageDraw.Draw(vis)
    colors = [(255, 80, 80), (80, 255, 80), (80, 120, 255), (255, 200, 50)]
    for i, (px, py) in enumerate(landmarks):
        c = colors[i % len(colors)]
        r = 3
        d.ellipse((px-r, py-r, px+r, py+r), outline=c, width=2)
    # 마스크 외곽선
    m = np.array(mask)
    edges = cv2.Canny(m, 64, 128)
    ys, xs = np.where(edges > 0)
    for xi, yi in zip(xs, ys):
        vis.putpixel((int(xi), int(yi)), (0, 255, 255))
    vis.save(out_path)


# ---------------------------------
# 파이프라인 빌더 (SDXL Inpainting + optional IP-Adapter)
# ---------------------------------

def build_inpaint_pipe(
    device: str = "cuda",
    cache_root: str = "/data/huggingface",
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    ip_repo: str = "h94/IP-Adapter",
    ip_weight: str = "ip-adapter-plus_sdxl_vit-h.safetensors",
    num_adapters: int = 1,
):
    setup_hf_cache(cache_root)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        add_watermarker=False,
        use_safetensors=True,
        cache_dir=cache_root,
    )

    # IP-Adapter (가능 시)
    try:
        weight_names = [ip_weight] * max(1, num_adapters)
        subfolder = "sdxl_models"
        pipe.load_ip_adapter(
            ip_repo,
            subfolder=subfolder,
            weight_name=weight_names,
            image_encoder_folder="models/image_encoder",
            cache_dir=cache_root,
        )
    except Exception as e:
        print(f"[WARN] load_ip_adapter failed with '{ip_weight}': {e}")
        try:
            fallback_weight = "ip-adapter_sdxl.safetensors"
            weight_names = [fallback_weight] * max(1, num_adapters)
            pipe.load_ip_adapter(
                ip_repo,
                subfolder="sdxl_models",
                weight_name=weight_names,
                cache_dir=cache_root,
            )
            print("[INFO] Fallback IP-Adapter weight loaded.")
        except Exception as e2:
            print(f"[INFO] IP-Adapter not applied (continuing without it): {e2}")

    if device == "cpu":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    pipe.safety_checker = None
    return pipe


# ---------------------------------
# 표정 수정 (rig.json 구동)
# ---------------------------------
@torch.inference_mode()
def modify_expression_with_rig(
    input_path: str,
    rig_path: str,
    out_path: str,
    emotion: Optional[str] = None,
    user_prompt: str = "",
    negative_prompt: str = DEFAULT_NEG,
    steps: int = 40,
    guidance: float = 5.5,
    base_strength: float = 0.4,
    seed: int = 123,
    device: str = "cuda",
    cache_root: str = "/data/huggingface",
    style_refs: Optional[List[Image.Image]] = None,
    style_weights: Optional[List[float]] = None,
    ip_strength: float = 0.6,
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    feather: int = 21,
    dilate_px: int = 8,
    mask_mode: str = "landmarks",   # "landmarks" | "bbox" | "auto"
    bbox_expand_px: int = 0,
    bbox_shape: str = "ellipse",    # "ellipse" | "rounded" | "rect"
    bbox_roundness: float = 0.25,
):
    assert os.path.exists(input_path), f"input not found: {input_path}"
    assert os.path.exists(rig_path), f"rig not found: {rig_path}"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rig = load_rig(rig_path)
    bbox = tuple(map(int, rig["bbox"]))
    image_size = tuple(map(int, rig["image_size"]))
    landmarks = rig.get("landmarks_world_emotion") or rig.get("landmarks_world_neutral")
    assert landmarks and len(landmarks) >= 6, "rig landmarks are missing or too few"

    # 강도 매핑: rig.intensity in [0,1] -> denoise strength 범위 조절
    intensity = float(rig.get("intensity", 0.6))
    strength = float(base_strength + 0.35 * np.clip(intensity, 0.0, 1.0))  # ~0.4~0.75
    strength = float(np.clip(strength, 0.3, 0.8))

    # 감정 설정(우선순위: 함수 인자 > rig.json)
    emo_key = (emotion or rig.get("emotion") or "neutral").lower().strip()
    emo_prompt = EMOTION_SNIPPETS.get(emo_key, emo_key)
    fused_prompt = fuse_prompts(emo_prompt, user_prompt=user_prompt, intensity=intensity)

    base_img = load_image(input_path).convert("RGB")
    W0, H0 = base_img.size
    assert (W0, H0) == tuple(image_size), f"rig image_size {image_size} != image size {(W0,H0)}"

    # 마스크 생성
    if mask_mode == "landmarks":
        mask_img = make_face_mask_from_rig(image_size, bbox, landmarks, feather=feather, dilate_px=dilate_px)
    elif mask_mode == "bbox":
        mask_img = make_face_mask_from_bbox(
            image_size, bbox,
            feather=feather, dilate_px=dilate_px,
            expand_px=bbox_expand_px, shape=bbox_shape, roundness=bbox_roundness
        )
    else:  # "auto": 랜드마크 충분하면 landmarks, 아니면 bbox
        if landmarks and len(landmarks) >= 6:
            mask_img = make_face_mask_from_rig(image_size, bbox, landmarks, feather=feather, dilate_px=dilate_px)
        else:
            mask_img = make_face_mask_from_bbox(
                image_size, bbox,
                feather=feather, dilate_px=dilate_px,
                expand_px=bbox_expand_px, shape=bbox_shape, roundness=bbox_roundness
            )

    # 디버그 오버레이 저장
    dbg_path = os.path.splitext(out_path)[0] + "_debug.png"
    save_debug_overlay(base_img, landmarks, mask_img, dbg_path)

    # 파이프라인
    num_adapters = len(style_refs) if style_refs else 0
    pipe = build_inpaint_pipe(
        device=device,
        cache_root=cache_root,
        model_id=model_id,
        num_adapters=max(1, num_adapters),
    )

    if style_refs and len(style_refs) > 0:
        scales = style_weights if (style_weights and len(style_weights) == len(style_refs)) else [ip_strength] * len(style_refs)
        try:
            pipe.set_ip_adapter_scale(scales)
            print(f"[IP-Adapter] scales = {scales}")
        except Exception as e:
            print(f"[WARN] set_ip_adapter_scale failed: {e}")

    print(f"[PROMPT] {fused_prompt}")
    print(f"[NEG]    {negative_prompt}")
    print(f"[RIG] bbox={bbox}, intensity={intensity:.2f}, strength={strength:.2f}, emotion={emo_key}")

    g = torch.Generator(device=device).manual_seed(seed)

    kwargs = dict(
        prompt=fused_prompt,
        negative_prompt=negative_prompt,
        image=base_img,
        mask_image=mask_img,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        generator=g,
    )

    if style_refs and len(style_refs) > 0:
        kwargs["ip_adapter_image"] = style_refs

    out_img = pipe(**kwargs).images[0]
    out_img.save(out_path)
    print(f"[OK] saved: {out_path}")

    # 보조 산출물 저장
    mask_vis = ImageOps.colorize(mask_img, black="black", white="white")
    mask_vis.save(os.path.splitext(out_path)[0] + "_mask.png")


# ---------------------------------
# CLI
# ---------------------------------
personality = "energetic"  # tsundere, polite_junior, energetic, onee_san, yandere, bratty
funiture = "pen"  # chair, table, pen, etc.
rig_path = f"/workspace/dhe_project/generation/out_{funiture}/rig.json"
img_path = f"/workspace/dhe_project/generation/save/pen_{personality}_face/generated_face.png"
out_path = f"/workspace/dhe_project/assets/emotion/{funiture}/{personality}_face.png"
# neutral, smile, happy, sad, angry, surprised, shy, confused, sleepy, panic


def main():
    ap = argparse.ArgumentParser(description="Modify ONLY the facial expression on an object using rig.json (bbox+landmarks) with SDXL Inpainting + optional IP-Adapter")
    # I/O
    ap.add_argument("--input", type=str, default=img_path, help="input image path")
    ap.add_argument("--rig", type=str, default=rig_path, help="rig.json path containing bbox, image_size, and landmarks")
    ap.add_argument("--out", type=str, default="/workspace/dhe_project/assets/edited/out.png", help="output image path")
    ap.add_argument("--cache_root", type=str, default="/data/huggingface")

    # Prompts
    ap.add_argument("--emotion", type=str, default='angry', help=f"override emotion (else use rig.json); one of: {','.join(EMOTION_SNIPPETS.keys())} or custom English phrase")
    ap.add_argument("--user_prompt", type=str, default="", help="extra constraints, e.g., 'keep same line thickness and color' ")
    ap.add_argument("--neg", type=str, default=DEFAULT_NEG, help="negative prompt")

    # Inference
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--base_strength", type=float, default=0.15, help="baseline denoise (final strength scales with rig.intensity)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--mask_mode", type=str, default="bbox",
                    choices=["landmarks", "bbox", "auto"],
                    help="how to build mask: landmarks | bbox | auto(fallback to bbox if landmarks insufficient)")
    ap.add_argument("--bbox_expand_px", type=int, default=0, help="expand bbox by N px on all sides before making mask")
    ap.add_argument("--bbox_shape", type=str, default="ellipse",
                    choices=["ellipse", "rounded", "rect"], help="shape for bbox mask")
    ap.add_argument("--bbox_roundness", type=float, default=0.15,
                    help="rounded-rect corner radius ratio (0~0.5)")
    
    # Style refs (optional)
    ap.add_argument("--style_refs", type=str, default=img_path, help="comma-separated image paths as style references")
    ap.add_argument("--style_weights", type=str, default="", help="comma-separated scales; must match refs count if provided")
    ap.add_argument("--ip_strength", type=float, default=0.25, help="uniform IP-Adapter scale if style_weights not given")

    # Mask tuning
    ap.add_argument("--feather", type=int, default=21, help="Gaussian feather(odd) for mask edges")
    ap.add_argument("--dilate_px", type=int, default=8, help="convex hull dilation radius in pixels")

    # Model
    ap.add_argument("--model_id", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", help="SDXL inpainting model repo id")

    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    refs = parse_csv_paths(args.style_refs)
    ref_imgs = load_style_refs(refs, size=512) if refs else None
    weights = parse_csv_floats(args.style_weights)

    # fix funiture, 
    funiture='sofa'
    for personality in [ 'yandere', 'energetic', 'tsundere', 'polite_junior', 'onee_san', 'bratty']:
        for emotion in EMOTION_SNIPPETS.keys():
            args.out = f"/workspace/dhe_project/assets/emotion/{funiture}/{personality}/{emotion}_face.png"
            args.rig = f"/workspace/dhe_project/generation/out_{funiture}/rig.json"
            args.input = f"/workspace/dhe_project/generation/save/{funiture}_{personality}_face/generated_face.png"
            args.style_refs = f"/workspace/dhe_project/generation/save/{funiture}_{personality}_face/generated_face.png"
            refs= parse_csv_paths(args.style_refs)
            ref_imgs = load_style_refs(refs, size=512) if refs else None
            weights = parse_csv_floats(args.style_weights)
    
            print(f"[INFO] Running with emotion: {args.emotion}, output: {args.out}")
            # 표정 수정 실행
            modify_expression_with_rig(
                input_path=args.input,
                rig_path=args.rig,
                out_path=args.out,
                emotion=emotion,  # 각 감정별로 실행
                user_prompt=args.user_prompt,
                negative_prompt=args.neg,
                steps=args.steps,
                guidance=args.cfg,
                base_strength=args.base_strength,
                seed=args.seed,
                device=device,
                cache_root=args.cache_root,
                style_refs=ref_imgs,
                style_weights=weights,
                ip_strength=args.ip_strength,
                model_id=args.model_id,
                feather=args.feather,
                dilate_px=args.dilate_px,
                mask_mode=args.mask_mode,
                bbox_expand_px=args.bbox_expand_px,
                bbox_shape=args.bbox_shape,
                bbox_roundness=args.bbox_roundness
            )

    # modify_expression_with_rig(
    #     input_path=args.input,
    #     rig_path=args.rig,
    #     out_path=args.out,
    #     emotion=args.emotion,
    #     user_prompt=args.user_prompt,
    #     negative_prompt=args.neg,
    #     steps=args.steps,
    #     guidance=args.cfg,
    #     base_strength=args.base_strength,
    #     seed=args.seed,
    #     device=device,
    #     cache_root=args.cache_root,
    #     style_refs=ref_imgs,
    #     style_weights=weights,
    #     ip_strength=args.ip_strength,
    #     model_id=args.model_id,
    #     feather=args.feather,
    #     dilate_px=args.dilate_px,
    #     mask_mode=args.mask_mode,
    #     bbox_expand_px=args.bbox_expand_px,
    #     bbox_shape=args.bbox_shape,
    #     bbox_roundness=args.bbox_roundness,
    # )


if __name__ == "__main__":
    main()
