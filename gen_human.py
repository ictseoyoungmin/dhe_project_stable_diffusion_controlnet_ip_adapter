import os
import argparse
from typing import List, Optional

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image


# -----------------------------
# Prompt utils
# -----------------------------
DEFAULT_CORE_PROMPT = (
    # --- 전신 강조 (핵심만 남김) ---
    "an adult human character with semi-realistic proportions, clean lines, colors from references, "
    # "kawaii full body shot, standing pose, casual clothing, "
    "full body shot, standing pose, casual clothing, "
    "kawaii anime-style full body shot, adult body proportions, detailed legs and arms, "
    "stylized cartoon facial features, white background, mobile game asset style, soft shadows, centered composition"
    "detailed eyes from references"
)
DEFAULT_NEG = (
    # --- 인물 복제 및 다수 인물 방지 ---
    # "dual, two people, multiple people, twins, duplicate, front and side view, multiple angles, group, 2girls, 2boys, "
    # # --- 기존 부정 프롬프트 ---
    "children, child, kid, "
    "unrealistic proportions, distorted anatomy, cartoonish exaggeration"
    "blurry, low quality, distorted face, extra limbs"
)


def fuse_prompts(user_prompt: str,
                 core_prompt: str = DEFAULT_CORE_PROMPT,
                 max_words: int = 120) -> str:
    """
    Combine user prompt + core prompt for human generation.
    User prompt comes first for specific details (e.g., clothing, pose).
    Truncate to max_words to avoid SDXL quality degradation.
    """
    user_prompt = (user_prompt or "").strip()
    fused = f"{user_prompt}, {core_prompt}" if user_prompt else core_prompt
    words = fused.split()
    if len(words) > max_words:
        fused = " ".join(words[:max_words])
    return fused


# -----------------------------
# Cache env setup (checkpoint rule)
# -----------------------------
def setup_hf_cache(cache_root: str):
    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(os.path.join(cache_root, "hub"), exist_ok=True)
    os.makedirs(os.path.join(cache_root, "diffusers"), exist_ok=True)
    os.makedirs(os.path.join(cache_root, "transformers"), exist_ok=True)
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_root, "transformers")
    os.environ["DIFFUSERS_CACHE"] = os.path.join(cache_root, "diffusers")


# -----------------------------
# Style helpers (IP-Adapter for SDXL)
# -----------------------------
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


def parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    parts = [x.strip() for x in s.split(",") if x.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            pass
    return vals or None


def parse_csv_paths(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# -----------------------------
# Pipeline builder
# -----------------------------
def build_sdxl_ip_adapter(
    device="cuda",
    cache_root="/data/huggingface",
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    ip_repo="h94/IP-Adapter",
    ip_weight="ip-adapter-plus_sdxl_vit-h.safetensors",
    num_adapters: int = 1,
):
    setup_hf_cache(cache_root)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        add_watermarker=False,
        use_safetensors=True,
        cache_dir=cache_root,
    )

    weight_names = [ip_weight] * num_adapters
    subfolder = "sdxl_models"

    try:
        pipe.load_ip_adapter(
            ip_repo,
            subfolder=subfolder,
            weight_name=weight_names,
            image_encoder_folder="models/image_encoder",
            cache_dir=cache_root,
        )
    except Exception as e:
        print(f"[WARN] load_ip_adapter failed with '{ip_weight}': {e}")
        print("[INFO] Falling back to 'ip-adapter_sdxl.safetensors'")
        fallback_weight = "ip-adapter_sdxl.safetensors"
        weight_names = [fallback_weight] * num_adapters
        pipe.load_ip_adapter(
            ip_repo,
            subfolder=subfolder,
            weight_name=weight_names,
            cache_dir=cache_root,
        )

    if device == "cpu":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.safety_checker = None
    return pipe


# -----------------------------
# Generation
# -----------------------------
@torch.inference_mode()
def generate(
    out_dir: str,
    user_prompt: str,
    core_prompt: str = DEFAULT_CORE_PROMPT,
    negative_prompt: str = DEFAULT_NEG,
    style_refs: Optional[List[Image.Image]] = None,
    style_weights: Optional[List[float]] = None,
    ip_strength: float = 0.65,
    width: int = 512,
    height: int = 768,
    steps: int = 40,
    guidance: float = 7.5,
    seed: int = 123,
    samples: int = 1,
    per_prompt_seed_shift: int = 0,
    device: str = "cuda",
    cache_root: str = "/data/huggingface",
):
    os.makedirs(out_dir, exist_ok=True)

    # Pipeline
    num_adapters = len(style_refs) if style_refs else 1
    pipe = build_sdxl_ip_adapter(device=device, cache_root=cache_root, num_adapters=num_adapters)

    # Style strength
    if style_refs and len(style_refs) > 0:
        if style_weights and len(style_weights) == len(style_refs):
            pipe.set_ip_adapter_scale(style_weights)
            print(f"[IP-Adapter] multi-ref scales = {style_weights}")
        else:
            # Replicate ip_strength for each adapter
            adapter_scales = [ip_strength] * len(style_refs)
            pipe.set_ip_adapter_scale(adapter_scales)
            print(f"[IP-Adapter] uniform scale = {ip_strength} for {len(style_refs)} adapters")

    fused_prompt = fuse_prompts(user_prompt, core_prompt)
    print(f"[PROMPT] {fused_prompt}")
    print(f"[NEG]    {negative_prompt}")

    # Generate N samples (optionally shift seed each time)
    for i in range(samples):
        cur_seed = seed + i * int(per_prompt_seed_shift)
        g = torch.Generator(device=device).manual_seed(cur_seed)

        kwargs = dict(
            prompt=fused_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=g,
        )
        if style_refs and len(style_refs) > 0:
            kwargs["ip_adapter_image"] = style_refs

        img = pipe(**kwargs).images[0]
        save_path = os.path.join(out_dir, f"human_{i:02d}.png")
        img.save(save_path)
        print(f"[OK] saved: {save_path} (seed={cur_seed})")

    # Save prompt log
    with open(os.path.join(out_dir, "prompt_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"[CORE]\n{core_prompt}\n\n[USER]\n{user_prompt}\n\n[FUSED]\n{fused_prompt}\n\n[NEG]\n{negative_prompt}\n")
    print(f"[OK] logs saved in: {out_dir}")


ASSET_DIR = "/workspace/dhe_project/assets"
DEFAULT_ASSETS = f"{ASSET_DIR}/chair_01.png,{ASSET_DIR}/pen_01.png,{ASSET_DIR}/ref_style_02.png"
assert os.path.exists(ASSET_DIR), f"Asset directory not found: {ASSET_DIR}"
assert all(os.path.exists(p) for p in DEFAULT_ASSETS.split(",")), f"Default assets not found: {DEFAULT_ASSETS}"


# -----------------------------
# CLI
# -----------------------------
funiture = "chair"  # 예시: "chair", "sofa", "pen"
personality = "bratty"  # 예시: "tsundere", "polite_junior", "energetic", "onee_san", "yandere", "bratty"
emotion= "neutral"  # 예시: "neutral", "smile", "happy", "sad", "angry", "surprised", "shy", "confused", "sleepy", "panic"

# ref path format : /workspace/dhe_project/assets/emotion/pen/energetic/neutral_face.png
 
def main():
    ap = argparse.ArgumentParser(description="Human generator with SDXL + IP-Adapter style transfer for game-compatible semi-realistic full-body characters")
    # I/O
    ap.add_argument("--out", type=str, default=f"/workspace/dhe_project/assets/humans/{funiture}/{personality}", help="output directory")
    ap.add_argument("--cache_root", type=str, default="/data/huggingface", help="HF/Diffusers cache root (checkpoint rule)")
    # Prompts
    ap.add_argument("--user_prompt", type=str, default="", help="User input (English). E.g., 'a person in a blue jacket, walking pose'")
    ap.add_argument("--core_prompt", type=str, default=DEFAULT_CORE_PROMPT, help="Core style prompt (always applied)")
    ap.add_argument("--neg", type=str, default=DEFAULT_NEG, help="negative prompt")
    # Style refs
    ap.add_argument("--style_refs", type=str, default=f'/workspace/dhe_project/generation/save/{funiture}_{personality}_face/generated_face.png',
                    help="Comma-separated style reference images. E.g., /mnt/data/pen_01.png,/mnt/data/chair_01.png")
    ap.add_argument("--style_weights", type=str, default="0.3,0.3,0.3", help="Comma-separated weights; matches refs count for multi-scale")
    ap.add_argument("--ip_strength", type=float, default=0.25, help="Style strength for uniform scaling (0.6~0.85 recommended)")
    # Image params
    ap.add_argument("--width", type=int, default=1024, help="Image width (SDXL default is 1024)")
    ap.add_argument("--height", type=int, default=1024, help="Image height (SDXL default is 1024)")
    # Sampling
    ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--cfg", type=float, default=7.5, help="Classifier-free guidance scale (CFG)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--samples", type=int, default=1, help="Number of images to generate")
    ap.add_argument("--seed_shift", type=int, default=0, help="Seed increment per sample for variations")
    # Device
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    refs = parse_csv_paths(args.style_refs)
    ref_imgs = load_style_refs(refs, size=512) if refs else None
    weights = parse_csv_floats(args.style_weights)

    generate(
        out_dir=args.out,
        user_prompt=args.user_prompt,
        core_prompt=args.core_prompt,
        negative_prompt=args.neg,
        style_refs=ref_imgs,
        style_weights=weights,
        ip_strength=args.ip_strength,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.cfg,
        seed=args.seed,
        samples=args.samples,
        per_prompt_seed_shift=args.seed_shift,
        device=device,
        cache_root=args.cache_root,
    )


if __name__ == "__main__":
    main()