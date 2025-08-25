# -*- coding: utf-8 -*-
"""
Bottom-weighted Squash & Stretch (with background removal & export options)

기능
- 배경 제거(코너 평균색 기반) 후 알파 합성
- img_size 사용(기본 512) 또는 원본 크기 유지 옵션
- 스쿼시&스트레치: 아래쪽(다리) 가중 + 포물선형 파형
- GIF로 저장 / 모든 프레임 PNG로 저장 / 둘 다

사용 예)
export_squash_stretch(
    img_path="pen.png", out_dir="./out",
    save_gif=True, save_png_frames=True,
    keep_original_size=False, img_size=512
)
"""

from typing import Literal, Optional, Tuple, Dict, List
import os
import numpy as np
from PIL import Image, ImageFilter

# -----------------------------
# Util: 보장된 디렉토리 생성
# -----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# -----------------------------
# 배경 제거
# -----------------------------
def remove_bg_from_image(
    img: Image.Image,
    dist_thresh: int = 26,      # 배경과의 색 거리 임계값(오프화이트 배경 기준)
    feather: float = 1.2,       # 가장자리 부드럽게
    grow: int = 2,              # 마스크 팽창 횟수
    shrink: int = 1             # 마스크 수축 횟수
) -> Image.Image:
    """
    코너 색 평균으로 배경색을 추정하고, 유클리드 RGB 거리로 마스크 생성.
    반환: 투명 배경의 RGBA 이미지
    """
    im = img.convert("RGBA")
    W, H = im.size
    arr = np.array(im)
    rgb = arr[..., :3].astype(np.float32)

    # 20x20 코너 패치 평균으로 배경색 추정
    k = max(10, min(W, H) // 20)
    patches = [
        rgb[0:k, 0:k], rgb[0:k, W-k:W],
        rgb[H-k:H, 0:k], rgb[H-k:H, W-k:W]
    ]
    bg = np.mean([p.reshape(-1,3).mean(axis=0) for p in patches], axis=0)

    dist = np.linalg.norm(rgb - bg[None, None, :], axis=-1)
    mask = (dist > float(dist_thresh)).astype(np.uint8) * 255
    m = Image.fromarray(mask, mode="L")

    # morphology: dilation -> erosion
    for _ in range(int(grow)):
        m = m.filter(ImageFilter.MaxFilter(3))
    for _ in range(int(shrink)):
        m = m.filter(ImageFilter.MinFilter(3))

    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(feather)))

    out = im.copy()
    out.putalpha(m)
    return out

# -----------------------------
# 리사이즈 (원본 유지 or img_size로 캔버스 맞추기)
# -----------------------------
def resize_sprite(
    img: Image.Image,
    keep_original_size: bool = False,
    img_size: int = 512,
    pad_to_square: bool = True
) -> Image.Image:
    """
    keep_original_size=True면 원본 크기 유지.
    아니면, 긴 변을 img_size로 맞추고(종횡비 유지), pad_to_square면 투명 배경으로 정사각 패딩.
    """
    if keep_original_size:
        return img.copy()

    W, H = img.size
    print(f"Resizing image from {W}x{H} to {img_size}x{img_size}...")
    if max(W, H) == 0:
        return img.copy()

    scale = float(img_size) / float(max(W, H))
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if not pad_to_square:
        return img_resized

    canvas = Image.new("RGBA", (img_size, img_size), (0,0,0,0))
    off_x = (img_size - new_w) // 2
    off_y = (img_size - new_h) // 2
    canvas.paste(img_resized, (off_x, off_y), img_resized)
    return canvas

# -----------------------------
# 파형 & 가중치
# -----------------------------
def parabola_wave(phi: float) -> float:
    """포물선형 파형 in [-1,1] (피크에서 기울기 0)"""
    return 2.0 * (1.0 - (2.0 * phi - 1.0) ** 2) - 1.0

def sine_wave(phi: float) -> float:
    """사인파 파형 in [-1,1]"""
    from math import sin, pi
    return float(sin(2.0 * pi * phi))

def spectral_weight(y01: np.ndarray, power: float = 2.2) -> np.ndarray:
    """아래쪽(1)에 더 큰 가중"""
    return np.clip(y01, 0.0, 1.0) ** float(power)

# -----------------------------
# 핵심 워프: 스쿼시&스트레치
# -----------------------------
def squash_stretch(
    img: Image.Image,
    *,
    amp: float = 0.18,
    phase: float = 0.0,
    power: float = 2.2,
    slices: int = 120,
    wave: Literal["parabola","sine"] = "parabola",
) -> Image.Image:
    """
    아래쪽 가중 + 주기 파형으로 세로 비선형 워프. 전체 높이는 보존.
    """
    if not isinstance(img, Image.Image):
        raise TypeError("img must be PIL.Image")
    W, H = img.size
    if W == 0 or H == 0 or slices < 2:
        return img.copy()

    ys = np.linspace(0.0, float(H), int(slices)+1, dtype=np.float64)
    ymid = 0.5*(ys[:-1] + ys[1:])
    y01  = ymid / float(H)
    w    = spectral_weight(y01, power=power)
    s    = parabola_wave(phase) if wave == "parabola" else sine_wave(phase)

    sy = 1.0 + (float(amp) * float(s) * w)

    # 높이 보존 정규화
    hseg  = (ys[1:] - ys[:-1])
    total = float(np.sum(sy * hseg))
    if total <= 1e-6:
        return img.copy()
    sy *= float(H) / total

    # 누적 목적 y 좌표
    ydst = np.zeros_like(ys)
    for i in range(int(slices)):
        ydst[i+1] = ydst[i] + sy[i] * (ys[i+1] - ys[i])

    # 정수, 단조 증가 보정
    ydst_int = [0]
    for i in range(1, int(slices)+1):
        nxt = int(round(float(ydst[i])))
        if nxt <= ydst_int[-1]:
            nxt = ydst_int[-1] + 1
        ydst_int.append(min(nxt, H))
    ydst_int[-1] = H

    # PIL MESH 구성 (dst_bbox, src_quad(UL,LL,LR,UR))
    mesh = []
    for i in range(int(slices)):
        top, bot = int(ydst_int[i]), int(ydst_int[i+1])
        if bot <= top:
            continue
        dst_bbox = (0, top, W, bot)
        src_quad = (
            0.0, float(ys[i]),         # UL
            0.0, float(ys[i+1]),       # LL
            float(W), float(ys[i+1]),  # LR
            float(W), float(ys[i])     # UR
        )
        mesh.append((dst_bbox, src_quad))

    return img.transform(
        (W, H),
        Image.Transform.MESH,
        mesh,
        resample=Image.Resampling.BICUBIC
    )

# -----------------------------
# 오케스트레이터: 배경 제거 + 리사이즈 + 내보내기
# -----------------------------
def export_squash_stretch(
    img_path: str,
    out_dir: str,
    *,
    # 배경 제거
    remove_background: bool = True,
    bg_dist_thresh: int = 26,
    bg_feather: float = 1.2,
    bg_grow: int = 2,
    bg_shrink: int = 1,
    # 리사이즈
    keep_original_size: bool = False,
    img_size: int = 512,
    pad_to_square: bool = True,
    # 애니 파라미터
    frames: int = 36,
    amp: float = 0.18,
    power: float = 2.2,
    slices: int = 120,
    wave: Literal["parabola","sine"] = "parabola",
    # 저장 옵션
    save_gif: bool = True,
    save_png_frames: bool = False,
    gif_name: str = "gait.gif",
    png_prefix: str = "frame_",
    fps: int = 16
) -> Dict[str, List[str]]:
    """
    반환: {'gif': [path], 'png_frames': [paths], 'preprocessed': [path]}
    """
    _ensure_dir(out_dir)
    img = Image.open(img_path).convert("RGBA")

    # 1) 배경 제거
    if remove_background:
        img = remove_bg_from_image(
            img,
            dist_thresh=bg_dist_thresh,
            feather=bg_feather,
            grow=bg_grow,
            shrink=bg_shrink
        )

    # 2) 리사이즈
    img = resize_sprite(
        img,
        keep_original_size=keep_original_size,
        img_size=img_size,
        pad_to_square=pad_to_square
    )

    # 프리프로세스 결과 저장(참고용)
    pre_path = os.path.join(out_dir, "preprocessed.png")
    img.save(pre_path)

    # 3) 프레임 생성
    seq = []
    png_paths: List[str] = []
    for f in range(int(frames)):
        phi = (f % frames) / float(frames)
        frame = squash_stretch(
            img, amp=amp, phase=phi, power=power, slices=slices, wave=wave
        )
        seq.append(frame)
        if save_png_frames:
            p = os.path.join(out_dir, f"{png_prefix}{f:03d}.png")
            frame.save(p)
            png_paths.append(p)

    # 4) GIF 저장
    gif_paths: List[str] = []
    if save_gif and len(seq) > 0:
        gif_path = os.path.join(out_dir, gif_name)
        duration_ms = max(1, int(round(1000.0 / float(max(1, fps)))))
        seq[0].save(
            gif_path,
            save_all=True,
            append_images=seq[1:],
            duration=duration_ms,
            loop=0,
            disposal=2
        )
        gif_paths.append(gif_path)

    return {"gif": gif_paths, "png_frames": png_paths, "preprocessed": [pre_path]}


# -----------------------------
# 예시 실행
# -----------------------------
if __name__ == "__main__":
    funiture = "pen"  # 예시: "chair", "sofa", "pen"
    personality = "bratty"  # 예시: "tsundere", "polite_junior", "energetic", "onee_san", "yandere", "bratty"
    # neutral, smile, happy, sad, angry, surprised, shy, confused, sleepy, panic

    for emotion in ["neutral", "smile", "happy", "sad", "angry", "surprised", "shy", "confused", "sleepy", "panic"]:
        print(f"Generating {emotion} face GIF...")
        paths = export_squash_stretch(
            img_path=f"/workspace/dhe_project/assets/emotion/{funiture}/{personality}/{emotion}_face.png",
            out_dir=f"/workspace/dhe_project/out_gif/{funiture}/{personality}/{emotion}",
            # 저장 모드
            save_gif=True,
            save_png_frames=True,
            # 크기 결정
            keep_original_size=False,   # True면 원본 해상도 사용
            img_size=1024,               # False일 때 기준(기본 512)
            # 배경 제거
            remove_background=True,
            # 애니 파라미터
            frames=36, amp=0.2, power=2.2, slices=120, wave="parabola",
            fps=16
        )
