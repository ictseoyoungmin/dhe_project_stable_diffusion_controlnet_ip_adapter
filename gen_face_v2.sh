#  "tsundere", "polite_junior", "energetic", "onee_san", "yandere", "bratty"

# ---------- Paths & inputs ----------
chair="tissue"   # 첫 인자로 chair/sofa/pen 등 넘길 수 있음
personality="tsundere"  # tsundere, polite_junior, energetic, onee_san, yandere, bratty
IMG="/workspace/dhe_project/assets/${chair}_01.png"
#
IMG="/workspace/dhe_project/assets/obj/tissue/object_00.png"  # tissue용
#
RIG="/workspace/dhe_project/generation/out_${chair}/rig.json"
#
RIG="/workspace/dhe_project/assets/obj/tissue/object_00/rig.json"  # tissue용
#
OUT="test/${chair}_${personality}_face"
# ---------- Switches ----------
USE_SHORT_PROMPT=true      # true 로 바꾸면 77토큰 내 압축 프롬프트 사용
USE_TRIANGULATE="none"      # none | mouth | all  (랜드마크 들로네 삼각분할)

# ---------- Prompts ----------
LONG_PROMPT="kawaii anime style, flat pastel shading, clean vector-style outline, clean lineart with no noise, soft glossy highlights, poketmonster, cohesive mobile game asset style"

# 77 토큰 안쪽으로 압축한 버전 (동일한 의미 유지)
SHORT_PROMPT="kawaii anime style, flat pastel shading, clean vector-style outline, clean lineart with no noise, soft glossy highlights, poketmonster, cohesive mobile game asset style"

NEG="realistic, photorealistic, complex shading, closed eyes, multiple eyes,  multiple mouth, deformed pupils, blurry, extra eyes, blurry eyes, inconsistent lighting, multiple mouth"

if $USE_SHORT_PROMPT; then
  PROMPT="$SHORT_PROMPT"
else
  PROMPT="$LONG_PROMPT"
fi
# ---------- Sampler/controls  ----------
STEPS=30
CFG=45.0
STRENGTH=0.99
CANNY=0.3
SCRIBBLE=0.60
FEATHER=0
SKETCH_WIDTH=2
MASK_SOURCE="sketch"
MASK_EXPAND=8

# ---------- Run ----------
CUDA_VISIBLE_DEVICES=1 python3 gen_face_v2.py \
  --personality "$personality" \
  --image "$IMG" \
  --rig "$RIG" \
  --out "$OUT" \
  --mask_source "$MASK_SOURCE" \
  --mask_expand "$MASK_EXPAND" \
  --steps "$STEPS" --cfg "$CFG" --strength "$STRENGTH" \
  --canny "$CANNY" --scribble "$SCRIBBLE" \
  --feather "$FEATHER" \
  --sketch_width "$SKETCH_WIDTH" \
  --prompt "$PROMPT" \
  --neg "$NEG" \
  --seed 123 \
  --triangulate "$USE_TRIANGULATE"
