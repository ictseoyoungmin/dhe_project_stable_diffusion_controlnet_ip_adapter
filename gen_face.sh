
# 표정 변화 실험 중
# 기본 얼굴 생성 폴더의 face 사진과 asset 생성 폴더의 rig.json사용
chair=tissue
CUDA_VISIBLE_DEVICES=0 python3 gen_face.py \
  --image /workspace/dhe_project/assets/obj/${chair}/object_00.png \
  --rig /workspace/dhe_project/assets/obj/${chair}/object_00/rig.json \
  --out test_face_2 \
  --mask_source sketch \
  --mask_expand 8 \
  --steps 30 --cfg 15.0 --strength 0.90 \
  --canny 0.4 --scribble 0.7 \
  --feather 21 \
  --sketch_width 1 \
  --prompt "kawaii chibi sticker face on the furniture surface, visible pupils and iris details, simple nose, expressionless tight lips, clean vector-style outline, consistent with furniture texture and lighting"  \
  --neg "realistic, photorealistic, complex shading, closed eyes, multiple eyes, deformed pupils, blurry, text" \
  --seed 123

# 기본 얼굴 부착 
# 첫 사용시 이 코드를 사용
# chair=washer
# CUDA_VISIBLE_DEVICES=0 python3 gen_face.py \
#   --image /workspace/dhe_project/assets/${chair}_01.png \
#   --rig /workspace/dhe_project/generation/out_${chair}/rig.json \
#   --out test_face_2 \
#   --mask_source sketch \
#   --mask_expand 8 \
#   --steps 30 --cfg 40.0 --strength 0.95 \
#   --canny 0.3 --scribble 0.7 \
#   --feather 2 \
#   --sketch_width 1 \
#   --prompt "kawaii chibi sticker face on the furniture surface, deeply sad facial expression, large expressive eyes with visible pupils and detailed iris reflecting subtle sorrow, downturned eyebrows, simple nose, tightly closed lips with a slight frown, clean vector-style outline, consistent with furniture texture and lighting, soft melancholic vibe"  \
#   --neg "realistic, photorealistic, complex shading, closed eyes, multiple eyes, deformed pupils, blurry, text, overly exaggerated expressions, neutral expression, happy expression, angry expression, extra eyes, blurry eyes, inconsistent lighting" \
#   --seed 123

# backup
# chair=chair
# python3 gen_face.py \
#   --image /workspace/dhe_project/assets/${chair}_01.png \
#   --rig /workspace/dhe_project/generation/out_${chair}/rig.json \
#   --out test_face \
#   --mask_source sketch \
#   --mask_expand 8 \
#   --steps 30 --cfg 15.0 --strength 0.90 \
#   --canny 0.4 --scribble 0.7 \
#   --feather 21 \
#   --sketch_width 1 \
#   --prompt "kawaii chibi sticker face on the furniture surface, visible pupils and iris details, simple nose, expressionless tight lips, clean vector-style outline, consistent with furniture texture and lighting"  \
#   --neg "realistic, photorealistic, complex shading, closed eyes, multiple eyes, deformed pupils, blurry, text" \
#   --seed 123