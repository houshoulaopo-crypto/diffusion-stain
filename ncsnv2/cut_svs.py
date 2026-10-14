import glob

import numpy as np
import openslide
from PIL import Image
import os

# 输入目录（里面有 SLE_001, SLE_002... 每个里面有 .svs）
svs_root = "/home/yubai402b/MZJ/Diffusion_stain/data/SLE_svs/"

# 输出 patch 保存目录（所有 patch 放在同一个目录）
out_dir = "/home/yubai402b/MZJ/Diffusion_stain/Trichrome"
os.makedirs(out_dir, exist_ok=True)

# 用 os.walk 找所有 .svs 文件
svs_files = glob.glob(os.path.join(svs_root, "**", "*Masson.svs"), recursive=True)

print(f"Found {len(svs_files)} svs files.")

patch_size = 512
level = 0
step = patch_size  # or patch_size // 2 for overlap

patch_id = 0
for svs_path in svs_files:
    slide = openslide.OpenSlide(svs_path)
    w, h = slide.level_dimensions[level]
    print(f"Processing {svs_path}, size={w}x{h}")

    svs_name = os.path.splitext(os.path.basename(svs_path))[0]

    for y in range(0, h - patch_size, step):
        for x in range(0, w - patch_size, step):
            patch = slide.read_region((x, y), level, (patch_size, patch_size))
            patch = patch.convert("RGB")

            patch_np = np.array(patch)

            # --- Tissue detection for H&E ---
            # White background → high mean intensity
            mean_intensity = np.mean(patch_np)
            # Tissue is usually less white → variance is higher
            std_intensity = np.std(patch_np)

            # keep if not too white and has enough texture
            if mean_intensity < 190 and std_intensity > 15:
                # 文件名加上 svs 名称防止重名
                patch.save(os.path.join(out_dir, f"{svs_name}_patch_{patch_id}.png"))
                patch_id += 1

print(f"🎉 Saved {patch_id} patches in {out_dir}")
