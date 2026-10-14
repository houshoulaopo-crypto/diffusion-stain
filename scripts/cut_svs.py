import numpy as np
import openslide
from PIL import Image
import os

# 输入文件路径
svs_path = "/home/yubai402b/MZJ/Diffusion_stain/SLE_svs/SLE_0001/SLE_0001_HE.svs"

# 输出 patch 保存目录
out_dir = "./patches"
os.makedirs(out_dir, exist_ok=True)

# 打开 svs 文件
slide = openslide.OpenSlide(svs_path)

# 设定 patch 大小，比如 512x512
patch_size = 512

# 选择读取的层级（0 是最高分辨率）
level = 0
w, h = slide.level_dimensions[level]
print(f"WSI size: {w} x {h}")

# 按步长裁剪
step = patch_size  # 不重叠
# step = patch_size // 2  # 50% 重叠

patch_id = 0
for y in range(0, h - patch_size, step):
    for x in range(0, w - patch_size, step):
        patch = slide.read_region((x, y), level, (patch_size, patch_size))  # PIL Image, RGBA
        patch = patch.convert("RGB")  # 去掉 alpha 通道

        patch_np = np.array(patch)
        white_fraction = np.mean(np.all(patch_np >= 245, axis=2))

        # Only save if less than 90% white
        if white_fraction < 0.3:
            patch.save(os.path.join(out_dir, f"patch_{patch_id}.png"))
            patch_id += 1

print(f"Saved {patch_id} patches in {out_dir}")
