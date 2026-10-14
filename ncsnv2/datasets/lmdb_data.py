import os
import lmdb
from PIL import Image
import io

patch_dir = "/home/yubai402b/MZJ/Diffusion_stain/ncsnv2/patches"
lmdb_out = "/home/yubai402b/MZJ/Diffusion_stain/ncsnv2/lmdb_data"
os.makedirs(lmdb_out, exist_ok=True)

# Estimate LMDB map size
total_bytes = sum(os.path.getsize(os.path.join(patch_dir, f))
                  for f in os.listdir(patch_dir) if f.endswith(".png"))
map_size = int(total_bytes * 10)   # double size for safety
print(f"Estimated map_size = {map_size/1e9:.2f} GB")

env = lmdb.open(lmdb_out, map_size=map_size)

with env.begin(write=True) as txn:
    idx = 0
    for fname in os.listdir(patch_dir):
        if not fname.endswith(".png"):
            continue
        fpath = os.path.join(patch_dir, fname)

        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as e:
            print(f"⚠️ Skipping corrupted image {fname}: {e}")
            continue

        buf = io.BytesIO()
        img.save(buf, format="png")

        key = f"{idx:08}".encode("ascii")
        txn.put(key, buf.getvalue())
        idx += 1

    # save length once
    txn.put(b"length", str(idx).encode("ascii"))

env.close()
print(f"✅ LMDB built at {lmdb_out} with {idx} valid patches")
