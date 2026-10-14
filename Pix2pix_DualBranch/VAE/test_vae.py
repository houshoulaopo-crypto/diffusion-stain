import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from VAE.vae import VAE

# ===== CONFIG =====
model_path = "/home/yubai402b/MZJ/Diffusion_stain/HEMIT/Pix2pix_DualBranch/vae_checkpoints/vae_best.pth"
input_dir = "/home/yubai402b/MZJ/Diffusion_stain/HEMIT/HEMIT H&E to Multiplex-immunohistochemistry Image Translation with Dual-Branch Pix2pix Generator/HEMIT/testA"
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== IMAGE TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((512, 512)),   # reduce memory (1024 → 512)
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===== LOAD MODEL =====
model = VAE()
model.load_state_dict(torch.load(model_path, map_location=device)['model_state'])
model.to(device)
model.eval()
model.eval()
model.to(device)

# ===== TEST LOOP =====
image_list = os.listdir(input_dir)

with torch.no_grad():
    for img_name in tqdm(image_list):

        img_path = os.path.join(input_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        # inference
        output,_,_ = model(img)
        # output = (output - output.min()) / (output.max() - output.min() + 1e-8)

        save_path = os.path.join(output_dir, img_name)

        transforms.ToPILImage()(output.squeeze().cpu()).save(save_path)

print("Testing finished. Results saved to:", output_dir)