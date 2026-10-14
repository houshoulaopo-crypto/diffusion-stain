import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class PNGFolderDataset(Dataset):
    def __init__(self, config, transform=None):
        if config.sampling.data_init:
            root_dir = os.path.join(config.data.sample_dir)
        else :
            root_dir = os.path.join(config.data.data_dir)
        self.transform = transform

        # Collect all .png files recursively
        self.files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(".png"):
                    self.files.append(os.path.join(dirpath, f))

        if len(self.files) == 0:
            raise RuntimeError(f"No .png files found in {root_dir}")

        print(f"✅ Found {len(self.files)} PNG files in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, 0   # dummy label (you can replace with real labels)


# Example usage
if __name__ == "__main__":
    data_root = "/home/yubai402b/MZJ/Diffusion_stain/ncsnv2/test_stain"

    transform = transforms.Compose([
        transforms.Resize(256),        # resize to 256x256
        transforms.CenterCrop(256),
        transforms.ToTensor()          # convert to [0,1] tensor
    ])

    dataset = PNGFolderDataset(data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    for batch in dataloader:
        imgs, labels = batch
        print(imgs)   # [16, 3, 256, 256]
        break
