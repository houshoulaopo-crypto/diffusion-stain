import os
import glob
from datasets import utils
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class MRI_Dataset(Dataset):
    def __init__(self, dir, image_size=(384,384), transform=utils.resize_tensor):
        """
        data_dir:  存放 k-space 数据的文件夹
        maps_dir:  存放 sensitivity maps 的文件夹
        """
        self.data_dir = dir + 'brain_T2'
        self.maps_dir = dir + 'brain_T2_maps'
        self.data_files = sorted(glob.glob(os.path.join(self.data_dir, "*.h5")))
        self.maps_files = sorted(glob.glob(os.path.join(self.maps_dir, "*.h5")))
        self.image_size = image_size
        self.transform = transform

        assert len(self.data_files) == len(self.maps_files), \
            "data_dir 和 maps_dir 中文件数不一致！"

        # 保存每个文件的切片数量，方便索引
        self.slice_index = []  # [(file_idx, slice_idx), ...]
        for file_idx, data_file in enumerate(self.data_files):
            with h5py.File(data_file, 'r') as f:
                num_slices = f['kspace'].shape[0]
                for slice_idx in range(num_slices):
                    self.slice_index.append((file_idx, slice_idx))

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_index[idx]

        data_file = self.data_files[file_idx]
        maps_file = self.maps_files[file_idx]

        # 读取 k-space
        with h5py.File(data_file, 'r') as f:
            kspace = np.array(f['kspace'][slice_idx])  # (C,H,W)

        # 读取 sensitivity maps
        with h5py.File(maps_file, 'r') as f:
            maps = np.array(f['s_maps'][slice_idx])  # (C,H,W)

        # 计算 MVUE 图像 (复数)
        mvue = utils.get_mvue(kspace, maps)

        # 转成 (2,H,W)，实部+虚部
        sample = utils.view_as_real(mvue)

        if self.transform:
            sample = self.transform(sample)

        return sample, 0







class MRIDataset(Dataset):
    def __init__(self, root, transform=None):
        # Collect all .h5 files inside the directory
        self.files = sorted(glob.glob(os.path.join(root, "*.h5")))
        self.transform = transform
        if len(self.files) == 0:
            raise RuntimeError(f"No .h5 files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        h5_path = self.files[idx]
        with h5py.File(h5_path, "r") as f:
            print(list(f.keys()))
            kspace = f["s_maps"][:]  # (coils, H, W) or (slices, coils, H, W)
        # Convert complex64 → 2 channels (real, imag)
        kspace_tensor = utils.view_as_real(kspace)
        print(kspace_tensor.shape)

        if self.transform:
            kspace = self.transform(kspace_tensor)

        return kspace
