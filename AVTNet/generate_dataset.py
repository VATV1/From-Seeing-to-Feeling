# Import necessary libraries for image processing and dataset creation
from __future__ import print_function
from PIL import Image
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset

# Dataset class for images with labels
class TouchFolderLabel(Dataset):
    """Folder datasets which returns the index of the image as well."""

    def __init__(self, root, transform=None, target_transform=None, two_crop=False,
                 mode='train', label='hard'):
        # Initialize parameters
        self.two_crop = two_crop
        self.dataroot = Path(root)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.label = label

        # Construct path to the data file based on the mode and label
        if label == 'rough' and mode in ['train', 'test']:
            data_file = os.path.join(root, f'{mode}_rough.txt')
        else:
            data_file = os.path.join(root, f'{mode}.txt')

        # Read file lines into env list
        with open(data_file, 'r') as f:
            self.env = [line.strip() for line in f.readlines()]
        # Set dataset length
        self.length = len(self.env)

    def __getitem__(self, index):
        # Return the image, its label, and index based on the given index
        assert index < self.length, 'index_A range error'

        raw, target = self.env[index].split(',')
        target = int(target)

        if self.label == 'hard' and target in [7, 8, 9, 11, 13]:
            target = 1
        elif self.label == 'hard':
            target = 0

        # Construct paths for image and gelsight
        idx = Path(raw).name
        dir_path = self.dataroot / raw[:16]
        A_img_path = dir_path / 'video_frame' / idx
        A_gelsight_path = dir_path / 'gelsight_frame' / idx

        # Load image and gelsight
        A_img= Image.open(A_img_path).convert('RGB')
        A_gel= Image.open(A_gelsight_path).convert('RGB')

        if self.transform:
            A_img_q, A_img_k = self.transform(A_img)
            A_gel_q, A_gel_k = self.transform(A_gel)

        # out = torch.cat((A_img, A_gel), dim=0)

        # Return image and label
        return A_img_q, A_img_k, A_gel_q, A_gel_k, target

    def __len__(self):
        """Return the total number of images."""
        return self.length


class CalandraLabel(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        # Initialize dataset parameters
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.subset = mode  # 直接使用传入的 mode ('train' 或 'test')
        self.samples = []

        # Define the root path based on mode
        # 假设你的目录结构是 root_dir/train/rgb 和 root_dir/train/tactile
        subset_path = self.root_dir / self.subset
        rgb_dir = subset_path / 'rgb'
        tactile_dir = subset_path / 'tactile'


        # Loop over each file in the category and add it to the samples list
        # 这里遍历 RGB 文件夹
        for img_file in rgb_dir.iterdir():
            if img_file.suffix in [".png"]:
                # --- 1. 解析文件名获取标签 ---
                # 示例文件名: obj0_3_0_rgb_00017 (注意：如果是遍历rgb文件夹，中间可能是rgb)
                # 分割结果: ['obj0', '3', '0', 'rgb', '00017']
                parts = img_file.stem.split('_')

                try:
                    # 根据你的说明：3是softness，0是position
                    # 它们分别位于分割后的索引 1 和 2
                    softness = int(parts[3])-1
                    position = int(parts[4])
                except (IndexError, ValueError):
                    print(f"Skipping file with bad format: {img_file.name}")
                    continue

                # --- 2. 构建配对路径 ---
                rgb_path = img_file

                # 关键：需要找到对应的 tactile 文件
                # 如果 rgb 文件名是 ..._rgb_...，需要替换为 ..._raw_... 才能在 tactile 文件夹找到
                # 假设 tactile 文件名为 obj0_3_0_raw_00017.png
                tactile_name = img_file.name.replace("_rgb_", "_tactile_")
                tactile_path = tactile_dir / tactile_name

                # --- 3. 检查文件是否存在并存储 ---
                if tactile_path.exists():
                    # 将路径和解析好的标签一起存入 samples
                    self.samples.append((rgb_path, tactile_path, softness, position))
                else:
                    # 如果找不到对应的 tactile 图片，可以选择打印警告
                    # print(f"Warning: Tactile image not found for {img_file.name}")
                    pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return images and labels based on the given index
        # --- 4. 解包数据 (包含标签) ---
        rgb_path, tactile_path, softness, position = self.samples[idx]

        # Load images
        rgb_image = Image.open(rgb_path).convert("RGB")
        tactile_image = Image.open(tactile_path).convert("RGB")

        # Apply transforms
        # 注意：这里假设 transform 返回两个视图 (q, k)，如果是标准 transform 可能只返回一个 tensor
        if self.transform:
            rgb_image_q, rgb_image_k = self.transform(rgb_image)
            tactile_image_q, tactile_image_k = self.transform(tactile_image)
        else:
            # 如果没有 transform，至少要转成 tensor，否则 DataLoader 会报错
            # 这里仅作示例，实际使用请确保传入了 transform
            pass

            # --- 5. 将标签转换为 Tensor ---
        # 你的原始代码中 torch.tensor(, ...) 是空的，这里补上了变量
        label_position = torch.tensor(position, dtype=torch.long)
        label_softness = torch.tensor(softness, dtype=torch.long)

        # Return processed images and label
        return rgb_image_q, rgb_image_k, tactile_image_q, tactile_image_k, label_position, label_softness

