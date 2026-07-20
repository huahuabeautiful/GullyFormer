import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class VOCDatasetFromTxt(Dataset):
    def __init__(self, voc_root, split='train', img_size=(256, 256), img_ext='.jpg', mask_ext='.png'):
        """
        voc_root: VOC2007 文件夹的根目录
        split: 'train', 'val', 'test', 或 'trainval'
        img_ext: 原图的后缀名 (如 '.jpg', '.tif', '.dat')
        mask_ext: 掩码图的后缀名 (如 '.png')
        """
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.mask_dir = os.path.join(voc_root, 'SegmentationClass')

        # 指向存放划分名单的 txt 文件
        split_file = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{split}.txt')

        # 读取 txt 文件，获取所有文件名（不包含后缀）
        with open(split_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]

        self.img_size = img_size
        self.img_ext = img_ext
        self.mask_ext = mask_ext

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        base_name = self.file_names[idx]

        # 🌟 修复点 1：智慧兼容 .jpg 和 .png 后缀，彻底解决难负样本混合带来的 FileNotFoundError 隐患
        img_path = os.path.join(self.image_dir, base_name + self.img_ext)
        if not os.path.exists(img_path):
            alternative_ext = '.png' if self.img_ext.lower() == '.jpg' else '.jpg'
            img_path = os.path.join(self.image_dir, base_name + alternative_ext)

        mask_path = os.path.join(self.mask_dir, base_name + self.mask_ext)

        # 读取图像并转换模式
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 缩放处理
        resize = transforms.Resize(self.img_size)
        image = resize(image)
        mask = resize(mask)

        # 转换为 Tensor (原图会自动归一化到 0~1)
        image_tensor = transforms.ToTensor()(image)
        mask_tensor = transforms.ToTensor()(mask)

        # 侵蚀沟掩码二值化处理 (大于0的作为目标类1，其余为0)
        mask_tensor = (mask_tensor > 0).float()

        return image_tensor, mask_tensor


# 🌟 核心修改点：在函数末尾增加了 train_split='train' 接收参数
def get_dataloaders(voc_root, batch_size=8, img_size=(256, 256), img_ext='.jpg', mask_ext='.png', train_split='train'):
    """
    根据 VOC 目录结构和 txt 文件创建训练集和验证集的 DataLoader
    """
    # 🌟 核心修改点：将原本写死的 split='train' 修改为接收外部传参的 split=train_split
    train_dataset = VOCDatasetFromTxt(voc_root, split=train_split, img_size=img_size, img_ext=img_ext, mask_ext=mask_ext)
    val_dataset = VOCDatasetFromTxt(voc_root, split='val', img_size=img_size, img_ext=img_ext, mask_ext=mask_ext)

    print(f"成功加载数据集: 训练集包含 {len(train_dataset)} 张, 验证集包含 {len(val_dataset)} 张。")

    # ℹ️ 这里的 shuffle=True 保持不变，可以完美打乱手动追加在 txt 末尾的难负样本
    # 🌟 修复点 2：加上 drop_last=True 核心参数，强制扔掉最后可能剩余的单张图片，彻底解决 BatchNorm 报错崩溃
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader