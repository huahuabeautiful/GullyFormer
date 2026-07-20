import torch
import random
import numpy as np
import os
import datetime

# 屏蔽煩人的軟鏈接警告
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from Data_loader import get_dataloaders
from training_loop import train_model
from Segformer_model import CustomSegFormer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 全局隨機種子已固定為: {seed}")


if __name__ == "__main__":
    # ================== 核心配置區域 ==================
    CURRENT_SEED = 42
    USE_PRETRAINED = True

    # 路徑配置
    VOC_ROOT = r"G:\Segformer2.0\VOCdevkit\VOC2007"
    SEGFORMER_LOCAL_WEIGHTS = r"G:\Segformer2.0\SegFormer"

    # 訓練超參數
    EPOCHS = 30
    BATCH_SIZE = 8
    IMG_SIZE = (256, 256)
    LR = 1e-4

    # 損失函數配置開關
    USE_BCE = True
    USE_EDGE_WEIGHT = False
    USE_DICE = False
    EDGE_PENALTY_VAL = 2.0

    # 難負樣本優化配置開關
    USE_HARD_NEG = False
    NEG_RATIO = "0.2"

    # 🌟🌟🌟 纯粹的结构开关：可以自由打开 (True) 或关闭 (False) 🌟🌟🌟
    # 它只控制网络结构，不会自作聪明地去修改你的文件夹名称
    USE_ASPP = False

    # 🌟🌟🌟 纯手写控制：在这里直接输入你想要的 log 文件夹备注名称 🌟🌟🌟
    # 想怎么命名就怎么命名，程序会完全听你的
    BASE_EXP_NAME = "Smart_MiT_NoASPP_BCE_30ep"
    # ==================================================

    # 數據集路由保持原樣
    if USE_HARD_NEG:
        TRAIN_SPLIT = f"train_N_{NEG_RATIO}"
        print(f"🚀 [難負樣本優化已開啟] 訓練將加載目錄: {TRAIN_SPLIT}.txt")
    else:
        TRAIN_SPLIT = "train"
        print(f"🍃 [常規訓練模式] 訓練將加載常規目錄: train.txt")

    # 🌟 文件夹命名完全遵循你手写的 BASE_EXP_NAME
    EXP_NAME = BASE_EXP_NAME

    set_seed(CURRENT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"準備就緒，使用設備: {device}")

    # 檔案夾命名拼接邏輯：runs/loss_年月日_時分秒_你指定的名稱
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    current_save_dir = os.path.join("runs", f"loss_{time_str}_{EXP_NAME}")

    # --- 初始化數據加載器 ---
    train_loader, val_loader = get_dataloaders(
        voc_root=VOC_ROOT,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        img_ext='.jpg',
        mask_ext='.png',
        train_split=TRAIN_SPLIT
    )

    print(f"🛠️ 正在構建模型骨架 | 當前 HybridASPP 開關狀態: {USE_ASPP}")

    # --- 初始化縫合模型与优化器 ---
    model = CustomSegFormer(
        out_channels=1,
        model_size="b0",
        pretrained_path=SEGFORMER_LOCAL_WEIGHTS if USE_PRETRAINED else None,
        use_aspp=USE_ASPP  # 🌟 直接透传开关给模型
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # --- 開始訓練 ---
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device,
        save_dir=current_save_dir,
        use_bce=USE_BCE,
        use_edge_weight=USE_EDGE_WEIGHT,
        use_dice=USE_DICE,
        edge_weight_val=EDGE_PENALTY_VAL
    )