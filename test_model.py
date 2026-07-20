import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from Segformer_model import CustomSegFormer


# ================= 1. 專用的測試集 Dataloader =================
class VOCDatasetTest(Dataset):
    def __init__(self, voc_root, split='test', img_size=(256, 256), img_ext='.jpg', mask_ext='.png'):
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.mask_dir = os.path.join(voc_root, 'SegmentationClass')
        split_file = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]
        self.img_size = img_size
        self.img_ext = img_ext
        self.mask_ext = mask_ext

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        base_name = self.file_names[idx]
        img_path = os.path.join(self.image_dir, base_name + self.img_ext)
        mask_path = os.path.join(self.mask_dir, base_name + self.mask_ext)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        resize = transforms.Resize(self.img_size)
        image = resize(image)
        mask = resize(mask)

        image_tensor = transforms.ToTensor()(image)
        mask_tensor = (transforms.ToTensor()(mask) > 0).float()
        return image_tensor, mask_tensor, base_name


# ================= 2. 柱狀圖繪製函數 =================
def plot_metric_bar(metric_name, bg_val, gully_val, mean_val, save_dir):
    plt.figure(figsize=(8, 6))
    classes = ['_background_', 'gully']
    values = [bg_val, gully_val]
    bars = plt.barh(classes, values, color='royalblue')
    plt.title(f'm{metric_name} = {mean_val * 100:.2f}%', fontsize=16)
    plt.xlabel(metric_name, fontsize=14)
    plt.xlim(0, max(values) + 0.15 if max(values) + 0.15 < 1.1 else 1.1)

    for bar, val in zip(bars, values):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}', va='center', ha='left', color='royalblue', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'm{metric_name}_bar.png'), dpi=300)
    plt.close()


# ================= 3. 核心測試與可視化函數 =================
def evaluate_and_visualize():
    # ================== 核心配置區域 ==================
    CHOOSE_MODEL = "MiT"
    USE_PRETRAINED = True

    # 🌟🌟🌟 纯粹的结构开关：必须与你测试的权重所用的网络结构严格对应 🌟🌟🌟
    USE_ASPP = False

    # 路徑配置
    VOC_ROOT = r"G:\Segformer2.0\VOCdevkit\VOC2007"
    SEGFORMER_LOCAL_WEIGHTS = r"G:\Segformer2.0\SegFormer"

    # 🌟 【手動填寫】要加載的權重路徑
    TRAINED_WEIGHT_PATH = r"G:\Segformer2.0\runs\loss_2026_06_08_16_29_23_Smart_MiT_NoASPP_ablation\best_epoch_weights.pth"

    # 🌟🌟🌟 纯手写控制：在這裡指定你測試結果輸出檔案夾的名稱 🌟🌟🌟
    CUSTOM_RESULT_DIR = "Smart_MiT_NoASPP_test_results"

    # 圖像配置
    IMG_SIZE = (256, 256)
    # ==================================================

    # 🌟 完全遵循你手写的名字，不再自动修改
    result_dir = CUSTOM_RESULT_DIR

    vis_dir = os.path.join(result_dir, "visualizations")
    plot_dir = os.path.join(result_dir, "plots")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = VOCDatasetTest(voc_root=VOC_ROOT, split='test', img_size=IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"🛠️ 正在初始化測試模型 | 當前 HybridASPP 開關狀態: {USE_ASPP}")

    model = CustomSegFormer(
        out_channels=1,
        model_size="b0",
        pretrained_path=SEGFORMER_LOCAL_WEIGHTS if USE_PRETRAINED else None,
        use_aspp=USE_ASPP  # 🌟 透传开关给模型
    ).to(device)

    # 加載已訓練好的權重
    try:
        model.load_state_dict(torch.load(TRAINED_WEIGHT_PATH, map_location=device, weights_only=True))
        print("✅ 成功載入您訓練的最新權重！")
    except Exception as e:
        print(f"❌ 權重載入失敗，請檢查 TRAINED_WEIGHT_PATH 是否填寫正確！\n錯誤訊息: {e}")
        return

    model.eval()

    print(f"🚀 開始在 {len(test_dataset)} 張測試圖上進行推理...")

    tp_1, fp_1, fn_1 = 0.0, 0.0, 0.0
    tp_0, fp_0, fn_0 = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, masks, base_names in tqdm(test_loader, desc="Testing & Visualizing"):
            images = images.to(device)
            masks = masks.to(device)
            base_name = base_names[0]

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            p_flat = preds.view(-1)
            t_flat = masks.view(-1)

            tp_1 += (p_flat * t_flat).sum().item()
            fp_1 += (p_flat * (1 - t_flat)).sum().item()
            fn_1 += ((1 - p_flat) * t_flat).sum().item()

            tp_0 += ((1 - p_flat) * (1 - t_flat)).sum().item()
            fp_0 += ((1 - p_flat) * t_flat).sum().item()
            fn_0 += (p_flat * (1 - t_flat)).sum().item()

            pred_np = preds.cpu().numpy().squeeze()
            target_np = masks.cpu().numpy().squeeze()

            rgb_mask = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)
            rgb_mask[(pred_np == 1) & (target_np == 1)] = [255, 255, 255]
            rgb_mask[(pred_np == 1) & (target_np == 0)] = [255, 0, 0]
            rgb_mask[(pred_np == 0) & (target_np == 1)] = [0, 0, 255]

            vis_img = Image.fromarray(rgb_mask)
            vis_img.save(os.path.join(vis_dir, f"{base_name}_vis.png"))

    smooth = 1e-5

    iou_1 = (tp_1 + smooth) / (tp_1 + fp_1 + fn_1 + smooth)
    prec_1 = (tp_1 + smooth) / (tp_1 + fp_1 + smooth)
    rec_1 = (tp_1 + smooth) / (tp_1 + fn_1 + smooth)
    pa_1 = rec_1
    f1_1 = (2 * prec_1 * rec_1) / (prec_1 + rec_1 + smooth)

    iou_0 = (tp_0 + smooth) / (tp_0 + fp_0 + fn_0 + smooth)
    prec_0 = (tp_0 + smooth) / (tp_0 + fp_0 + smooth)
    rec_0 = (tp_0 + smooth) / (tp_0 + fn_0 + smooth)
    pa_0 = rec_0
    f1_0 = (2 * prec_0 * rec_0) / (prec_0 + rec_0 + smooth)

    m_iou = (iou_1 + iou_0) / 2.0
    m_prec = (prec_1 + prec_0) / 2.0
    m_rec = (rec_1 + rec_0) / 2.0
    m_pa = (pa_1 + pa_0) / 2.0
    m_f1 = (f1_1 + f1_0) / 2.0

    print("\n📊 正在生成評價指標柱狀圖...")
    plot_metric_bar('IoU', iou_0, iou_1, m_iou, plot_dir)
    plot_metric_bar('Precision', prec_0, prec_1, m_prec, plot_dir)
    plot_metric_bar('Recall', rec_0, rec_1, m_rec, plot_dir)
    plot_metric_bar('PA', pa_0, pa_1, m_pa, plot_dir)
    plot_metric_bar('F1-score', f1_0, f1_1, m_f1, plot_dir)

    print("\n" + "=" * 20 + f" {CHOOSE_MODEL} 最終測試報告 " + "=" * 20)
    print(f"可視化圖像已保存至: {vis_dir}")
    print(f"評價指標柱狀圖已保存至: {plot_dir}")
    print("-" * 52)
    print(f"{'Metric':<15} | {'Background':<12} | {'Gully':<10} | {'Macro-Avg':<10}")
    print("-" * 52)
    print(f"{'IoU':<15} | {iou_0:<12.4f} | {iou_1:<10.4f} | {m_iou:<10.4f}")
    print(f"{'Precision':<15} | {prec_0:<12.4f} | {prec_1:<10.4f} | {m_prec:<10.4f}")
    print(f"{'Recall (PA)':<15} | {rec_0:<12.4f} | {rec_1:<10.4f} | {m_rec:<10.4f}")
    print(f"{'F1-score':<15} | {f1_0:<12.4f} | {f1_1:<10.4f} | {m_f1:<10.4f}")
    print("=" * 52)


if __name__ == "__main__":
    evaluate_and_visualize()