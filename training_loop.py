import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F  # 🌟 新增導入：形態學操作與動態邊緣提取需要
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt


# ================= 1. 三級可調控組合損失函數 =================
class FlexibleLoss(nn.Module):
    def __init__(self, use_bce=True, use_edge_weight=False, use_dice=True, edge_weight_val=2.0):
        super(FlexibleLoss, self).__init__()
        self.use_bce = use_bce
        self.use_edge_weight = use_edge_weight
        self.use_dice = use_dice
        self.edge_weight_val = edge_weight_val

        # 初始化基礎單元
        self.bce_mean = nn.BCEWithLogitsLoss()  # 用於標準BCE
        self.bce_none = nn.BCEWithLogitsLoss(reduction='none')  # 用於邊緣加權BCE像素點矩陣相乘

        if not self.use_bce and not self.use_edge_weight and not self.use_dice:
            raise ValueError("❌ 錯誤：不能同時將 use_bce, use_edge_weight 和 use_dice 設置為 False！")

    def forward(self, inputs, targets):
        total_loss = torch.tensor(0.0, device=inputs.device)

        # 核心 1：標準 BCE 損失分支
        if self.use_bce:
            total_loss = total_loss + self.bce_mean(inputs, targets)

        # 核心 2：邊緣感知加權 BCE 損失分支 (Edge-Aware BCE)
        if self.use_edge_weight:
            # 使用最大池化進行形態學膨脹，擴張侵蝕溝邊界
            target_dilated = F.max_pool2d(targets, kernel_size=3, stride=1, padding=1)
            # 膨脹後的標籤減去原始標籤，精確鎖定侵蝕溝外邊緣的一圈像素
            edge_mask = target_dilated - targets

            # 計算不平均的基礎 BCE 損失
            base_ce_loss = self.bce_none(inputs, targets)
            # 構建加權權重矩陣：普通像素為 1.0，邊緣像素增強為 edge_weight_val
            weight_matrix = torch.ones_like(targets) + edge_mask * (self.edge_weight_val - 1.0)
            # 矩陣點乘並求全局均值
            weighted_ce_loss = (base_ce_loss * weight_matrix).mean()
            total_loss = total_loss + weighted_ce_loss

        # 核心 3：Dice 損失分支
        if self.use_dice:
            inputs_sigmoid = torch.sigmoid(inputs)
            smooth = 1e-5
            intersection = (inputs_sigmoid * targets).sum(dim=(2, 3))
            union = inputs_sigmoid.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
            dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
            total_loss = total_loss + dice_loss.mean()

        return total_loss


# ================= 2. 精度計算函數 =================
def calculate_metrics(preds, targets, threshold=0.5):
    """
    基於混淆矩陣計算嚴格的雙類平均 (Macro-Average) 指標：
    mIoU, mF1 (mDice), mPrecision, mRecall
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)

    smooth = 1e-5

    # 類別 1: 侵蝕溝
    tp_1 = (preds * targets).sum()
    fp_1 = (preds * (1 - targets)).sum()
    fn_1 = ((1 - preds) * targets).sum()

    iou_1 = (tp_1 + smooth) / (tp_1 + fp_1 + fn_1 + smooth)
    prec_1 = (tp_1 + smooth) / (tp_1 + fp_1 + smooth)
    rec_1 = (tp_1 + smooth) / (tp_1 + fn_1 + smooth)
    f1_1 = (2. * prec_1 * rec_1) / (prec_1 + rec_1 + smooth)

    # 類別 0: 背景
    tp_0 = ((1 - preds) * (1 - targets)).sum()
    fp_0 = ((1 - preds) * targets).sum()
    fn_0 = (preds * (1 - targets)).sum()

    iou_0 = (tp_0 + smooth) / (tp_0 + fp_0 + fn_0 + smooth)
    prec_0 = (tp_0 + smooth) / (tp_0 + fp_0 + smooth)
    rec_0 = (tp_0 + smooth) / (tp_0 + fn_0 + smooth)
    f1_0 = (2. * prec_0 * rec_0) / (prec_0 + rec_0 + smooth)

    # 雙類平均
    m_iou = (iou_1 + iou_0) / 2.0
    m_prec = (prec_1 + prec_0) / 2.0
    m_rec = (rec_1 + rec_0) / 2.0
    m_f1 = (f1_1 + f1_0) / 2.0

    return m_iou.item(), m_f1.item(), m_prec.item(), m_rec.item()


# ================= 3. 繪圖函數 =================
def save_plots(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['val_iou'], 'g', label='Validation mIoU')
    plt.title('Validation mIoU Change')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'miou_curve.png'))
    plt.close()


# ================= 4. 主訓練函數 =================
def train_model(model, train_loader, val_loader, optimizer, epochs=50, device='cuda', save_dir="runs/experiment",
                use_bce=True, use_edge_weight=False, use_dice=True, edge_weight_val=2.0):
    start_time_raw = datetime.now()

    # 確保保存實驗數據的資料夾存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 🌟 實例化全新升級的三級可控損失函數
    criterion = FlexibleLoss(use_bce=use_bce, use_edge_weight=use_edge_weight, use_dice=use_dice,
                             edge_weight_val=edge_weight_val)
    writer = SummaryWriter(log_dir=save_dir)

    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    best_iou = 0.0

    print("=" * 50)
    print(f"🚀 訓練開始時間: {start_time_raw.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 實驗數據路徑: {save_dir}")
    print(f"🧪 損失函數配置: BCE={use_bce} | Edge_BCE={use_edge_weight}(Weight={edge_weight_val}) | Dice={use_dice}")
    print("=" * 50)

    start_tick = time.time()

    try:
        for epoch in range(epochs):
            # --- 訓練階段 ---
            model.train()
            train_loss = 0
            train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

            for images, masks in train_loop:
                images, masks = images.to(device), masks.to(device)
                predictions = model(images)
                loss = criterion(predictions, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_loop.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / len(train_loader)

            # --- 驗證階段 ---
            model.eval()
            val_loss = 0
            val_miou_total, val_mf1_total = 0.0, 0.0
            val_mprec_total, val_mrec_total = 0.0, 0.0

            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
                for val_images, val_masks in val_loop:
                    val_images, val_masks = val_images.to(device), val_masks.to(device)
                    val_preds = model(val_images)

                    v_loss = criterion(val_preds, val_masks)
                    val_loss += v_loss.item()

                    m_iou, m_f1, m_prec, m_rec = calculate_metrics(val_preds, val_masks)

                    val_miou_total += m_iou
                    val_mf1_total += m_f1
                    val_mprec_total += m_prec
                    val_mrec_total += m_rec

            n_val_batches = len(val_loader)
            avg_val_loss = val_loss / n_val_batches
            avg_val_miou = val_miou_total / n_val_batches
            avg_val_mf1 = val_mf1_total / n_val_batches
            avg_val_mprec = val_mprec_total / n_val_batches
            avg_val_mrec = val_mrec_total / n_val_batches

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_iou'].append(avg_val_miou)

            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Val', avg_val_loss, epoch)
            writer.add_scalar('Metrics/mIoU', avg_val_miou, epoch)
            writer.add_scalar('Metrics/mF1', avg_val_mf1, epoch)
            writer.add_scalar('Metrics/mPrecision', avg_val_mprec, epoch)
            writer.add_scalar('Metrics/mRecall', avg_val_mrec, epoch)

            print(f"\nEpoch {epoch + 1} 總結: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(
                f"雙類平均指標 -> mIoU: {avg_val_miou:.4f} | mF1: {avg_val_mf1:.4f} | mPrec: {avg_val_mprec:.4f} | mRec: {avg_val_mrec:.4f}")

            # ==================================================== #
            #                  三級權重保存邏輯                    #
            # ==================================================== #

            if avg_val_miou > best_iou:
                best_iou = avg_val_miou
                best_model_path = os.path.join(save_dir, "best_epoch_weights.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"🌟 mIoU 提升至 {best_iou:.4f}，已更新: best_epoch_weights.pth")

            last_model_path = os.path.join(save_dir, "last_epoch_weights.pth")
            torch.save(model.state_dict(), last_model_path)

            if (epoch + 1) % 5 == 0:
                periodic_filename = f"ep{epoch + 1:03d}-loss{avg_train_loss:.3f}-val_loss{avg_val_loss:.3f}.pth"
                periodic_model_path = os.path.join(save_dir, periodic_filename)
                torch.save(model.state_dict(), periodic_model_path)
                print(f"💾 達到 5 輪節點，已保存里程碑權重: {periodic_filename}")

            # ==================================================== #

            save_plots(history, save_dir)

    except KeyboardInterrupt:
        print("\n訓練被手動中斷。")

    # 寫入時間記錄日誌
    end_tick = time.time()
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time_str = start_time_raw.strftime('%Y-%m-%d %H:%M:%S')

    total_seconds = int(end_tick - start_tick)
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    print("\n" + "=" * 50)
    print(f"🏁 訓練結束時間: {end_time_str}")
    print(f"⏱️ 訓練總耗時: {minutes} 分鐘 {seconds} 秒")
    print(f"📊 最終圖像與日誌已保存至: {save_dir}")
    print("=" * 50)

    time_log_path = os.path.join(save_dir, "training_time_log.txt")
    with open(time_log_path, "w", encoding="utf-8") as f:
        f.write("=" * 30 + "\n")
        f.write("        訓練時間記錄\n")
        f.write("=" * 30 + "\n")
        f.write(f"訓練開始時間: {start_time_str}\n")
        f.write(f"訓練結束時間: {end_time_str}\n")
        f.write(f"訓練總耗時:   {minutes} 分钟 {seconds} 秒\n")

    writer.close()