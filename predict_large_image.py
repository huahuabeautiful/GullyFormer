import os
import math
import time
import datetime
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.enums import Resampling
from torchvision import transforms
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import shape

from Segformer_model import CustomSegFormer

# =========================
# 1. 计算全图 2%-98% 极值 (极速且不占内存)
# =========================
def get_global_percentiles(tif_path, scale_factor=0.05):
    """通过读取缩小的概览图来计算全局极值，避免内存溢出"""
    print(f"📊 正在计算 {os.path.basename(tif_path)} 的全局 2%-98% 极值...")
    with rasterio.open(tif_path) as src:
        out_shape = (
            src.count,
            int(src.height * scale_factor),
            int(src.width * scale_factor)
        )
        overview = src.read(out_shape=out_shape, resampling=Resampling.bilinear)

    mins, maxs = [], []
    for c in range(3):
        band = overview[c].astype(np.float32)
        valid_pixels = band[band > 0]

        if len(valid_pixels) == 0:
            mins.append(0.0)
            maxs.append(1e-5)
        else:
            mins.append(np.percentile(valid_pixels, 2))
            maxs.append(np.percentile(valid_pixels, 98))

    print(f"✅ 全局极值计算完成: Min={mins}, Max={maxs}")
    return np.array(mins), np.array(maxs)


# =========================
# 2. 应用全局拉伸
# =========================
def apply_global_stretch_with_stats(patch_np, global_mins, global_maxs):
    stretched = np.zeros_like(patch_np, dtype=np.float32)
    for c in range(3):
        band = patch_np[:, :, c].astype(np.float32)
        min_val = global_mins[c]
        max_val = global_maxs[c]

        if max_val - min_val < 1e-5:
            band = np.zeros_like(band)
        else:
            band = np.clip((band - min_val) / (max_val - min_val), 0, 1)

        stretched[:, :, c] = band * 255
    return stretched.astype(np.uint8)


# =========================
# 3. 生成 2D 汉宁窗 (用于加权拼接)
# =========================
def get_weight_window(size):
    y = np.hanning(size)
    x = np.hanning(size)
    window = np.outer(y, x).astype(np.float16)
    return np.maximum(window, 1e-5)


# =========================
# 4. SHP 生成与去噪
# =========================
def save_polygons_to_shp(geoms, crs, shp_path, min_area_pixels=80):
    if not geoms:
        print(f"⚠️ 无有效预测结果，未生成SHP: {shp_path}")
        return

    print("🗺️ 正在融合多边形并生成 SHP 文件...")
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)

    gdf["area"] = gdf.geometry.area
    gdf = gdf[gdf["area"] >= min_area_pixels]

    if len(gdf) == 0:
        print(f"⚠️ 面积过滤后无有效斑块: {shp_path}")
        return

    gdf["value"] = 1
    gdf = gdf.dissolve(by="value")
    gdf["geometry"] = gdf.geometry.simplify(1)

    gdf.to_file(shp_path, driver="ESRI Shapefile")
    print(f"✅ SHP文件已成功保存至: {shp_path}")


# =========================
# 辅助功能：写入日志
# =========================
def write_log(log_path, content):
    """将日志信息追加写入本地文件"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")
    print(content)


# =========================
# 5. 主预测流程
# =========================
def process_large_image_predict():
    WEIGHT_PATH = r"G:\Segformer2.0\runs\Negative_runs\loss_2026_06_02_14_25_33_Smart_MiT_HybridASPP_BCE_Dice_30ep_HardNeg_0.15\best_epoch_weights.pth"
    MODEL_PATH = r"G:\Segformer2.0\SegFormer"
    SAVE_DIR = r"G:\Segformer2.0\kedong_output\thershould0.99"
    IMAGE_PATHS = [r"D:\kedong\kedong_cropped.tif"]

    # 设置日志文件路径
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_FILE = os.path.join(SAVE_DIR, r"G:\Segformer2.0\kedong_output\thershould0.99\Processing_Log.txt")
    write_log(LOG_FILE, f"\n{'='*50}\n🚀 任务启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 预测参数设置
    WINDOW_SIZE = 256
    STRIDE = 128  # 50% 重叠
    THRESHOLD = 0.99

    # 物理分块参数
    BLOCK_SIZE = 2048
    BUFFER = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = CustomSegFormer(out_channels=1, model_size="b0", pretrained_path=MODEL_PATH).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()

    transform = transforms.ToTensor()
    task_list = [p for p in IMAGE_PATHS if os.path.isfile(p)]

    weight_window = get_weight_window(WINDOW_SIZE)

    for idx, img_path in enumerate(tqdm(task_list, desc="🌍 总体影像进度", position=0)):
        img_start_time = time.time()  # 记录单张影像开始时间
        total_patch_time = 0.0        # 累加切片处理耗时
        total_patch_count = 0         # 记录切片总数

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_tif = os.path.join(SAVE_DIR, f"{base}_Result.tif")
        out_shp = os.path.join(SAVE_DIR, f"{base}.shp")

        print(f"\n🚀 开始处理: {base}")
        global_mins, global_maxs = get_global_percentiles(img_path, scale_factor=0.05)
        all_geoms = []

        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            width, height = src.width, src.height
            crs = src.crs
            res_x, res_y = src.res  # 获取分辨率

            # 计算面积
            pixel_area = width * height
            # 假设投影单位为米(m)，面积转为平方公里(km²)；若为经纬度度(deg)则无物理意义，仅供参考
            physical_area_sq_km = (pixel_area * res_x * res_y) / 1_000_000

            meta.update({'count': 3, 'dtype': 'uint8'})

            blocks_y = math.ceil(height / BLOCK_SIZE)
            blocks_x = math.ceil(width / BLOCK_SIZE)
            total_blocks = blocks_y * blocks_x

            with rasterio.open(out_tif, "w", **meta) as dst, \
                    tqdm(total=total_blocks, desc=f"🧩 分块加权预测 {base}", position=1, leave=False) as pbar:

                for y_off in range(0, height, BLOCK_SIZE):
                    for x_off in range(0, width, BLOCK_SIZE):

                        block_w = min(BLOCK_SIZE, width - x_off)
                        block_h = min(BLOCK_SIZE, height - y_off)

                        read_x = max(0, x_off - BUFFER)
                        read_y = max(0, y_off - BUFFER)
                        read_w = min(width - read_x, x_off + block_w + BUFFER - read_x)
                        read_h = min(height - read_y, y_off + block_h + BUFFER - read_y)

                        window = Window(read_x, read_y, read_w, read_h)
                        img_block = src.read([1, 2, 3], window=window)
                        img_block = np.transpose(img_block, (1, 2, 0))

                        pad_h = max(0, WINDOW_SIZE - read_h)
                        pad_w = max(0, WINDOW_SIZE - read_w)
                        if pad_h > 0 or pad_w > 0:
                            img_block = np.pad(img_block, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                            read_h += pad_h
                            read_w += pad_w

                        local_prob = np.zeros((read_h, read_w), dtype=np.float16)
                        local_weight = np.zeros((read_h, read_w), dtype=np.float16)

                        y_coords = list(range(0, read_h - WINDOW_SIZE + 1, STRIDE))
                        if read_h > WINDOW_SIZE and (read_h - WINDOW_SIZE) not in y_coords:
                            y_coords.append(read_h - WINDOW_SIZE)

                        x_coords = list(range(0, read_w - WINDOW_SIZE + 1, STRIDE))
                        if read_w > WINDOW_SIZE and (read_w - WINDOW_SIZE) not in x_coords:
                            x_coords.append(read_w - WINDOW_SIZE)

                        # 执行滑动窗口预测
                        for y in y_coords:
                            for x in x_coords:
                                patch_start_time = time.time()  # 🌟 开始单张切片计时

                                patch = img_block[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]
                                patch_stretch = apply_global_stretch_with_stats(patch, global_mins, global_maxs)
                                tensor = transform(patch_stretch).unsqueeze(0).to(device)

                                with torch.no_grad():
                                    pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

                                local_prob[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE] += (pred * weight_window)
                                local_weight[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE] += weight_window

                                total_patch_time += (time.time() - patch_start_time)  # 🌟 累加单切片耗时
                                total_patch_count += 1                                # 🌟 累加切片数量

                        local_prob /= local_weight
                        local_mask = (local_prob > THRESHOLD).astype(np.uint8)

                        start_x = x_off - read_x
                        start_y = y_off - read_y
                        valid_mask = local_mask[start_y: start_y + block_h, start_x: start_x + block_w]

                        write_window = Window(x_off, y_off, block_w, block_h)
                        rgb = np.stack([valid_mask * 255] * 3, axis=0).astype(np.uint8)
                        dst.write(rgb, window=write_window)

                        block_transform = src.window_transform(write_window)
                        for geom, value in shapes(valid_mask, transform=block_transform):
                            if value == 1:
                                all_geoms.append(shape(geom))

                        del local_prob, local_weight, img_block, local_mask, valid_mask
                        pbar.update(1)

        # 统一去重融合，生成 SHP
        try:
            save_polygons_to_shp(all_geoms, crs, out_shp, min_area_pixels=80)
        except Exception as e:
            print(f"❌ SHP生成失败 {base}: {e}")

        img_end_time = time.time()
        total_time = img_end_time - img_start_time
        avg_patch_time_ms = (total_patch_time / total_patch_count * 1000) if total_patch_count > 0 else 0

        # 🌟 汇总当前影像统计信息并写入日志
        report = f"""
--------------------------------------------------
📄 处理报告: {base}
--------------------------------------------------
📏 图像尺寸: {width} × {height} 像素
📐 空间分辨率: {res_x} × {res_y}
🌐 物理面积: 约 {physical_area_sq_km:.2f} 平方公里 (仅投影坐标系下准确)
🧱 物理分块: {total_blocks} 块 ({blocks_x}列 × {blocks_y}行)
✂️ 模型切片: {total_patch_count} 张 (窗口 {WINDOW_SIZE}, 步长 {STRIDE})
⏱️ 预测总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)
⚡ 平均切片处理耗时: {avg_patch_time_ms:.2f} 毫秒/张 (含前处理+推理+加权)
--------------------------------------------------"""
        write_log(LOG_FILE, report)

    write_log(LOG_FILE, f"✅ 所有任务于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 处理完毕！\n{'='*50}\n")


if __name__ == "__main__":
    process_large_image_predict()