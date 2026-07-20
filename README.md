# GullyFormer
> 本项目为硕士学位论文相关研究成果公开版本，仅用于学术交流与非商业研究。
# ⚠️ 项目声明（Project Statement）

## 原创性声明

GullyFormer 是作者在硕士研究期间针对**东北黑土地侵蚀沟智能提取任务**自主设计与实现的深度学习模型框架。本项目相关内容来源于作者硕士学位论文研究工作，主要包括模型结构设计、网络优化方法、实验流程、数据处理方法以及代码实现等。

本项目旨在促进遥感影像智能解译领域的学术交流与方法研究，目前仅作为科研学习与非商业用途公开。

任何个人或组织在使用本项目代码、模型结构或相关研究成果时，应遵守以下规范：

1. **禁止将本项目中的代码、模型结构、实验结果或研究方法直接作为自身原创成果进行发表、申请专利或其他形式的成果申报；**

2. **禁止删除、修改作者信息后重新发布本项目代码或衍生项目；**

3. **未经作者授权，不得将本项目用于商业软件开发、商业产品集成或商业服务；**

4. **如基于本项目开展相关研究工作，请在论文、报告或其他公开成果中规范引用本项目，并明确说明相关方法来源；**

5. **如对本项目代码进行修改或扩展并公开发布，应继续遵守相同的许可协议。**

本项目采用：
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License  
（CC BY-NC-SA 4.0）**

该协议允许：

- ✅ 学术研究与学习使用；
- ✅ 非商业环境下的代码复现与方法验证；
- ✅ 在保留作者署名和许可协议的前提下进行修改与扩展。


该协议禁止：

- ❌ 商业用途使用；
- ❌ 未经授权的成果包装与二次发布；
- ❌ 未注明来源的研究成果引用。

作者保留本项目相关研究成果的署名权、解释权及其他合法权益。

**Copyright © 2026 GullyFormer Author. All Rights Reserved.**

该模型专门为侵蚀沟提取任务设计，以DeeplabV3+模型架构为基础，使用SegFormer作为主干网络，并且将ASPP模块换成作者的HybridASPP模块，同时兼具全局和局部信息提取，提高模型侵蚀沟提取任务中的表现
本项目提供了一套端到端的深度学习遥感影像分割解决方案，专为高分辨率卫星影像中的地理要素（如侵蚀沟）自动化提取而设计。基于改进的 SegFormer 架构，结合了 MiT 编码器与 HybridASPP 解码器。项目涵盖了从模型训练、难负样本挖掘、精度评估到大规模遥感影像滑动窗口预测及 Shapefile 矢量化输出的完整工作流。

## ✨ 核心特性 (Key Features)

* **定制化网络架构**：基于 `CustomSegFormer`，将预训练的 Transformer 编码器 (MiT) 与 HybridASPP 模块进行缝合。提供独立的 `use_aspp` 开关，方便进行消融实验。
* **工业级大图推理**：针对超大遥感影像，采用 2%-98% 全局极值拉伸防溢出策略。内置 2D 汉宁窗 (Hanning Window) 的滑动窗口加权拼接算法，有效消除分块预测产生的拼接缝。
* **GIS 软件无缝衔接**：预测结果不仅输出为 `.tif` 栅格，还会自动进行面积过滤并转换为 ESRI Shapefile (`.shp`) 格式，生成的矢量折线可直接导入 ArcGIS、ArcGIS Pro 或 QGIS 进行后续编辑与后处理。
* **三级可控组合损失函数**：训练循环内置 `FlexibleLoss`，支持标准 BCE、边缘感知加权 BCE (Edge-Aware BCE) 以及 Dice Loss 的自由组合。（实验表明，针对当前任务，Cross Entropy + Dice 组合通常能取得最佳效果）。
* **难负样本优化**：Dataloader 智慧兼容 `.jpg` 和 `.png` 混合后缀，彻底解决向训练集中动态追加难负样本时导致的 `FileNotFoundError` 隐患。

---

## 📂 项目结构 (Project Structure)

```text
├── Data_loader.py       # VOC 格式数据集加载，支持灵活的 train_split 及难负样本路由
├── Segformer_model.py   # 模型定义文件，包含 HybridASPP 模块及 CustomSegFormer 缝合逻辑
├── training_loop.py     # 训练核心逻辑，包含 FlexibleLoss、指标计算及 TensorBoard 日志记录
├── main.py              # 模型训练主入口，管理超参数、随机种子、损失函数开关及文件夹命名
├── test_model.py        # 模型评估脚本，输出 mIoU、F1 等多项宏平均指标及柱状图可视化
├── predict_large_image.py # 大图推理脚本，执行分块预测、Hanning窗加权及 Shapefile 生成
└── SegFormer            #存放模型的预训练权重，这里作者使用的b0版本

```
<img width="1890" height="1758" alt="图片11" src="https://github.com/user-attachments/assets/467c2720-dad6-4239-89d9-6ac996962773" />

---

## 🚀 快速开始 (Quick Start)

### 1. 环境依赖
确保您的环境中安装了以下基础依赖：
`torch`, `torchvision`, `transformers`, `rasterio`, `geopandas`, `shapely`, `tqdm`, `matplotlib`, `PIL` (Pillow)。

### 2. 数据准备
项目默认采用标准 VOC2007 目录结构组织遥感切片数据集：
* 原图放置于 `VOCdevkit/VOC2007/JPEGImages/`。
* 标签掩码放置于 `VOCdevkit/VOC2007/SegmentationClass/`。
* 划分名单 `.txt` 放置于 `VOCdevkit/VOC2007/ImageSets/Segmentation/`。

### 3. 模型训练
通过修改 `main.py` 中的核心配置区域来启动训练。
* **损失函数推荐配置**：在 `main.py` 中，建议将 `USE_BCE = True` 和 `USE_DICE = True` 开启，这是经过实验验证的最佳组合。
* **难负样本开关**：开启 `USE_HARD_NEG` 并设置 `NEG_RATIO`，程序将自动加载对应的 `train_N_xxx.txt` 数据划分。
* **模块控制**：通过 `USE_ASPP` 可以手动且自由地控制是否在训练中启用 ASPP 模块。
* **启动命令**：
```bash
python main.py
```
训练过程中的最优权重 (`best_epoch_weights.pth`)、最新权重及每 5 轮的里程碑权重将自动保存在 `runs/` 目录下，并同步生成 TensorBoard 记录及 Loss/mIoU 变化曲线图。

---

## 📊 评估与测试 (Evaluation)

使用 `test_model.py` 对训练好的模型进行精度评估。
1. 在脚本中配置您的 `TRAINED_WEIGHT_PATH` 和自定义结果目录 `CUSTOM_RESULT_DIR`。
2. 确保 `USE_ASPP` 的开关状态与您训练该权重时的网络结构严格一致。
3. 运行脚本：
```bash
python test_model.py
```
程序将输出背景与目标类的 IoU、Precision、Recall (PA) 和 F1-score 报告，并将假阳性、假阴性的对比可视化图像及评价指标柱状图保存在结果文件夹中。

---
<img width="996" height="255" alt="image" src="https://github.com/user-attachments/assets/24e15e88-3174-4321-88f9-eda543987c11" />


<img width="1778" height="1960" alt="图片13" src="https://github.com/user-attachments/assets/0e25580b-5897-4baa-abc0-9fb3f3dc0169" />


## 🌍 大图推理与矢量化 (Large Image Inference)

针对原始高分辨率卫星影像，使用 `predict_large_image.py` 进行端到端提取。
1. 在脚本中配置待处理的 `.tif` 影像路径 (`IMAGE_PATHS`)、权重路径 (`WEIGHT_PATH`) 及输出目录 (`SAVE_DIR`)。
2. 脚本使用 `rasterio` 动态读取影像区块，通过滑动窗口与 Hanning 窗叠加融合局部预测概率。
3. 预测概率图经过二值化后，脚本利用 `geopandas` 过滤面积小于设定像素阈值的细碎斑块，简化几何边界生成折线轮廓，最终输出可以直接用于制图或空间分析的 `.shp` 文件。
```bash
python predict_large_image.py
```
Copyright © 2026 GullyFormer Author. All Rights Reserved.
## 📜 开源协议
本项目采用 **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License（CC BY-NC-SA 4.0）协议**。
详细协议内容请参考：
https://creativecommons.org/licenses/by-nc-sa/4.0/
根据该协议：
- 使用者可以用于学术研究、教学和非商业实验；
- 使用者需要保留作者署名及项目来源；
- 基于本项目修改或扩展的公开成果需要采用相同协议；
- 未经作者许可，不得用于商业用途。


如果您希望将本项目用于商业应用、商业软件开发或其他授权用途，请提前联系作者。
