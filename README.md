# Monocular Depth Estimation in the Wild

面向真实场景的单目深度估计与鲁棒性分析。项目使用预训练深度模型，系统考察光照、模糊、噪声和低分辨率对预测稳定性的影响。

> 本项目评估的是**相对深度**与预测稳定性。模型输出不能直接解释为“距离摄像头多少米”。在没有真实深度标注时，结果只能说明预测发生了怎样的变化，不能单独证明哪张深度图更准确。

## 研究问题

1. 哪类真实场景变化最容易破坏模型的相对深度顺序、整体结构和边界？
2. 不同模型或输入处理方法在相同扰动下有何差异？
3. 能否用低成本预处理改善某一类失效场景？（进阶）

当前版本覆盖前 4 周的可复现实验基础：

- Depth Anything V2 Small 与可选 MiDaS/DPT 推理；
- 确定性的亮度、模糊、噪声、低分辨率扰动；
- 单图和批量推理，保存原始深度数组与可视化；
- 无真实标注时的稳定性指标与 CSV 汇总；
- 基础曲线图和最差案例表；
- 不下载模型即可运行的单元测试。

## 目录结构

```text
.
├── configs/experiment.json       # 固定实验设置
├── data/
│   ├── raw/                      # 原始图片（不提交）
│   └── README.md
├── outputs/                      # 生成结果（不提交）
├── src/depth_wild/
│   ├── perturb.py                # 构建扰动测试集
│   ├── model.py                  # 预训练模型适配器
│   ├── infer.py                  # 批量推理
│   ├── metrics.py                # 稳定性指标
│   ├── evaluate.py               # 评估与 CSV 汇总
│   └── plot.py                   # 图表与最差案例
├── tests/                        # 无模型下载测试
├── REPORT_TEMPLATE.md
├── pyproject.toml
└── requirements.txt
```

## 1. 安装

推荐 Python 3.10–3.12。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

默认依赖安装 CPU 版本的 PyTorch。若以后使用 NVIDIA GPU，请按 [PyTorch 官方安装器](https://pytorch.org/get-started/locally/) 选择匹配的 CUDA 版本，不要混装多个版本。

首次推理会从 Hugging Face 下载权重，需联网并占用本地缓存。默认模型为 `depth-anything/Depth-Anything-V2-Small-hf`。

## 2. 准备图片

把许可使用的 `.jpg`、`.jpeg` 或 `.png` 图片放入 `data/raw/`。建议先用 10 张图片验证流程，再扩展到 100–200 张。文件名应唯一，尽量使用英文、数字、连字符或下划线。

数据应覆盖室内、室外、低光、细小物体、玻璃/反光表面等场景。项目不会自动下载或提交数据。

## 3. 生成固定扰动集

```powershell
depth-wild-perturb --input data/raw --output outputs/dataset --config configs/experiment.json
```

输出包含处理后的图片和 `outputs/dataset/manifest.csv`。每个原图都会保存一个 `original` 基准版本；噪声由图片 ID 和全局种子共同决定，多次运行结果一致。

默认实验强度：

- 亮度：`0.4, 0.7, 1.3, 1.6`
- 高斯模糊半径：`1.0, 2.0, 4.0`
- 高斯噪声标准差（0–255）：`5, 15, 30`
- 缩放比例：`0.75, 0.5, 0.25`（缩小后再恢复到原尺寸）

修改强度时请保存配置并在报告中说明；模型之间必须使用同一份 manifest。

## 4. 批量推理

```powershell
depth-wild-infer --manifest outputs/dataset/manifest.csv --output outputs/predictions/depth_anything_v2
```

默认保存：

- `raw/*.npy`：浮点深度预测，用于评估；
- `visualizations/*.png`：逐图归一化的彩色预览，只用于观察；
- `predictions.csv`：模型、输入、输出路径和耗时记录。

CPU 首次测试建议只处理 1–3 张图。模型仍会在内部使用其预训练输入尺寸，因此即使原图较小，单张也可能需要数分钟；实际速度取决于 CPU、线程设置和软件版本。

```powershell
depth-wild-infer --manifest outputs/dataset/manifest.csv --output outputs/smoke --limit 3 --device cpu
```

比较 MiDaS/DPT：

```powershell
depth-wild-infer --manifest outputs/dataset/manifest.csv --output outputs/predictions/midas --model Intel/dpt-hybrid-midas
```

`--device auto` 会优先使用 CUDA，其次 Apple MPS，否则使用 CPU。可显式传入 `--device cpu`。

## 5. 评估稳定性

```powershell
depth-wild-evaluate --predictions outputs/predictions/depth_anything_v2/predictions.csv --output outputs/analysis/depth_anything_v2
```

每个扰动预测会与同一原图的 `original` 预测比较：

| 指标 | 含义 | 趋势 |
|---|---|---|
| `spearman` | 像素相对深度排序相关性 | 越高越稳定 |
| `order_consistency` | 随机像素对前后关系的一致率 | 越高越稳定 |
| `affine_nrmse` | 对尺度和偏移对齐后的结构误差 | 越低越稳定 |
| `edge_f1` | 强深度边缘位置的重合程度 | 越高越稳定 |

这些指标衡量“相对原预测是否稳定”，不等同于相对真实深度的准确率。若加入 KITTI、NYU Depth V2 或 DA-2K，必须另行遵循官方划分、裁剪、尺度对齐和评估协议。

## 6. 生成图表

```powershell
depth-wild-plot --metrics outputs/analysis/depth_anything_v2/metrics.csv --output outputs/analysis/depth_anything_v2/figures
```

会生成各扰动强度下的均值曲线与 `worst_cases.csv`。正式报告还应检查原图、深度图和边缘，不应只按单一指标下结论。

## 7. 测试

```powershell
pytest
```

测试不需要下载模型，覆盖扰动的确定性、尺寸保持、仿射不变误差和排序一致性。

## 公平实验清单

- 固定原始图片、扰动强度、随机种子和输入清单；
- 不根据结果临时删除失败样本；损坏文件应记录并单独说明；
- 同一比较中的模型使用相同输入文件；
- `.npy` 用于计算，彩色预览不能作为定量输入；
- 报告每类样本数、均值、标准差，并展示代表性与最差案例；
- 明确区分“稳定性下降”和“有真实标注证明的准确率下降”；
- 记录模型名称、依赖版本、硬件和运行日期。

## 已知局限

- 当前流程没有真实深度标注，不能计算 AbsRel、RMSE 等标准准确率指标；
- 逐图仿射对齐会忽略全局尺度和偏移变化，适合相对深度稳定性分析；
- 边缘指标使用图像级分位数阈值，适合模型内对比，不替代标准基准协议；
- 合成扰动不能覆盖真实相机在低光、运动模糊和压缩中的全部退化；
- 不同预训练模型的训练数据可能重叠，跨模型结论需谨慎解释。

## 参考资料

- Yang et al., **Depth Anything V2**, NeurIPS 2024. [论文](https://arxiv.org/abs/2406.09414) · [官方仓库](https://github.com/DepthAnything/Depth-Anything-V2)
- Ranftl et al., **Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer**, TPAMI 2022. [论文](https://arxiv.org/abs/1907.01341)
- [Hugging Face 单目深度估计文档](https://huggingface.co/docs/transformers/tasks/monocular_depth_estimation)
- [KITTI Depth Prediction Benchmark](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)

请在实验完成后再把真实数字写入简历或报告；不要把预训练模型调用描述成原创算法。
