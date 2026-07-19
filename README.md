# RDE + DDI：噪声对应下的文本行人检索

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-ee4c2c.svg)](https://pytorch.org/)
[![CVPR 2024](https://img.shields.io/badge/CVPR-2024-8a2be2.svg)](https://openaccess.thecvf.com/)

本仓库基于 CVPR 2024 项目 [RDE](https://github.com/QinYang79/RDE) 进行改进、训练和实验。除保留原始 RDE 的噪声对应学习框架外，本仓库还提供：

- RSTPReid 在 `0.0 / 0.2 / 0.5` 噪声率下的训练脚本、配置与完整日志；
- 对应实验的 `best.pth` 和 `last.pth` 权重；
- 双粒度分歧交互（Dual-grained Disagreement Interaction, DDI）扩展；
- 两组 Qwen 视觉语言模型的 DDI 实验与消融结果；
- 与实际训练环境一致的 Python 依赖版本。

> 原始论文：**Noisy-Correspondence Learning for Text-to-Image Person Re-identification**，CVPR 2024。论文 PDF 见 [`src/RDE_main.pdf`](src/RDE_main.pdf)。

## 方法概览

RDE 使用两种不同粒度的跨模态表示：

- **BGE（Basic Global Embedding）**：建模图像与文本的全局对应关系；
- **TSE（Token Selection Embedding）**：选择有信息量的局部 token，增强细粒度匹配。

在噪声对应学习中，RDE 使用 Confident Consensus Division（CCD）筛选可信样本，并通过 Triplet Alignment Loss（TAL）进行稳健相似度学习。

本仓库进一步加入 **DDI**：当 BGE 与 TSE 的候选排序存在分歧时，利用视觉语言模型生成一个针对性问题，并将确认后的可见属性写回文本查询，从而在测试时补充判别信息。

![RDE framework](src/frame.png)

## RSTPReid 训练结果

下表来自三个 Release 中 `best.pth` 在完整 RSTPReid 测试集上的 BGE+TSE 文本到图像检索结果。测试集包含 2,000 条文本和 1,000 张图像。

| 噪声率 | 最佳 Epoch | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 50 | 65.55 | 84.00 | 89.35 | 51.09 | 28.26 |
| 0.2 | 46 | 65.45 | 83.45 | 88.75 | 50.69 | 27.83 |
| 0.5 | 27 | 63.00 | 82.20 | 88.35 | 46.93 | 23.84 |

对应训练日志和配置已保存在 [`2024-CVPR-RDE/run_logs/RSTPReid`](2024-CVPR-RDE/run_logs/RSTPReid)。

## DDI 实验摘要

DDI 使用固定的 200-query RSTPReid 子集和完整的 1,000 张测试图库。以下结果只用于 DDI 内部对比，不能与上方完整 2,000-query 结果直接比较。

| 方法 | 最佳轮次 | Rank-1 | Rank-5 | Rank-10 | mAP |
|---|---:|---:|---:|---:|---:|
| RDE 基线 | 0 | 69.50 | 84.50 | 89.00 | 53.99 |
| RDE + `qwen3-vl-flash` | 2 | 71.50 | 85.50 | 89.00 | 55.11 |
| RDE + `qwen3.6-flash` | 1 | 71.50 | 85.00 | 91.00 | 55.15 |

完整设置、逐轮结果、消融实验和案例分析见：

- [`DDI_README.md`](2024-CVPR-RDE/DDI_README.md)
- [`DDI_EXPERIMENT_RESULTS.md`](2024-CVPR-RDE/DDI_EXPERIMENT_RESULTS.md)

## 项目结构

```text
RDE/
├── README.md
├── requirements.txt
├── src/                         # 论文、框架图和海报
└── 2024-CVPR-RDE/
    ├── datasets/                # CUHK-PEDES、ICFG-PEDES、RSTPReid
    ├── model/                   # BGE、TSE 与 CLIP 主干
    ├── processor/               # 训练和推理流程
    ├── solver/                  # 优化器和学习率调度器
    ├── utils/                   # 日志、指标、checkpoint 等工具
    ├── noiseindex/              # 三个数据集的噪声索引
    ├── ddi/                     # DDI 核心逻辑与 Qwen 客户端
    ├── tests/                   # DDI 单元测试
    ├── run_logs/                # 已提交的配置、文本日志和 TensorBoard 日志
    ├── train.py
    ├── test.py
    └── run_ddi_experiment.py
```

模型权重体积较大，不存放在普通 Git 历史中，而是作为 GitHub Release Assets 发布。

## 环境安装

复现实验使用的环境为：

- Python 3.8.20
- PyTorch 1.9.0
- Torchvision 0.10.0
- CUDA 可用的 NVIDIA GPU

创建环境：

```bash
conda create -n rde python=3.8 -y
conda activate rde
```

建议先根据本机 CUDA 版本安装 PyTorch。原实验环境使用 PyTorch 1.9.0，例如 CUDA 11.1：

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

随后安装其余依赖：

```bash
pip install -r requirements.txt
```

如果使用不同 CUDA 版本，请从 [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) 选择对应的 PyTorch 和 Torchvision 构建。

## 数据集准备

本项目支持：

- CUHK-PEDES
- ICFG-PEDES
- RSTPReid

数据集不包含在仓库中，请分别遵循其官方许可获取。以 RSTPReid 为例，目录应为：

```text
/path/to/datasets/
└── RSTPReid/
    ├── data_captions.json
    └── imgs/
```

训练脚本中的 `root_dir` 应指向 `/path/to/datasets`，而不是直接指向 `RSTPReid` 子目录。

## 下载预训练权重

| 噪声率 | Release | best.pth | last.pth |
|---:|---|---|---|
| 0.0 | [weights-rstp-n00](https://github.com/rentianyu0415/RDE/releases/tag/weights-rstp-n00) | [下载](https://github.com/rentianyu0415/RDE/releases/download/weights-rstp-n00/best.pth) | [下载](https://github.com/rentianyu0415/RDE/releases/download/weights-rstp-n00/last.pth) |
| 0.2 | [weights-rstp-n02](https://github.com/rentianyu0415/RDE/releases/tag/weights-rstp-n02) | [下载](https://github.com/rentianyu0415/RDE/releases/download/weights-rstp-n02/best.pth) | [下载](https://github.com/rentianyu0415/RDE/releases/download/weights-rstp-n02/last.pth) |
| 0.5 | [weights-rstp-n05](https://github.com/rentianyu0415/RDE/releases/tag/weights-rstp-n05) | [下载](https://github.com/rentianyu0415/RDE/releases/download/weights-rstp-n05/best.pth) | [下载](https://github.com/rentianyu0415/RDE/releases/download/weights-rstp-n05/last.pth) |

如需复用仓库中的配置文件或运行 DDI，请把权重下载到相应日志目录。例如 `n=0.0`：

```bash
cd 2024-CVPR-RDE

RUN_DIR="run_logs/RSTPReid/20260714_224938_RDE_TAL+sr0.3_tau0.015_margin0.1_n0.0"

wget -O "$RUN_DIR/best.pth" \
  https://github.com/rentianyu0415/RDE/releases/download/weights-rstp-n00/best.pth

wget -O "$RUN_DIR/last.pth" \
  https://github.com/rentianyu0415/RDE/releases/download/weights-rstp-n00/last.pth
```

## 训练

进入代码目录：

```bash
cd 2024-CVPR-RDE
```

首先修改训练脚本中的：

```bash
root_dir=/path/to/datasets
```

运行 RSTPReid 三组实验：

```bash
# 0% 噪声
bash run_rde_rstp.sh

# 20% 噪声
bash run_rde_rstp_n02.sh

# 50% 噪声
bash run_rde_rstp_n05.sh
```

核心参数如下：

| 参数 | 值 |
|---|---|
| Batch size | 64 |
| Epochs | 60 |
| Optimizer | Adam |
| Learning rate | `1e-5` |
| Select ratio | 0.3 |
| Tau | 0.015 |
| Margin | 0.1 |
| Loss | TAL |
| Image/Text augmentation | 开启 |

每次训练会在 `run_logs/<dataset>/<timestamp>_<experiment>/` 中生成：

- `configs.yaml`
- `train_log.txt`
- TensorBoard event 文件
- `best.pth`
- `last.pth`

查看 TensorBoard：

```bash
tensorboard --logdir run_logs
```

`train.py` 会在训练结束后自动评估 `best.pth` 和 `last.pth`。如需单独使用 `test.py`，请先将其中的 `sub` 修改为目标运行目录，再运行：

```bash
python test.py
```

## 运行 DDI

DDI 默认读取 `n=0.0` 的 RSTPReid `best.pth`。确认权重已下载到默认目录后，设置兼容 OpenAI Chat Completions 格式的 Qwen 视觉模型接口：

```bash
export DASHSCOPE_API_KEY="YOUR_API_KEY"
export DASHSCOPE_BASE_URL="https://YOUR_ENDPOINT/compatible-mode/v1"
```

运行完整主实验和消融实验：

```bash
cd 2024-CVPR-RDE
python run_ddi_experiment.py --mode all
```

其他模式：

```bash
python run_ddi_experiment.py --mode main
python run_ddi_experiment.py --mode ablation
```

API Key 只从环境变量读取，不应写入代码、日志或 Git 仓库。更多参数和输出说明见 [`DDI_README.md`](2024-CVPR-RDE/DDI_README.md)。

## 测试

```bash
conda activate rde
cd 2024-CVPR-RDE
python -m unittest discover -s tests -p 'test_ddi*.py'
```

## 原始 RDE 说明

原始 RDE 的框架、论文结果和噪声对应研究汇总请参考：

- [QinYang79/RDE](https://github.com/QinYang79/RDE)
- [Noisy-Correspondence-Summary](https://github.com/QinYang79/Noisy-Correspondence-Summary)
- [IRRA](https://github.com/anosorae/IRRA)

## Citation

如果本项目对研究有帮助，请引用原始 RDE 论文：

```bibtex
@inproceedings{qin2024noisy,
  title={Noisy-Correspondence Learning for Text-to-Image Person Re-identification},
  author={Qin, Yang and Chen, Yingke and Peng, Dezhong and Peng, Xi and Zhou, Joey Tianyi and Hu, Peng},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## License 与致谢

本仓库是对原始 RDE 的研究性扩展。原始 README 声明项目采用 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)，使用和分发时请同时遵守原 RDE、IRRA、数据集及所调用模型服务的许可条款。

感谢原始 RDE 与 IRRA 作者公开代码和研究成果。
