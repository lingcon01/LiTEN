# LiTEN
![Static Badge](https://img.shields.io/badge/release-v%200.1.2-blue?label=release&color=blue) ![Static Badge](https://img.shields.io/badge/License-MIT%202.0-%23006400)

## 🌟 About LiTEN
**LiTEN (Learning Interactions with Tensorized Equivariant Networks)** is a novel equivariant neural network designed for high-precision, efficient 3D molecular property prediction. It introduces the **Tensorized Quadrangle Attention (TQA)** mechanism, which operates directly in Cartesian space, enabling accurate modeling of many-body interactions **without relying on spherical harmonics**, leading to significant computational savings.

We further provide **LiTEN-FF**, a **foundation model for biomolecular force fields**, pretrained and fine-tuned to support a wide range of tasks in both **vacuum and solvated environments**.


## 🔬 Applications

LiTEN-FF is designed to support **end-to-end molecular simulation and drug discovery pipelines**, and can be readily applied to:

* 🧬 **Conformer geometry optimization**
* 📈 **Free energy surface (FES) construction**
* 🔍 **Dihedral angle scanning**
* 🧩 **Batch conformer search**
* 📐 **Internal coordinate distribution analysis**
* ⚛️ **Large-scale molecular dynamics with ab initio accuracy**

More application scenarios are being explored and will be continuously added to this repository.

## 🧠 Future Plans

In upcoming versions of LiTEN-FF, we aim to:

* 🔄 Incorporate **knowledge distillation** for model compression and acceleration.
* ⚡ Enhance inference speed and scalability.
* 🔧 Expand compatibility with **protein folding** and **molecular generation models**.

## 📦 Installation

```bash
git clone https://github.com/lingcon01/LiTEN.git
cd LiTEN
conda create -n liten python=3.10
conda activate liten
pip install -r requirements.txt
```

## 📁 Project Structure

```
LiTEN/
├── config/                # training params of rmd17, md22 and chignolin
├── model/                 # Model architecture (LiTEN)
├── dataset/               # Datasets and loaders
├── scripts/               # Training and inference scripts
├── utils/                 # Need for use
└── README.md
```

## 📊 Reproduce Our Results

To reproduce the benchmark results reported in our paper (on datasets such as **rMD17**, **MD22**, **Chignolin**), we provide train scripts in the) directory.

### 🧪 Benchmark Evaluation Example

The following script demonstrates how to evaluate **LiTEN-FF** on the SPICE dataset using a provided pretrained checkpoint:

```bash
# Examples: train and test on rmd17. (unit: meV and meV/A)
python scripts/train_rmd17.py \
    --config_path ./config/md17.yml \
    --save_path ./ckpt \
    --molecule asiprin \
    --num_train 950 \
    --num_val 50 \
    --device cuda
```
```bash
# Examples: train and test on md22. (unit: meV and meV/A)
python scripts/train_md22.py \
    --config_path ./config/md22.yml \
    --save_path ./ckpt \
    --molecule AT_AT \
    --num_train 2500 \
    --num_val 500 \
    --device cuda
```
```bash
# Examples: train and test on chignolin. (unit: kcal/mol and kcal/mol/A)
python scripts/train_chignolin.py \
    --config_path ./config/chignolin.yml \
    --save_path ./ckpt \
    --molecule chignolin \
    --device cuda
```


