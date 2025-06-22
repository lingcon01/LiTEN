# LiTEN
![Static Badge](https://img.shields.io/badge/release-v%200.1.2-blue?label=release&color=blue) ![Static Badge](https://img.shields.io/badge/License-MIT%202.0-%23006400)

## 🌟 About LiTEN
**LiTEN (Learning Interactions with Tensorized Equivariant Networks)** is a novel equivariant neural network designed for high-precision, efficient 3D molecular property prediction. It introduces the **Tensorized Quadrangle Attention (TQA)** mechanism, which operates directly in Cartesian space, enabling accurate modeling of many-body interactions **without relying on spherical harmonics**, leading to significant computational savings.

We further provide **LiTEN-FF**, a **foundation model for biomolecular force fields**, pretrained and fine-tuned to support a wide range of tasks in both **vacuum and solvated environments**.
![workflow](https://github.com/user-attachments/assets/ffc2d515-a0c0-4a5f-ab85-a9edc284f8f9)


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
## ⚙️ Utilities Based on LiTEN-FF

We provide a suite of practical utilities built upon the pretrained LiTEN-FF model for downstream molecular simulation tasks, including **geometry optimization**, **molecular dynamics**, and **batch conformer generation**. These scripts are based on [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/) and can be easily adapted to custom molecules and workflows.

---

### 🧬 ASE-OPT: Geometry Optimization

**Function**:
Performs energy minimization of molecules using LiTEN-FF as a force field backend via ASE's `BFGS` optimizer or other algorithms.

**Example Usage**:

```bash
python tools/ase_opt.py \
    --input path/to/molecule.sdf \
    --model checkpoint/LiTEN_FF_pretrained.pt \
    --output optimized.sdf
```

**Key Features**:

* Supports SDF, XYZ, or PDB input.
* Compatible with vacuum or solvated structures.
* Output optimized geometries in the same format.

---

### 🔁 ASE-MD: Molecular Dynamics

**Function**:
Runs molecular dynamics (MD) simulations under NVE, NVT, or Langevin dynamics using LiTEN-FF as the force provider.

**Example Usage**:

```bash
python tools/ase_md.py \
    --input molecule.sdf \
    --model checkpoint/LiTEN_FF_pretrained.pt \
    --temperature 300 \
    --timestep 0.5 \
    --steps 10000 \
    --output trajectory.xyz
```

**Options**:

* `--temperature`: simulation temperature (in Kelvin)
* `--timestep`: timestep in femtoseconds
* `--steps`: number of simulation steps
* `--thermostat`: use `langevin`, `nvt`, or `nve`

---

### 🧩 Batch-confgen: Conformer Generation

**Function**:
Generates low-energy conformers for multiple molecules using LiTEN-FF with geometry optimization for each initial guess. Suitable for 3D dataset construction and screening.

**Example Usage**:

```bash
python tools/batch_confgen.py \
    --input molecules.sdf \
    --model checkpoint/LiTEN_FF_pretrained.pt \
    --n_conf 10 \
    --output conformers.sdf
```

**Features**:

* Automatically embeds and optimizes 3D structures.
* Can handle hundreds to thousands of molecules in batch.
* Parallelized for performance on multicore systems.

---

## 🧪 Requirements

These scripts require the following dependencies:

```bash
pip install ase rdkit torch numpy tqdm
```

For GPU acceleration, ensure that PyTorch is installed with CUDA support.

## 🙏 Acknowledgement

This work was supported by **Drug Design and Discovery Laboratory, Zhejiang University**, and inspired by previous developments in equivariant neural networks and deep potential models. We especially acknowledge the creators of **PaiNN**, **MACE**, and **VisNet**, which provided critical insights and technical foundations for our development of LiTEN and LiTEN-FF.

We also thank the maintainers of the [ASE](https://wiki.fysik.dtu.dk/ase/), [RDKit](https://www.rdkit.org/), and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) projects for their indispensable tools.

---

## 📖 Citation

If you use **LiTEN** or **LiTEN-FF** in your research or projects, please cite:

```bibtex
@article{LiTEN2025,
  title     = {LiTEN: A Foundational Equivariant Neural Potential for Biomolecular Simulations},
  author    = {Qun su},
  year      = {2025},
  url       = {https://github.com/lingcon01/LiTEN}
}
```

> 📌 The arXiv link will be updated upon public release of the paper.

---

## 👥 Contributors

* **Qun Su** ([@lingcon01](https://github.com/lingcon01)) – Lead developer, model architecture, training pipeline, downstream benchmarking
* **Kai Zhu and Jintu Zhang** – downstream benchmarking
* **Qiaolin Gou** – Optimization scripts, inference utilities, write articles

We welcome contributions from the community. If you'd like to contribute, please feel free to open issues or submit pull requests!


