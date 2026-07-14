# Distributed MCR²

Official research implementation accompanying the IEEE Transactions on Signal Processing paper:

> **Semantic-based Distributed Learning for Diverse and Discriminative Representations**  
> Zhuojun Tian, Chaouki Ben Issaid, and Mehdi Bennis, 2026.

This repository studies distributed representation learning with the maximal coding rate reduction (MCR²) objective under IID and non-IID client data distributions.

## Repository status

The repository is packaged with a standard `src/` layout and can be installed with `pip`. CUDA is optional: `--device auto` uses a GPU when available and otherwise falls back to CPU. The original research model, loss, client, and optimization logic are preserved.

## Project structure

```text
.
├── pyproject.toml             # Package metadata and console commands
├── src/distributed_mcr2/      # Installable Python package
│   ├── dataset/               # Dataset loaders
│   ├── data/                  # Packaged client partition tables
│   ├── resources/             # Packaged adjacency matrices
│   ├── train_iid.py           # IID training implementation
│   └── train_niid.py          # Non-IID training implementation
├── scripts/                   # Shell launchers
├── tests/                     # Basic package smoke tests
```

## Data partition files

The files in `data/` describe which classes and how many samples are assigned to each client.

| Experiment | Clients | Partition files |
|---|---:|---|
| IID setting | 10 | `CategoryToClients10.txt`, `LocalDist_iid10.txt`, `LocalDist_iid_test10.txt` |
| First non-IID setting | 4 | `CategoryToClients4.txt`, `LocalDist_niid4.txt`, `LocalDist_niid_test4.txt` |
| Second non-IID setting | 5 | `CategoryToClients5_4.txt`, `LocalDist_niid5_4.txt`, `LocalDist_niid_test5_4.txt` |

The current non-IID entry script is hard-coded to the 4-client configuration.

## Requirements

Recommended environment:

- Linux
- Python 3.9 or 3.10
- NVIDIA GPU with a CUDA-compatible PyTorch installation
- Conda or Miniconda

The core Python dependencies are listed in `requirements.txt`. A reproducible Conda environment definition is provided in `environment.yml`.

## Installation

### Install with pip

From the repository root:

```bash
python -m pip install --upgrade pip
pip install .
```

For development, use an editable install:

```bash
pip install -e .
```

For a CUDA environment, install the PyTorch build compatible with the local CUDA driver before installing this project.

### Conda installation

```bash
conda env create -f environment.yml
conda activate distributed-mcr2
pip install -e .
```

## Packaged experiment resources

The maintained package includes the supplied adjacency matrices and client partition tables. Default console commands locate these resources inside the installed package. A custom matrix can be supplied with `--adjacency-file /path/to/matrix.txt`.

The included `4_adj_matrix.txt` was derived in stage 2 from the leading 4×4 block of the supplied 5×5 matrix. It is dimensionally compatible with the active four-client non-IID script, but the intended research topology should still be confirmed before reproducing paper results.

## Running the experiments

After installation, the preferred commands are:

```bash
distributed-mcr2-iid --help
distributed-mcr2-niid --help
```

Example short smoke runs:

```bash
distributed-mcr2-iid --epochs 1 --save-every 1 --device auto
distributed-mcr2-niid --epochs 1 --save-every 1 --device auto
```

Equivalent module commands are:

```bash
python -m distributed_mcr2.train_iid
python -m distributed_mcr2.train_niid
```

Shell wrappers are also provided:

```bash
./scripts/run_iid.sh --help
./scripts/run_niid.sh --help
```

## Dataset download

The CIFAR-10 loader uses `torchvision.datasets.CIFAR10(..., download=True)`. On the first run, CIFAR-10 is downloaded to the directory specified by `--data-dir`, whose default value is:

```text
./Datasets_Sim/
```

Internet access is therefore required for the first run unless the dataset is already available in that directory.

## Outputs

Training outputs are written under `Exp_results/` by the current entry scripts. Depending on the experiment, outputs may include:

- per-client loss text files;
- averaged evaluation results;
- correlation visualizations;
- model checkpoints under a `models/` subdirectory.

Generated datasets, results, checkpoints, caches, and local environments are excluded through `.gitignore`.

## Important implementation notes

1. `--device auto` selects CUDA when available and otherwise uses CPU; full training is expected to be much faster on a GPU.
2. Dataset and output paths are resolved relative to the current working directory; packaged experiment resources are resolved from the installation.
3. CIFAR-10 is the actively maintained experiment path. Other dataset modules are retained but are not exposed as dedicated console commands.
4. The default schedule is 1,000 epochs and may require substantial compute time.
5. The four-client non-IID adjacency matrix is a maintenance-derived compatibility resource and should be scientifically verified before result reproduction.

## Basic repository check

You can verify that all Python files are syntactically valid without starting training:

```bash
python -m compileall .
```

This check does not verify the missing adjacency matrices, GPU compatibility, data download, or numerical correctness.

## Reproducibility checklist

Before attempting to reproduce results:

- use a CUDA-enabled PyTorch installation;
- restore the correct adjacency matrix files;
- run commands from the repository root;
- verify the client count and adjacency matrix dimensions;
- record Python, PyTorch, torchvision, CUDA, and GPU versions;
- retain the random seed used by the experiment;
- confirm output directories have sufficient disk space.

## Citation

```bibtex
@ARTICLE{tian2026semantic,
  author={Tian, Zhuojun and Issaid, Chaouki Ben and Bennis, Mehdi},
  journal={IEEE Transactions on Signal Processing},
  title={Semantic-based Distributed Learning for Diverse and Discriminative Representations},
  year={2026},
  pages={1-16},
  doi={10.1109/TSP.2026.3702222}
}
```

## License

The supplied source archive did not include an open-source license. A conservative `LICENSE` notice has therefore been added that keeps all rights reserved. Before publishing the repository as open source, the copyright holders should explicitly select and approve an open-source license such as MIT, BSD-3-Clause, or Apache-2.0.

## Stage 2 maintenance changes

The second maintenance stage improves reliability while retaining the original
training objective and model definitions:

- experiment parameters are no longer overwritten after argument parsing;
- `--device auto` selects CUDA when available and otherwise uses CPU;
- dataset, result, partition, and adjacency paths are resolved relative to the
  repository rather than the current shell directory;
- adjacency matrices are checked before training for existence and dimensions;
- output directories are created automatically;
- learning rate, local epochs, save interval, client count, and adjacency file
  are exposed as command-line options;
- unused imports and several inconsistent path expressions were removed.


### Updated commands

IID defaults:

```bash
python main_colMCR_iid.py --device auto --epochs 1000
```

Non-IID defaults:

```bash
python main_colMCR_niid.py --device auto --epochs 1000
```

A quick argument and import check can be run without downloading data:

```bash
python main_colMCR_iid.py --help
python main_colMCR_niid.py --help
python -m compileall .
```
