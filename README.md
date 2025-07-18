# GCLSC: Single-cell clustering model based on graph contrast learning

## Introduction

In this study, we propose GCLSC (Graph Contrastive Learning for Single-Cell Clustering), a novel computational framework for the joint optimization of dimensionality reduction and clustering in scRNA-seq data analysis. The architecture of GCLSC integrates Graph Attention Networks (GAT) with Graph Transformers, resolving the fundamental challenge of synergistic local-global structure modeling. This architecture concurrently captures local micro-topologies (e.g., cell-cell interactions) and global macro-perspectives (e.g., developmental trajectories), establishing a robust and biologically interpretable computational paradigm for deep mining of single-cell omics data in the post-genomic era. To overcome limitations in discriminative power under high technical noise, we introduce contrastive learning leveraging graph edge features (e.g., gene co-expression relationships). This enhances cellular representation robustness against batch effects and dropout noise. To address heterogeneous density distributions across cell populations, we implement unsupervised clustering using Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN), enabling adaptive identification of rare cell types within developmental continua without predefined cluster specifications.

## Installation

The package can be installed by `git`. The testing setup involves a Windows operating system with 16GB of RAM, powered by an NVIDIA GeForce GTX 1050 Ti GPU and an Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz, running at a clock speed of 2.80 GHz.

### 1. Git clone from github

```
git clone https://github.com/JianjunTan-Beijing/GCLSC
cd ./GCLSC/
```

### 2. Utilize a virtual environment using Anaconda

You can set up the primary environment for GCLSC by using the following command:

```
conda env create -f environment.yml
conda activate GCSLC
```

## Running GCLSC

### 1. Data Preprocessing

For those who require swift data preparation, we offer a convenient Python script named preprocess_data.py located within the preprocess directory. This script, built on the foundation of Scanpy, streamlines the process of format transformation and preprocessing. It supports three types of input file formats: **H5AD, H5, and CSV data**. Throughout the preprocessing procedure, there are a total of five operations, encompassing cell-gene filtering, normalization, logarithmic transformation, scaling, and the selection of highly variable genes.

```python
# H5AD files
python preprocess/preprocess_data.py --input_h5ad_path=Path_to_input --save_h5ad_dir=Path_to_save --filter --norm --log --scale --select_hvg
# H5 files
python preprocess/preprocess_data.py --input_h5_path=Path_to_input --save_h5ad_dir=Path_to_save --filter --norm --log --scale --select_hvg
# CSV files
python preprocess/preprocess_data.py --count_csv_path=Path_to_input --label_csv_path=Path_to_input --save_h5ad_dir=Path_to_save --filter --norm --log --scale --select_hvg
```

### 2. Apply GCLSC

By utilizing the preprocessed input data, you have the option to invoke the subsequent script for executing the GCLSC method:

```python
python GCLSC.py --input_h5ad_path="data_name.h5ad" --epochs 100 --lr 1 --batch_size 512 --low_dim 256 --aug_prob 0.5 
```

In this context, we offer a collection of commonly employed GCLSC parameters for your reference. For additional details, you can execute `python GCLSC.py -h`.

**Note**: output files are saved in ./result, including `embeddings `, `ground truth labels `, `cluster results `, `KNN graph` and some `log files `.

## Running example

### 1. Collect Dataset.

Our sample dataset is stored in the directory "data/original/yan.h5".

### 2. Generate Preprocessed H5AD File.

```python
python preprocess/preprocess_data.py --input_h5_path="./data/original/yan.h5" --save_h5ad_dir="./data/preprocessed/" --filter --norm --log --scale --select_hvg
```

### 3. Apply GCLSC

```python
python GCLSC.py --input_h5ad_path="data/preprocessed/yan_preprocessed.h5ad" --epochs 100 --lr 1 --batch_size 512 --low_dim 256 --aug_prob 0.5
```
