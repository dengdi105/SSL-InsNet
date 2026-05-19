# SSL-InsNet: a Sequence Structured Labeling Framework for Large-Scale Genomic Insertion Detection Leveraging Time-Distributed Dual Transformer

<img width="1119" height="421" alt="SSL-InsNet Architecture" src="https://raw.githubusercontent.com/dengdi105/SSL-InsNet/main/imgs/main_plot_new.png" />

<div align="center">
  
**Overall architecture of SSL-InsNet with time-distributed dual transformer and dynamic gated-sparse attention.**

</div>

# 📖 Overview

**SSL-InsNet** (Sequence Structured Labeling Framework) is a novel deep learning framework for large-scale genomic insertion variant detection in third-generation sequencing data. The framework integrates semantic nucleotide features and syntactic alignment signatures via a dual-stream time-distributed architecture, significantly enhancing insertion calling precision while maintaining high computational scaling performance.

# ✨ Key Features

🔀 Dual-Modal Sequence Labeling: Simultaneously encodes nucleotide sequence identities (semantics) and structural alignment metrics (syntactics)

⏳ Time-Distributed Contextualization: Processes chromosome-scale features in optimized sub-temporal windows to drastically reduce memory overhead

🛡️ Dynamic Gated-Sparse Attention (DGSA): Cross-modal attention fusion for robust signal filtering and background noise suppression

🚀 Native Multi-GPU Acceleration: Highly optimized data parallel processing for fast chromosome-wide execution

📊 Standardized VCF Exporter: Automatically generates production-ready VCF v4.2 callsets with comprehensive read-support annotations

# 🚀 Quick Start

## Prerequisites

- Python 3.9+
- CUDA 12.4+ (for GPU acceleration)
- 11+ GB VRAM recommended for inference
- 16+ GB RAM

## Installation

### Create a virtual environment
conda create -n ssl_insnet python=3.9.1 -y
conda activate ssl_insnet

### Install core deep learning dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pandas==2.2.3 numpy==1.26.4

### Install specialized bioinformatics packages
pip install pysam==0.23.0

## Essential Bioinformatics Dependencies

| Package | Purpose |
|---------|---------|
| ![pysam](https://img.shields.io/badge/pysam-0.23.0-14a2b8?logo=python&logoColor=white) | BAM/CRAM file streaming and header parsing |
| ![pandas](https://img.shields.io/badge/pandas-2.2.3-14a2b8?logo=pandas&logoColor=white) | Feature tracking and matrix metadata handling |
| ![numpy](https://img.shields.io/badge/numpy-1.26.4-14a2b8?logo=numpy&logoColor=white) | Vectorized array serialization and processing |

# 📁 Data Preparation & Calling Pipeline

## 1. Produce Data for Call SV (`generate_feature`)

Extract alignment signatures and raw genomic tracks into chunked feature matrices (`.npy`) across targeted genomic spans.

python SSL_InsNet.py generate_feature

--bam_file /path/to/your/HG002_PB_5x_RG_HP10XtrioRTG.bam

--output_path ./features_dir

--contigs_list [12,13]

--max_worker 5

--vcf_file /path/to/your/HG002_SVs_Tier1_v0.6.vcf.gz

Notes: Pass empty list [] to contigs_list to automatically use all chromosomes.


## 2. Call Insertion (`call_insertion`)

Stream the generated sequence window blocks through the Time-Distributed network to perform candidate merging and final insertion calling.

python SSL_InsNet.py call_insertion

--gpu_name '0,1'

--save_length 10000000

--timesteps 100

--ins_predict_weight ./weights/SSL_InsNet_best.pth

--data_path ./features_dir

--bam_file /path/to/your/HG002_PB_10x_RG_HP10XtrioRTG.bam

--out_vcf_file ./calls/output_variants.vcf

--contigs [12,13,14,15,16,17,18,19,20,21,22]

--support 5


## Key Executive Parameters:

| Parameter  | Description                        | Default               |
|------------|------------------------------------|-----------------------|
| `--gpu_name` | GPU device indices (e.g., '0' or '0,1') | '0'                   |
| `--save_length` | Basepair sequence length spanned per chunk | 10000000             |
| `--timesteps` | Time step dimension for the recurrent layer | 100                   |
| `--support` | Minimum read support to retain an insertion | 5                     |

# 🏗️ Architecture Overview

<img width="70%" alt="DGSA Module" src="https://raw.githubusercontent.com/dengdi105/SSL-InsNet/main/imgs/DGSA_new2.png" />

Input:
├── Semantic Stream (BASES) → Feature Expansion → Time-Distributed Backbone
└── Syntactic Stream (CIGAR) → Structural Mapping → Time-Distributed Backbone

Core Module:
└── Dynamic Gated-Sparse Attention (DGSA)
├── Cross-Modal Dual Attention Fusion
├── Confidence-Gated Residual Recalibration
└── Sparse Context Selection

Output: Standard Insertion Variant Calling (VCF v4.2)


# 📊 Tested Data

## HG002 CLR data
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/Baylor_NGMLR_bam_GRCh37/HG002_PB_70x_RG_HP10XtrioRTG.bam


## HG002 ONT data
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/UCSC_Ultralong_OxfordNanopore_Promethion/HG002_GRCh37_ONT-UL_UCSC_20200508.phased.bam


## HG002 CCS data
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_CCS_15kb/alignment/HG002.Sequel.15kb.pbmm2.hs37d5.whatshap.haplotag.RTG.10x.trio.bam
