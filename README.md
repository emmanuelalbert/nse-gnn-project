Market Shock Propagation in NSE Stocks via Temporal Graph Attention Networks (TGAT)

A deep-learning research pipeline that models how financial shocks propagate across Nifty 50 stocks during market crises, using a Temporal Graph Attention Network (TGAT) architecture. This work adapts and extends the methodology of Bentaleb et al. (2024, IEEE Access), which applied a Temporal Graph Convolutional Network (TGCN) to the S&P 500, translating it to the Indian Nifty 50 universe with a TGAT-based approach.

Overview

Stock markets are interconnected systems where shocks in one company or sector can ripple outward and affect others. This project builds a graph-based deep learning pipeline to:

Detect historical shock periods in Nifty 50 stock price data
Construct causal relationships between stocks using Granger causality
Train a TGAT model to learn and predict dynamic inter-stock relationships during shocks
Compare network structure before vs. after training to understand propagation patterns
Analyze which companies and sectors are most affected, and in what order
Crisis periods studied
2008 Global Financial Crisis (GFC)
2018 IL&FS Crisis
2020 COVID-19 onset
Pipeline Architecture

The pipeline runs end-to-end across ten stages:

Data ingestion — Nifty 50 closing price data
Preprocessing — cleaning, missing value imputation
Shock detection — identifying statistically significant shock windows
Granger causality — constructing directed causal graphs between stocks
Dataset construction — building temporal graph datasets for training
TGAT training — learning dynamic node embeddings and edge relationships
Post-training adjacency reconstruction — deriving updated relationship graphs
Network analysis — degree, centrality, density, clustering metrics
Cross-shock comparison — comparing propagation patterns across crises
Visualization — network graphs, heatmaps, and summary plots

The pipeline CLI supports staged, checkpointed execution via --stage and --resume-from-fold flags, and uses growing-window (expanding) chronological Monte Carlo splits for evaluation.

Model Details
Architecture: Custom TGAT — 2 layers, 4 attention heads, hidden dimension 64 (~76K parameters)
Loss: BCEWithLogitsLoss
Optimizer: Adam
Framework: PyTorch (CUDA-enabled)
Tech Stack
Category	Tools
ML / Deep Learning	PyTorch (CUDA), custom TGAT implementation
Data	yfinance, pandas, pyarrow/parquet
Graph & Statistics	NetworkX, Pyvis, statsmodels (Granger causality via ssr_chi2test)
Visualization	Matplotlib, Pyvis
Reporting	Markdown research report, PptxGenJS-generated presentation
Project Structure
nse-gnn-project/
├── comparison/            # Cross-shock comparison logic
├── config/                # Pipeline configuration
├── data/
│   └── raw/                # Raw ingested price data
├── evaluation/             # AUC-ROC, AUPR, F1 evaluation metrics
├── graph/                  # Graph/dataset construction
├── model/                  # TGAT architecture
├── network_analysis/       # Degree, centrality, density, clustering metrics
├── notebooks/               # Exploratory / analysis notebooks
├── preprocessing/           # Data cleaning, imputation
├── results/
│   ├── adjacency/
│   │   ├── pre_training/
│   │   └── post_training/
│   ├── checkpoints/
│   │   └── fold_01/
│   ├── figures/
│   └── metrics/
├── shock_detection/         # Shock period identification (Granger causality inputs)
├── tests/
├── training/                 # Training loop, Monte Carlo split logic
├── visualization/             # Network plots and heatmaps
├── requirements.txt
└── README.md

Note: venv/, __pycache__/, and generated results/ artifacts (checkpoints, figures, metrics) are excluded from version control via .gitignore. Only code and configuration are tracked — see Setup for regenerating results locally.

Setup
bash
git clone https://github.com/emmanuelalbert/nse-gnn-project.git
cd nse-gnn-project
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
Usage
bash
python run_pipeline.py --stage all

Run a specific stage:

bash
python run_pipeline.py --stage train

Resume training from a specific fold:

bash
python run_pipeline.py --stage train --resume-from-fold 3
