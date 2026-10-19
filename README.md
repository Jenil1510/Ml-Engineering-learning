# 🚀 Machine Learning Engineering Foundations: Core Libraries & Tools

[![Open In Colab: NumPy](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jenil1510/Ml-Engineering-learning/blob/main/01_numpy_playground/numpy_playground.ipynb)
[![Open In Colab: EDA](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jenil1510/Ml-Engineering-learning/blob/main/02_exploratory_data_analysis/eda_bank_marketing.ipynb)

A comprehensive, production-grade guide, hands-on laboratory, and exploratory project covering the foundational libraries of the Machine Learning ecosystem: **Google Colab, NumPy, Pandas, Matplotlib, and Seaborn**.

---

## 📚 Ecosystem Deep Dive: When, Why & How Each Tool is Used

```
                         ┌──────────────────────────────────────────────┐
                         │              Google Colab                    │
                         │    (Cloud Compute, GPUs/TPUs, Zero-Setup)    │
                         └──────────────────────┬───────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
   ┌───────────────────────────┐                                 ┌───────────────────────────┐
   │          NumPy            │                                 │          Pandas           │
   │  Contiguous N-D Arrays,   │                                 │   DataFrames, Alignment,  │
   │  Vectorization, SIMD,     │                                 │   Cleaning, Aggregations, │
   │  Linear Algebra Engine    │                                 │   Relational Operations   │
   └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │           Matplotlib            │
                               │  Low-Level Object-Oriented Plot │
                               │  Engine & Figure Architecture   │
                               └────────────────┬────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │            Seaborn              │
                               │  High-Level Statistical Visuals │
                               │  Distributions, Heatmaps, Trends│
                               └─────────────────────────────────┘
```

### 1. 📓 Google Colab
* **Why it exists:** Provides browser-based Jupyter notebook execution hosted in Google Cloud, offering free access to hardware accelerators (NVIDIA T4/A100 GPUs and Google Cloud TPUs).
* **When to use:**
  - Prototyping deep learning models requiring GPU acceleration without local hardware investment.
  - Collaborative pair-programming, sharing interactive reproducible research, and portfolio demonstration.
  - Eliminating environment installation friction across distributed teams.
* **Key capabilities:** Native integration with Google Drive and GitHub, bash terminal commands with `!`, hardware profiling with `!nvidia-smi`, form widgets, and instant sharing.

### 2. 🔢 NumPy (`numpy`)
* **Why it exists:** C-Python loops are prohibitively slow for numerical data due to dynamic type dispatch and fragmented pointer structures. NumPy solves this by providing the **`ndarray`**: a contiguous, homogeneous memory buffer backed by compiled C and Fortran linear algebra libraries (BLAS/LAPACK).
* **When to use:**
  - Low-level mathematical modeling, matrix operations, custom algorithm implementations (loss functions, optimization routines).
  - High-throughput array transformations utilizing CPU SIMD (Single Instruction Multiple Data) registers.
  - Serves as the universal data interchange format for modern AI frameworks (PyTorch tensors and JAX arrays share NumPy's exact API design).
* **Key capabilities:** Multi-dimensional slicing, strided views, broadcasting, universal functions (`ufunc`), and fast pseudo-random generators (`PCG64`).

### 3. 🐼 Pandas (`pandas`)
* **Why it exists:** While NumPy excels at homogeneous matrix operations, real-world data is messy, tabular, and heterogeneous (mixing timestamps, text categories, floating-point metrics, and missing values). Pandas provides the **`DataFrame`** and **`Series`** abstractions with explicit index alignment.
* **When to use:**
  - Ingesting, parsing, and cleaning structured datasets (CSV, Parquet, SQL, JSON).
  - Reshaping tables via pivot tables, melting, grouping (`groupby`), merging, and time-series resampling.
  - Imputing missing data, detecting duplicates, and feature engineering prior to modeling.
* **Key capabilities:** Fast label-based slicing (`loc`/`iloc`), split-apply-combine aggregations, categorical memory optimization, and rolling window operations.

### 4. 📊 Matplotlib (`matplotlib`)
* **Why it exists:** The foundational, battle-tested 2D/3D plotting engine of the Python scientific stack. It implements an Object-Oriented architecture modeled after MATLAB.
* **When to use:**
  - Constructing custom multi-panel figures (`Figure` and `Axes` hierarchy).
  - Fine-grained control over every pixel, tick mark, annotation, legend, colorbar, and typography.
  - Generating publication-quality charts for technical reports and academic papers.
* **Key capabilities:** `plt.subplots`, line plots, scatter plots, custom patches, secondary axes (`twinx`), and export to vector formats (PDF, SVG).

### 5. 🎨 Seaborn (`seaborn`)
* **Why it exists:** Matplotlib requires dozens of lines of code to create sophisticated statistical figures. Seaborn sits on top of Matplotlib and integrates directly with Pandas DataFrames to generate complex statistical graphics with concise, semantic syntax.
* **When to use:**
  - Exploratory Data Analysis (EDA) to quickly uncover multivariate interactions.
  - Visualizing distributions (`histplot`, `kdeplot`, `ecdfplot`).
  - Analyzing category comparisons (`boxplot`, `violinplot`, `catplot`, `barplot`).
  - Correlation matrices and hierarchical clustering (`heatmap`, `clustermap`).
* **Key capabilities:** Built-in statistical estimation (confidence intervals, linear regression trendlines with `regplot`), facet grids (`pairplot`), and curated color palettes.

---

## 📂 Repository Structure

```
Ml-Engineering-learning/
│
├── README.md                                 # Master documentation and ecosystem guide
├── .gitignore                                # Environment and cache ignores
│
├── 01_numpy_playground/                      # Module 1: NumPy Mastery
│   ├── README.md                             # Memory layout, strides, SIMD & benchmark analysis
│   └── numpy_playground.ipynb                # Fully executed Colab-ready notebook
│
└── 02_exploratory_data_analysis/             # Module 2: Production-Grade EDA Project
    ├── README.md                             # Comprehensive EDA report, findings & recommendations
    ├── data/
    │   └── bank_additional_full.csv          # 41,188 rows x 21 columns dataset
    └── eda_bank_marketing.ipynb              # Fully executed Colab-ready EDA notebook
```

---

## 🛠️ Hands-on Deliverables

### 1. 🔢 Mini Exercise – NumPy Playground
* **Interactive Notebook:** [01_numpy_playground/numpy_playground.ipynb](file:///home/innvonix/Ml-Engineering-learning/01_numpy_playground/numpy_playground.ipynb)
* **Highlights:**
  - 1D, 2D, and 3D array creation, memory footprints, and stride mechanics.
  - Array indexing, multi-dimensional slicing, views vs copies (`shares_memory`), and boolean masking.
  - Mathematical aggregations, axis reductions, and the 3 rules of broadcasting.
  - Array reshaping, flattening (`ravel` vs `flatten`), transpositions, stacking, and splitting.
  - Modern random number generation with `np.random.default_rng` (PCG64).
  - **4 Real-World Numerical Problems with ZERO Python Loops:**
    1. *Pairwise Euclidean Distance Matrix* using 3D broadcasting.
    2. *Ordinary Least Squares (OLS) Linear Regression* from scratch via Normal Equation ($\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$).
    3. *2D Digital Image Convolution* using `sliding_window_view`.
    4. *Monte Carlo Simulation* of 10,000 asset paths (Geometric Brownian Motion) with VaR and CVaR calculations.
  - **Performance Benchmark:** Evaluated pure Python loops against vectorized NumPy across $N = 10^3$ to $N = 10^6$, proving a **~65x+ speedup**.

### 2. 🚀 Mini Project – Exploratory Data Analysis (EDA)
* **Interactive Notebook:** [02_exploratory_data_analysis/eda_bank_marketing.ipynb](file:///home/innvonix/Ml-Engineering-learning/02_exploratory_data_analysis/eda_bank_marketing.ipynb)
* **Dataset:** Portuguese Bank Marketing Campaign (UCI Machine Learning Repository).
  - **Volume:** **41,188 rows** and **21 columns**.
* **Highlights:**
  - Automated download pipeline with Google Colab compatibility.
  - Data inspection, deduplication (12 duplicates removed), and auditing of "unknown" survey values.
  - Feature engineering: Binary target mapping, previously contacted flag ($pdays \ne 999$), age cohorts, and campaign intensity brackets.
  - Multi-dimensional visualizations with Matplotlib & Seaborn (class imbalance, age distributions, job conversion, campaign fatigue, call duration KDEs, and interest rate regimes).
  - Synthesis of key observations and strategic recommendations for retail banking leadership.

---

## 💻 Local Setup & Execution Guide

To reproduce or extend this workspace locally:

```bash
# 1. Clone the repository
git clone https://github.com/Jenil1510/Ml-Engineering-learning.git
cd Ml-Engineering-learning

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip numpy pandas matplotlib seaborn jupyter nbformat

# 4. Launch JupyterLab to interact with the notebooks
jupyter lab
```
