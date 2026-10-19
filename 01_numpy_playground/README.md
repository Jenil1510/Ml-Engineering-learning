# 🔢 01 - NumPy Playground: Vectorized Computing & Internal Architecture

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jenil1510/Ml-Engineering-learning/blob/main/01_numpy_playground/numpy_playground.ipynb)

This module provides a deep exploration of **NumPy**, the bedrock library of modern numerical computing, data science, and deep learning.

---

## 📋 What is Covered in this Playground

1. **The NumPy Internal Mental Model**
   - Memory layout: C-contiguous vs Fortran-contiguous buffers.
   - Strides, data types (`dtype`), item sizes, and pointers.
   - Why Python lists suffer from cache misses and pointer-chasing overhead.
2. **Array Creation & Constructors**
   - 1D, 2D (matrices), and 3D (tensors) array initialization.
   - Linear and logarithmic generators: `np.arange`, `np.linspace`, `np.logspace`.
   - Special initializers: `np.zeros`, `np.ones`, `np.eye`, `np.full`.
3. **Indexing, Slicing, Views & Copies**
   - Basic slicing syntax (`start:stop:step`) creating zero-copy views.
   - Checking memory sharing with `np.shares_memory()`.
   - Multi-dimensional sub-matrix selection.
   - Boolean masking and vectorized ternary conditioning (`np.where`).
   - Integer array (fancy) indexing and coordinate extraction (`np.ix_`).
4. **Mathematical & Statistical Operations**
   - Element-wise arithmetic via Universal Functions (`ufunc`).
   - **Broadcasting Mechanics:** Step-by-step alignment of trailing dimensions.
   - Reductions across dimensions (`axis=0`, `axis=1`, `keepdims=True`).
   - Vectorized Feature Standardization ($Z$-score normalization).
5. **Array Reshaping & Structural Transformations**
   - `reshape()` with automatic dimension inference (`-1`).
   - `ravel()` (view) vs `flatten()` (copy).
   - Transposing tensors and reordering spatial dimensions (`np.transpose`, `np.swapaxes`).
   - Stacking and splitting: `np.vstack`, `np.hstack`, `np.vsplit`, `np.hsplit`.
6. **Modern Random Number Generation (`default_rng`)**
   - Superior statistical properties of the **PCG64** pseudo-random generator.
   - Sampling from Uniform, Gaussian/Normal, Poisson, and Binomial distributions.
   - Shuffling and permutations for mini-batch stochastic gradient descent.
7. **Real-World Numerical Problems (Zero Explicit Python Loops)**
   - **Problem 1:** Pairwise Euclidean Distance Matrix using broadcasting $(N \times 1 \times D) - (1 \times M \times D)$.
   - **Problem 2:** Ordinary Least Squares (OLS) Linear Regression from Scratch using the Normal Equation $\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$.
   - **Problem 3:** 2D Digital Image Convolution with Gaussian Kernel using `sliding_window_view`.
   - **Problem 4:** Monte Carlo Simulation of 10,000 Asset Price Trajectories (Geometric Brownian Motion) with Value at Risk (VaR) and Expected Shortfall (CVaR).
8. **Performance Benchmark: Vectorized NumPy vs Pure Python Loops**
   - Polynomial evaluation benchmark across $N = 10^3, 10^4, 10^5, 5 \times 10^5, 10^6$.
   - Speedup analysis demonstrating **30x to 150x+** acceleration.
   - Detailed breakdown of SIMD hardware instructions (AVX-512, AVX2, ARM NEON) and CPU L1/L2 cache locality.

---

## 🚀 How to Run

### Option 1: Run in Google Colab (Recommended)
Click the badge above or upload [numpy_playground.ipynb](file:///home/innvonix/Ml-Engineering-learning/01_numpy_playground/numpy_playground.ipynb) to [Google Colab](https://colab.research.google.com). No local setup or dependencies required.

### Option 2: Run Locally
Activate the virtual environment and launch Jupyter:
```bash
# From workspace root
source .venv/bin/activate
jupyter lab 01_numpy_playground/numpy_playground.ipynb
```

---

## 📊 Summary of Benchmark Results

| Array Size ($N$) | Pure Python (ms) | NumPy Vectorized (ms) | Speedup Multiplier |
| :--- | :--- | :--- | :--- |
| **1,000** | ~0.15 ms | ~0.006 ms | **~25x** |
| **10,000** | ~1.45 ms | ~0.035 ms | **~41x** |
| **100,000** | ~14.8 ms | ~0.24 ms | **~62x** |
| **500,000** | ~75.2 ms | ~1.18 ms | **~64x** |
| **1,000,000** | ~152.0 ms | ~2.35 ms | **~65x+** |

### Key Takeaway for Machine Learning
Deep learning frameworks (PyTorch, JAX, TensorFlow) build directly on NumPy's vectorized tensor semantics. Using Python loops in data loading or model forward/backward passes starves the GPU of data. Vectorization is not merely an optimization; it is a foundational requirement for modern scalable machine learning.
