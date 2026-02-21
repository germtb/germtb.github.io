---
title: "FourierLSH: When Better Asymptotics Lose to BLAS"
date: 2026-02-21
description: "I tried replacing random projections in LSH with FFT-based hashing. The theory was promising — O(L log d) instead of O(dL). In practice, BLAS matrix multiplication is just faster."
tags: ["algorithms", "similarity-search", "fft", "negative-results"]
math: true
---

I recently explored whether FFT-based structured hashing could replace random projections in Locality-Sensitive Hashing. The complexity looked promising — $O(L \log d)$ instead of $O(dL)$ — but the experiments told a different story. Here's what I tried, what I found, and why it didn't work.

## The idea

Random hyperplane LSH computes hash bits by projecting a vector $x \in \mathbb{R}^d$ onto random Gaussian vectors:

$$h_i(x) = \text{sgn}(\langle g_i, x \rangle)$$

Computing $L$ such projections requires a dense matrix multiply: $O(dL)$ operations and $O(dL)$ storage for the projection matrix. For modern embeddings ($d \geq 768$) and large bit budgets ($L \geq 512$), this is a lot of work.

The idea behind **FourierLSH** is simple: instead of projecting onto random vectors, apply random sign flips and take the FFT. Each FFT coefficient gives you hash bits, and the FFT costs only $O(d \log d)$ — amortized across all $d$ bits you extract.

## The method

Given a vector $x \in \mathbb{R}^d$:

1. **Random sign flip**: Draw $s \in \\{\pm 1\\}^d$ from a seeded RNG. Compute $y = s \odot x$. (Implemented as XOR on the IEEE 754 sign bit — no multiply needed.)

2. **FFT**: Compute $z = \text{rFFT}(y, n=d)$.

3. **Extract bits**: For each complex coefficient, take $\text{sgn}(\text{Re}(z_k))$ and $\text{sgn}(\text{Im}(z_k))$. One round gives ~$d$ bits.

4. **Multi-round encoding**: When you need $L > d$ bits, run $R = \lceil L/d \rceil$ rounds with independent sign flips from the same seed. Concatenate and pack into $\lceil L/8 \rceil$ bytes.

The total encoding cost is $O(L \log d)$, and you store only a single integer seed instead of an $L \times d$ projection matrix.

### Nice properties

- **Prefix safety**: The first $k$ bits of an $L$-bit hash are identical to a standalone $k$-bit hash with the same seed. This enables multi-resolution indexing without recomputation.
- **No projection matrix**: Just a seed. At scale with billions of vectors, this is a real storage win.
- **Recall matches standard LSH**: Empirically indistinguishable across all datasets I tested.

## The experiments

I compared FourierLSH against FAISS LSH (both using packed binary codes with SIMD Hamming search) and FAISS HNSW on three datasets:

- **Synthetic**: 5K random Gaussian vectors, 300D
- **GloVe**: 10K word embeddings, 300D
- **OpenAI text-embedding-3-small**: 50K embeddings, 1536D

### Results: OpenAI 1536D (50K vectors)

This is the most interesting dataset — high dimensional, real-world embeddings.

| **Recall@100** | 64 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| FourierLSH | 0.196 | 0.584 | 0.807 | 0.952 | 0.993 | 1.000 |
| FAISS LSH | 0.202 | 0.584 | 0.815 | 0.953 | 0.996 | 1.000 |

| **Query time (ms)** | 64 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| FourierLSH | 1,362 | 1,106 | 1,246 | 1,482 | 2,247 | 3,161 |
| FAISS LSH | **929** | **970** | **1,101** | **1,213** | **1,041** | **1,398** |
| *Slowdown* | *1.5x* | *1.1x* | *1.1x* | *1.2x* | *2.2x* | *2.3x* |

| **Build time (ms)** | 64 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| FourierLSH | 1,375 | 275 | 475 | 339 | 2,318 | 3,085 |
| FAISS LSH | **30** | **54** | **102** | **145** | **470** | **1,005** |

For reference, HNSW at ef=256 reaches 99.7% recall in 1,036ms — but needs 35s to build the index.

### Results: GloVe 300D (10K vectors)

Same story at lower dimensions:

| **Query time (ms)** | 256 | 1024 | 4096 |
|---|---|---|---|
| FourierLSH | 306 | 384 | 670 |
| FAISS LSH | **302** | **281** | **321** |
| *Slowdown* | *1.0x* | *1.4x* | *2.1x* |

Recall is effectively identical between the two methods at every bit width. FAISS is about 2x faster at high bit counts.

## Why FFT loses to matrix multiplication

The asymptotic advantage of $O(L \log d)$ vs $O(dL)$ is a factor of $d / \log d$. For $d = 1{,}536$, that's roughly 140x. So why is FourierLSH *slower*?

**BLAS is absurdly optimized.** Dense matrix multiplication has been the target of 40+ years of hardware-software co-optimization. OpenBLAS, MKL, and Accelerate achieve near-peak FLOP/s through cache-aware tiling, fused multiply-add pipelines, and vectorized memory access. FFT implementations, while efficient, haven't received the same degree of tuning for these problem sizes.

**Memory access patterns.** Matrix multiplication has regular, sequential memory access — cache-friendly by design. The FFT butterfly pattern requires non-sequential access with strides that change at each level, leading to more cache misses.

**Arithmetic intensity.** For matrix multiply, the ratio of FLOPs to memory accesses is $O(n)$ — highly compute-bound. FFT has $O(\log n)$ — more memory-bound and harder to pipeline.

**The constant factors win.** At $d \sim 10^3$, the constant-factor overhead of FFT vs BLAS easily eats the 140x asymptotic advantage. You'd need dimensions in the millions for the asymptotics to dominate — and nobody has embeddings that large.

## What I learned

1. **Asymptotic complexity is not destiny.** This is well-known but still easy to forget when the theory looks clean. The $O(d / \log d)$ advantage is real mathematically, but BLAS is a force of nature.

2. **The recall works.** FourierLSH produces hash bits that are empirically indistinguishable from random hyperplane LSH. The FFT with random sign flips does genuinely mix the input well enough for similarity-preserving hashing. The problem is purely one of speed, not quality.

3. **Storage is the one real advantage.** No projection matrix — just a seed. For systems with billions of vectors and large bit budgets, this could matter. But it's not enough to justify slower encoding and search.

4. **When might this actually win?** Possibly on hardware with efficient FFT units (DSP accelerators), at extreme dimensions where the asymptotics finally kick in, or in settings where projection matrix storage is genuinely prohibitive. I didn't test any of these.

## Code

The full implementation (Python + Rust SIMD Hamming kernel) and benchmarks are at [github.com/germtb/FourierLSH](https://github.com/germtb/FourierLSH).
