# Compound Subgraph Definitions (`layer/subgraphs/`)

This directory holds self-contained subgraph implementations constructed with the LiteRT Tensor API and reference framework APIs.

## Canonical Subgraphs

Subgraphs are parameterized by the architectural dimensions of target models:

1. **Attention & Sequence Mixing Subgraphs (`attention/`):**
   - Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Sliding-Window Attention (SWA) with positional embeddings and KV-cache read/append across batch prefill (`T > 1`) and single-step decode (`T = 1`, varying context depth $S_{\text{kv}}$).
2. **Feed-Forward & MoE Subgraphs (`mlp/`, `moe/`):**
   - Dense gated MLP (`Norm -> Gate/Up Proj -> Activation -> Down Proj`).
   - Sparse Mixture-of-Experts (`Router Top-K -> Permute/Group -> Active Expert MLPs -> Weighted Combine`), measuring routing, gather/scatter, and active-expert memory bandwidth efficiency.
3. **Full Block Subgraphs (`block/`):**
   - Complete transformer / recurrent blocks combining attention or sequence mixing, state/KV updates, residual connections, normalization, and dense/MoE FFN to evaluate end-to-end block fusion and thread barrier overhead.
