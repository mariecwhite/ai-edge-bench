# Layer Benchmark Suites (`layer/suites/`)

Suite configurations in this directory define the subgraph shapes, concurrency settings (thread counts, GPU streams/command buffers), and precision tracks evaluated at the Layer tier.

## Suite Structure

Each suite JSON specifies:
- `subgraph`: Target subgraph family (`gqa`, `mlp`, `moe_block`, `transformer_block`) and architectural hyperparameters (`hidden_size`, `intermediate_size`, `num_heads`, `num_kv_heads`, `head_dim`, `num_experts`, `top_k_experts`).
- `regimes`:
  - **Prefill (`T > 1`):** Batch token lengths (`64`, `256`, `1024`, `4096`) testing compute-bound tile fusion.
  - **Decode (`T = 1`):** Single-token step across KV context depths (`S_kv = 64..128K`) testing memory-bound weight + KV streaming and multi-thread scaling.
- `quality_gate`: Maximum allowable relative $L_2$ error, minimum cosine similarity, and maximum KL-divergence against the `FP32` reference subgraph.
