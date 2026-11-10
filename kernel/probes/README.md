# Hardware Ceiling Software Probes (`kernel/probes/`)

Software probes empirically measure the practical hardware ceilings used as denominators in the [Kernel](../README.md), [Layer](../../layer/README.md), and [Model](../../model/README.md) roofline formulas.

## Available Probes

- [cpu_ceilings.c](cpu_ceilings.c) — Repeatable x86_64 / Intel Sapphire Rapids hardware ceiling probe measuring both single-core (`1T`, physical core `0`) and multi-core (`28T`, NUMA Node 0 physical cores `0..27`) compute and memory-hierarchy ceilings:
  - **Compute Ceilings ($C_{\text{prac}}(\text{op}, \text{dtype})$)**:
    - `amx_int8_tdpbssd` (`tdpbssd` `16×16×64`, `32,768` INT8 OPs/inst)
    - `amx_bf16_tdpbf16ps` (`tdpbf16ps` `16×16×32`, `16,384` BF16 FLOPs/inst)
    - `avx512_vnni_int8_vpdpbusd` (`vpdpbusd %zmm`, `128` INT8 OPs/inst)
    - `avx_vnni_256_int8_vpdpbusd` (`vpdpbusd %ymm`, `64` INT8 OPs/inst)
    - `avx512_fp32_fma` (`vfmadd231ps %zmm`, `32` FP32 FLOPs/inst)
  - **Memory Hierarchy Bandwidth Ceilings ($BW_{\text{prac}}(\text{WorkingSet})$)**:
    - **L1D Cache** (`16 KB`/core): Streaming `vmovaps` Read (`6,691.22 GB/s` 28C) and Read+Write Copy (`5,560.20 GB/s` 28C)
    - **L2 Cache** (`512 KB`/core): Streaming `vmovaps` Read (`4,473.83 GB/s` 28C) and Read+Write Copy (`2,216.39 GB/s` 28C)
    - **Shared L3 LLC** (`32 MB` shared pool, `8 MB`/thread window): Streaming Read (`780.35 GB/s` 28C) and Read+Write Copy (`535.86 GB/s` 28C)
    - **DDR5 DRAM** (`896 MB` across 28 physical cores on Socket 0): Streaming Read (`246.60 GB/s` 28C), Read+Write Copy (`134.61 GB/s` 28C), and STREAM Triad (`151.58 GB/s` 28C)

---

## How the Probes Are Computed

### 1. Compute Ceiling Probes ($C_{\text{prac}}(\text{op}, \text{dtype})$)

Each compute probe measures the maximum sustained arithmetic throughput of a target instruction family by executing an unrolled inline-assembly loop operating **purely on architectural registers** (zero memory loads or stores inside the timed region):

1. **Zero-Memory In-Register Setup**: Before entering the timed loop, each thread zeroes its tile (`_tile_zero(0..7)`) or vector (`vpxord` / `vxorps`) registers. For AMX probes, the thread first requests OS `XFEATURE_XTILEDATA` permission via `syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 18)` and loads a `palette_id = 1` configuration with eight `16 rows × 64 bytes` tiles (`_tile_loadconfig`).
2. **Independent Accumulators to Hide Pipeline Latency**: To prevent Read-After-Write (RAW) accumulator hazards from stalling the execution pipes, each loop iteration issues instructions across multiple independent destination registers:
   - **AMX (`tdpbssd` / `tdpbf16ps`)**: Uses `tmm6` and `tmm7` as source operands and accumulates into **6 independent destination tiles (`tmm0..tmm5`)**, unrolled $2\times$ per loop body (**12 tile instructions per iteration**).
   - **AVX-512 / AVX-VNNI (`vpdpbusd` / `vfmadd231ps`)**: Uses `zmm12`/`zmm13` (or `ymm12`/`ymm13`) as source operands and accumulates into **12 independent destination registers (`zmm0..zmm11` or `ymm0..ymm11`)**, unrolled $2\times$ per loop body (**24 vector instructions per iteration**).
3. **Throughput Formula**: Following the standard convention that 1 Multiply-Accumulate (MAC) = **2 algorithmic operations** (1 multiply + 1 add), the achieved throughput for a timed trial of wall-clock duration $\Delta t = t_1 - t_0$ seconds is:

$$C_{\text{prac}} = \frac{N_{\text{threads}} \times N_{\text{iters}} \times \text{Ops}_{\text{iter}}}{\Delta t \times 10^9} \quad [\text{GOPS or GFLOPS}]$$

where $\text{Ops}_{\text{iter}} = N_{\text{inst/iter}} \times \text{MACs}_{\text{inst}} \times 2$:

| Probe Key | Instruction | Tile / Vector Geometry | MACs / Inst | OPs / Inst ($2 \times \text{MACs}$) | Inst / Iter | $\text{Ops}_{\text{iter}}$ | $N_{\text{iters}}$ / Trial |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `amx_int8_tdpbssd` | `tdpbssd %tmm6, %tmm7, %tmm0..5` | $16 \times 16 \times 64$ (`u8×s8→s32`) | $16{,}384$ | $32{,}768\text{ OPs}$ | $12$ | $393{,}216$ | $2{,}000{,}000$ |
| `amx_bf16_tdpbf16ps` | `tdpbf16ps %tmm6, %tmm7, %tmm0..5` | $16 \times 16 \times 32$ (`bf16×bf16→fp32`) | $8{,}192$ | $16{,}384\text{ FLOPs}$ | $12$ | $196{,}608$ | $2{,}000{,}000$ |
| `avx512_vnni_int8_vpdpbusd` | `vpdpbusd %zmm12, %zmm13, %zmm0..11` | $16\text{ lanes} \times 4\text{ bytes}$ (`512-bit`) | $64$ | $128\text{ OPs}$ | $24$ | $3{,}072$ | $16{,}000{,}000$ |
| `avx_vnni_256_int8_vpdpbusd` | `vpdpbusd %ymm12, %ymm13, %ymm0..11` | $8\text{ lanes} \times 4\text{ bytes}$ (`256-bit`) | $32$ | $64\text{ OPs}$ | $24$ | $1{,}536$ | $16{,}000{,}000$ |
| `avx512_fp32_fma` | `vfmadd231ps %zmm12, %zmm13, %zmm0..11` | $16\text{ lanes}$ (`512-bit FP32`) | $16$ | $32\text{ FLOPs}$ | $24$ | $768$ | $16{,}000{,}000$ |

---

### 2. Memory-Hierarchy Bandwidth Probes ($BW_{\text{prac}}(\text{WorkingSet})$)

Each bandwidth probe streams contiguous, 64-byte cache-line-aligned FP32 buffers (`aligned_alloc(64, ...)`) sized to reside strictly inside a target level of the Sapphire Rapids memory hierarchy (`48 KB` L1D/core, `2 MB` L2/core, `52.5 MB` shared L3 LLC, and Socket 0 DDR5 DRAM). Buffers are allocated and initialized with normal FP32 constants (`1.0f`, `2.0f`, `0.5f`) inside the pinned OpenMP parallel region to guarantee local NUMA first-touch placement and eliminate page-fault or subnormal traps during timing.

#### Access Patterns & Traffic Multiplier (`traffic_mult`)

Every inner loop processes `64` `float` elements (`256 bytes` = `4 cache lines`) per step using 4-way unrolled 512-bit AVX-512 intrinsics:

1. **Streaming Read (`read_only`, `traffic_mult = 1.0`)**:
   Issues four `_mm512_load_ps` (`vmovaps`) instructions per step, accumulating into four independent registers (`acc0..acc3`) pinned via `__asm__ volatile("" : "+v"(...))` so the compiler cannot dead-code-eliminate the reads. Counts **$1 \times \text{Bytes}_{\text{window}}$** per pass.
2. **Read + Write Copy (`read_write_copy`, `traffic_mult = 2.0`)**:
   Issues four `_mm512_load_ps(a + i)` loads followed by four `_mm512_store_ps(b + i, v)` caching stores (`vmovaps`). Counts logical bytes read plus bytes written (**$2 \times \text{Bytes}_{\text{window}}$** per pass, intentionally reflecting hardware write-allocate / Read-For-Ownership overhead experienced by standard kernel stores).
3. **STREAM Triad (`stream_triad`, `traffic_mult = 3.0`)**:
   Computes `c[i] = 0.5f * a[i] + b[i]` via eight `_mm512_load_ps` (`a` and `b`), four `_mm512_fmadd_ps`, and four `_mm512_store_ps` (`c`). Counts 2 reads + 1 write (**$3 \times \text{Bytes}_{\text{window}}$** per pass).

#### Working-Set Sizing Across Cache Levels

| Hierarchy Level (Capacity) | Mode | 1-Core (`1T`) Buffer Size | 28-Core (`28T`) Buffer Size | $N_{\text{reps}}$ (`1T` / `28T`) | Residency Rationale |
| :--- | :--- | :--- | :--- | ---: | :--- |
| **L1D Cache** (`48 KB`/core) | `read_only`<br>`read_write_copy` | `16 KB` (`a`)<br>`8 KB` × 2 (`16 KB` total) | `16 KB`/thread<br>`8 KB` × 2/thread (`16 KB`/T) | `150,000` / `150,000` | `16 KB` total footprint per core fits well inside the `48 KB` private L1D cache. |
| **L2 Cache** (`2 MB`/core) | `read_only`<br>`read_write_copy` | `512 KB` (`a`)<br>`256 KB` × 2 (`512 KB` total) | `512 KB`/thread<br>`256 KB` × 2/thread (`512 KB`/T) | `8,000` / `8,000` | `512 KB` per core exceeds `48 KB` L1D ($>10\times$) while fitting comfortably inside the `2 MB` private L2 cache. |
| **Shared L3 LLC** (`52.5 MB` Socket 0) | `read_only`<br>`read_write_copy` | `16 MB` (`a`)<br>`8 MB` × 2 (`16 MB` total) | `32 MB` shared pool (`8 MB` window/T)<br>`32 MB` × 2 shared pools (`8 MB` window/T) | `300` / `200` | **1T** uses a `16 MB` private buffer (`> 2 MB` L2, `< 52.5 MB` L3). **28T** (`run_shared_l3_bw_probe`) allocates a shared `32 MB` pool (`< 52.5 MB` L3) where each thread `tid` streams an `8 MB` window (`> 2 MB` private L2) at offset `(tid × 512 KB) % 24 MB`, preventing 28-thread aggregate buffers from spilling into DRAM. |
| **DDR5 DRAM** (Socket 0) | `read_only`<br>`read_write_copy`<br>`stream_triad` | `256 MB` (`a`)<br>`128 MB` × 2 (`256 MB` total)<br>`128 MB` × 3 (`384 MB` total) | `32 MB`/T (`896 MB` total)<br>`16 MB` × 2/T (`896 MB` total)<br>`16 MB` × 3/T (`1,344 MB` total) | `30` / `40` | Working set (`256–1,344 MB`) is $5\times\text{–}25\times$ larger than the `52.5 MB` L3 LLC, forcing sustained DDR5 memory-channel streaming. |

#### Bandwidth Formula

For each timed trial of wall-clock duration $\Delta t = t_1 - t_0$ seconds, sustained bandwidth in decimal gigabytes per second ($\text{GB/s} = 10^9\text{ Bytes/s}$) is computed as:

$$BW_{\text{prac}} = \frac{N_{\text{threads}} \times N_{\text{reps}} \times \text{Bytes}_{\text{window}} \times \text{traffic\_mult}}{\Delta t \times 10^9} \quad [\text{GB/s}]$$

where $\text{Bytes}_{\text{window}}$ is the 512-byte-aligned per-thread array size (`aligned_bytes` in `run_bw_probe`, or `window_bytes = 8 MB` in `run_shared_l3_bw_probe`).

---

### 3. Multi-Trial Statistical Aggregation (`compute_stats`)

After an untimed warmup pass, every probe runs $T$ independent barrier-synchronized trials (`--trials 7` by default). The trial measurements are sorted ascending ($x_{(0)} \le x_{(1)} \le \dots \le x_{(T-1)}$) to compute:

- **`median` (Reported Headline Ceiling)**:
  $$\text{median} = \begin{cases} x_{(\lfloor T/2 \rfloor)} & \text{if } T \text{ is odd} \\ \frac{1}{2}\left(x_{(T/2 - 1)} + x_{(T/2)}\right) & \text{if } T \text{ is even} \end{cases}$$
- **`mean`**: $\mu = \frac{1}{T}\sum_{i=0}^{T-1} x_i$
- **`min` / `max`**: $x_{(0)}$ and $x_{(T-1)}$
- **`stdev` (Sample Standard Deviation)**: $\sigma = \sqrt{\frac{1}{T - 1}\sum_{i=0}^{T-1} (x_i - \mu)^2}$
- **`cv_pct` (Coefficient of Variation %)**: $\text{CV}(\%) = 100 \times \frac{\sigma}{\mu}$

---

## Repeatability Design

Every probe in [cpu_ceilings.c](cpu_ceilings.c) enforces strict measurement repeatability:
1. **Physical Core Pinning (`sched_setaffinity`)**: Threads `0..N-1` are pinned 1-to-1 to physical cores `0..N-1` on Socket 0, preventing OS thread migration, SMT sibling contention, and cross-socket UPI traffic.
2. **Monotonic Hardware Timing (`CLOCK_MONOTONIC_RAW`)**: Immune to NTP slew adjustments.
3. **Barrier-Synchronized Windows (`#pragma omp barrier`)**: All threads synchronize immediately before and after the timed loop, eliminating OpenMP thread-pool fork/join overhead from timed windows.
4. **Explicit Warmup + Multi-Trial Statistics**: Runs an untimed warmup pass to transition cores to sustained AVX-512/AMX turbo frequencies and warm TLBs/caches, followed by `N` independent timed trials (`--trials 7` by default), emitting `median`, `mean`, `min`, `max`, `stdev`, and Coefficient of Variation (`cv_pct`).

## Usage

```bash
gcc -O3 -march=sapphirerapids -fopenmp kernel/probes/cpu_ceilings.c -o /tmp/cpu_ceilings -lm
/tmp/cpu_ceilings --threads 28 --trials 7 \
    --output kernel/reports/gemma4-31b-sapphire-rapids/ceilings_sapphire_rapids.json
```

## Reference Output

- [ceilings_sapphire_rapids.json](../reports/gemma4-31b-sapphire-rapids/ceilings_sapphire_rapids.json) — Full 7-trial empirical ceiling profile for Intel Sapphire Rapids.
- [kernel_report_gemma4_31b_sapphire_rapids.md](../reports/gemma4-31b-sapphire-rapids/kernel_report_gemma4_31b_sapphire_rapids.md) — General kernel report (hardware roofline comparison and numerical precision verification) for load-bearing XNNPACK and YNNPACK microkernels.
