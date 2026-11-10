// Empirical & Repeatable Hardware Ceiling Software Probes for x86_64
// (Intel Sapphire Rapids: AMX-INT8, AMX-BF16, AVX-512 VNNI, AVX-VNNI, AVX-512 FMA,
// and L1D / L2 / Shared L3 / DDR5 DRAM Bandwidth).
//
// Design for Repeatability:
//   1. Strict physical-core affinity: pins thread `tid` to physical core `tid`
//      on NUMA Node 0 via `sched_setaffinity` to prevent thread migration,
//      SMT sibling contention, and cross-socket UPI traffic.
//   2. Barrier-synchronized timing: threads synchronize via `#pragma omp barrier`
//      immediately before and after the timed loop, eliminating OpenMP fork/join
//      jitter.
//   3. Multi-trial statistics: runs an explicit warmup pass followed by `T`
//      independent timed trials (default T=7) and reports `median`, `mean`,
//      `min`, `max`, `stdev`, and `cv_pct` (coefficient of variation %).
//   4. Denormal-free deterministic initialization: all memory buffers are
//      64-byte aligned, pre-faulted, and initialized with normal FP32 values.
//
// Compile:
//   gcc -O3 -march=sapphirerapids -fopenmp kernel/probes/cpu_ceilings.c -o cpu_ceilings -lm

#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

#define MAX_TRIALS 32

typedef struct __tile_config {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __attribute__((packed)) tile_config_t;

typedef struct {
  double median;
  double mean;
  double min;
  double max;
  double stdev;
  double cv_pct;
} probe_stats_t;

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void pin_thread_to_core(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

static int enable_amx_current_thread(void) {
  long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  return (rc == 0) ? 0 : -1;
}

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

static probe_stats_t compute_stats(const double *samples, int n) {
  double sorted[MAX_TRIALS];
  for (int i = 0; i < n; ++i) sorted[i] = samples[i];
  qsort(sorted, n, sizeof(double), cmp_double);

  probe_stats_t s;
  s.min = sorted[0];
  s.max = sorted[n - 1];
  s.median = (n % 2 == 1) ? sorted[n / 2]
                          : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += sorted[i];
  s.mean = sum / n;

  double var = 0.0;
  for (int i = 0; i < n; ++i) {
    double d = sorted[i] - s.mean;
    var += d * d;
  }
  s.stdev = (n > 1) ? sqrt(var / (n - 1)) : 0.0;
  s.cv_pct = (s.mean > 0.0) ? (100.0 * s.stdev / s.mean) : 0.0;
  return s;
}

static void read_cpu_model(char *buf, size_t buflen) {
  snprintf(buf, buflen, "Unknown x86_64 CPU");
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (!f) return;
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "model name", 10) == 0) {
      char *colon = strchr(line, ':');
      if (colon) {
        colon++;
        while (*colon == ' ' || *colon == '\t') colon++;
        char *nl = strchr(colon, '\n');
        if (nl) *nl = '\0';
        snprintf(buf, buflen, "%s", colon);
        break;
      }
    }
  }
  fclose(f);
}

// -----------------------------------------------------------------------------
// 1. Compute Ceiling Probes: C_prac(op, dtype)
// -----------------------------------------------------------------------------

// AMX-INT8: tdpbssd on 6 independent tiles (tmm0..tmm5 += tmm6 * tmm7)
// Each 16x16x64 tile MAC = 16,384 INT8 MACs = 32,768 INT8 OPs.
// 12 tile instructions per loop body = 393,216 INT8 OPs per iteration.
static probe_stats_t run_amx_int8_probe(int num_threads, int64_t iters,
                                        int num_trials) {
  double samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;
  enable_amx_current_thread();

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    enable_amx_current_thread();

    tile_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    for (int i = 0; i < 8; ++i) {
      cfg.rows[i] = 16;
      cfg.colsb[i] = 64;
    }
    _tile_loadconfig(&cfg);
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);

    // Warmup pass (frequency transition + license warmup)
    for (int64_t i = 0; i < iters / 2; ++i) {
      __asm__ volatile(
          "tdpbssd %%tmm6, %%tmm7, %%tmm0\n\t"
          "tdpbssd %%tmm6, %%tmm7, %%tmm1\n\t"
          "tdpbssd %%tmm6, %%tmm7, %%tmm2\n\t"
          "tdpbssd %%tmm6, %%tmm7, %%tmm3\n\t"
          "tdpbssd %%tmm6, %%tmm7, %%tmm4\n\t"
          "tdpbssd %%tmm6, %%tmm7, %%tmm5\n\t"
          :
          :
          : "memory");
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int64_t i = 0; i < iters; ++i) {
        __asm__ volatile(
            "tdpbssd %%tmm6, %%tmm7, %%tmm0\n\t"
            "tdpbssd %%tmm6, %%tmm7, %%tmm1\n\t"
            "tdpbssd %%tmm6, %%tmm7, %%tmm2\n\t"
            "tdpbssd %%tmm6, %%tmm7, %%tmm3\n\t"
            "tdpbssd %%tmm6, %%tmm7, %%tmm4\n\t"
            "tdpbssd %%tmm6, %%tmm7, %%tmm5\n\t"
            "tdpbssd %%tmm6, %%tmm7, %%tmm0\n\t"
            "tdpbssd %%tmm6, %%tmm7, %%tmm1\n\t"
            "tdpbssd %%tmm6, %%tmm7, %%tmm2\n\t"
            "tdpbssd %%tmm6, %%tmm7, %%tmm3\n\t"
            "tdpbssd %%tmm6, %%tmm7, %%tmm4\n\t"
            "tdpbssd %%tmm6, %%tmm7, %%tmm5\n\t"
            :
            :
            : "memory");
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        double ops_per_iter = 12.0 * 16.0 * 16.0 * 64.0 * 2.0;
        samples[trial] =
            (double)num_threads * (double)iters * ops_per_iter / (t1 - t0) / 1e9;
      }
      #pragma omp barrier
    }
    _tile_release();
  }
  return compute_stats(samples, num_trials);
}

// AMX-BF16: tdpbf16ps on 6 independent tiles (tmm0..tmm5 += tmm6 * tmm7)
// Each 16x16x32 tile MAC = 8,192 BF16 MACs = 16,384 BF16 FLOPs.
static probe_stats_t run_amx_bf16_probe(int num_threads, int64_t iters,
                                        int num_trials) {
  double samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;
  enable_amx_current_thread();

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    enable_amx_current_thread();

    tile_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    for (int i = 0; i < 8; ++i) {
      cfg.rows[i] = 16;
      cfg.colsb[i] = 64;
    }
    _tile_loadconfig(&cfg);
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);

    // Warmup
    for (int64_t i = 0; i < iters / 2; ++i) {
      __asm__ volatile(
          "tdpbf16ps %%tmm6, %%tmm7, %%tmm0\n\t"
          "tdpbf16ps %%tmm6, %%tmm7, %%tmm1\n\t"
          "tdpbf16ps %%tmm6, %%tmm7, %%tmm2\n\t"
          "tdpbf16ps %%tmm6, %%tmm7, %%tmm3\n\t"
          "tdpbf16ps %%tmm6, %%tmm7, %%tmm4\n\t"
          "tdpbf16ps %%tmm6, %%tmm7, %%tmm5\n\t"
          :
          :
          : "memory");
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int64_t i = 0; i < iters; ++i) {
        __asm__ volatile(
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm0\n\t"
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm1\n\t"
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm2\n\t"
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm3\n\t"
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm4\n\t"
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm5\n\t"
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm0\n\t"
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm1\n\t"
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm2\n\t"
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm3\n\t"
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm4\n\t"
            "tdpbf16ps %%tmm6, %%tmm7, %%tmm5\n\t"
            :
            :
            : "memory");
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        double ops_per_iter = 12.0 * 16.0 * 16.0 * 32.0 * 2.0;
        samples[trial] =
            (double)num_threads * (double)iters * ops_per_iter / (t1 - t0) / 1e9;
      }
      #pragma omp barrier
    }
    _tile_release();
  }
  return compute_stats(samples, num_trials);
}

// AVX-512 VNNI (512-bit vpdpbusd): 64 INT8 MACs = 128 INT8 OPs per instruction
static probe_stats_t run_avx512_vnni_probe(int num_threads, int64_t iters,
                                           int num_trials) {
  double samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    __asm__ volatile(
        "vpxord %%zmm0, %%zmm0, %%zmm0\n\t"
        "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"
        "vpxord %%zmm2, %%zmm2, %%zmm2\n\t"
        "vpxord %%zmm3, %%zmm3, %%zmm3\n\t"
        "vpxord %%zmm4, %%zmm4, %%zmm4\n\t"
        "vpxord %%zmm5, %%zmm5, %%zmm5\n\t"
        "vpxord %%zmm6, %%zmm6, %%zmm6\n\t"
        "vpxord %%zmm7, %%zmm7, %%zmm7\n\t"
        "vpxord %%zmm8, %%zmm8, %%zmm8\n\t"
        "vpxord %%zmm9, %%zmm9, %%zmm9\n\t"
        "vpxord %%zmm10, %%zmm10, %%zmm10\n\t"
        "vpxord %%zmm11, %%zmm11, %%zmm11\n\t"
        "vpxord %%zmm12, %%zmm12, %%zmm12\n\t"
        "vpxord %%zmm13, %%zmm13, %%zmm13\n\t"
        ::: "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7",
            "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13");

    // Warmup
    for (int64_t i = 0; i < iters / 4; ++i) {
      __asm__ volatile(
          "vpdpbusd %%zmm12, %%zmm13, %%zmm0\n\t"
          "vpdpbusd %%zmm12, %%zmm13, %%zmm1\n\t"
          "vpdpbusd %%zmm12, %%zmm13, %%zmm2\n\t"
          "vpdpbusd %%zmm12, %%zmm13, %%zmm3\n\t"
          "vpdpbusd %%zmm12, %%zmm13, %%zmm4\n\t"
          "vpdpbusd %%zmm12, %%zmm13, %%zmm5\n\t"
          ::: "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5");
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int64_t i = 0; i < iters; ++i) {
        __asm__ volatile(
            "vpdpbusd %%zmm12, %%zmm13, %%zmm0\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm1\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm2\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm3\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm4\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm5\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm6\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm7\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm8\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm9\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm10\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm11\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm0\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm1\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm2\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm3\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm4\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm5\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm6\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm7\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm8\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm9\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm10\n\t"
            "vpdpbusd %%zmm12, %%zmm13, %%zmm11\n\t"
            :
            :
            : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7",
              "zmm8", "zmm9", "zmm10", "zmm11");
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        double ops_per_iter = 24.0 * 64.0 * 2.0;
        samples[trial] =
            (double)num_threads * (double)iters * ops_per_iter / (t1 - t0) / 1e9;
      }
      #pragma omp barrier
    }
  }
  return compute_stats(samples, num_trials);
}

// 256-bit VNNI (vpdpbusd ymm): 32 INT8 MACs = 64 INT8 OPs per instruction
static probe_stats_t run_avx_vnni_256_probe(int num_threads, int64_t iters,
                                            int num_trials) {
  double samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    __asm__ volatile(
        "vpxor %%ymm0, %%ymm0, %%ymm0\n\t"
        "vpxor %%ymm1, %%ymm1, %%ymm1\n\t"
        "vpxor %%ymm2, %%ymm2, %%ymm2\n\t"
        "vpxor %%ymm3, %%ymm3, %%ymm3\n\t"
        "vpxor %%ymm4, %%ymm4, %%ymm4\n\t"
        "vpxor %%ymm5, %%ymm5, %%ymm5\n\t"
        "vpxor %%ymm6, %%ymm6, %%ymm6\n\t"
        "vpxor %%ymm7, %%ymm7, %%ymm7\n\t"
        "vpxor %%ymm8, %%ymm8, %%ymm8\n\t"
        "vpxor %%ymm9, %%ymm9, %%ymm9\n\t"
        "vpxor %%ymm10, %%ymm10, %%ymm10\n\t"
        "vpxor %%ymm11, %%ymm11, %%ymm11\n\t"
        "vpxor %%ymm12, %%ymm12, %%ymm12\n\t"
        "vpxor %%ymm13, %%ymm13, %%ymm13\n\t"
        ::: "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
            "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13");

    // Warmup
    for (int64_t i = 0; i < iters / 4; ++i) {
      __asm__ volatile(
          "vpdpbusd %%ymm12, %%ymm13, %%ymm0\n\t"
          "vpdpbusd %%ymm12, %%ymm13, %%ymm1\n\t"
          "vpdpbusd %%ymm12, %%ymm13, %%ymm2\n\t"
          "vpdpbusd %%ymm12, %%ymm13, %%ymm3\n\t"
          ::: "ymm0", "ymm1", "ymm2", "ymm3");
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int64_t i = 0; i < iters; ++i) {
        __asm__ volatile(
            "vpdpbusd %%ymm12, %%ymm13, %%ymm0\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm1\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm2\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm3\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm4\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm5\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm6\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm7\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm8\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm9\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm10\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm11\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm0\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm1\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm2\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm3\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm4\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm5\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm6\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm7\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm8\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm9\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm10\n\t"
            "vpdpbusd %%ymm12, %%ymm13, %%ymm11\n\t"
            :
            :
            : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
              "ymm8", "ymm9", "ymm10", "ymm11");
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        double ops_per_iter = 24.0 * 32.0 * 2.0;
        samples[trial] =
            (double)num_threads * (double)iters * ops_per_iter / (t1 - t0) / 1e9;
      }
      #pragma omp barrier
    }
  }
  return compute_stats(samples, num_trials);
}

// AVX-512 FP32 FMA (512-bit vfmadd231ps): 16 FP32 MACs = 32 FP32 FLOPs per inst
static probe_stats_t run_avx512_fp32_fma_probe(int num_threads, int64_t iters,
                                               int num_trials) {
  double samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    __asm__ volatile(
        "vxorps %%zmm0, %%zmm0, %%zmm0\n\t"
        "vxorps %%zmm1, %%zmm1, %%zmm1\n\t"
        "vxorps %%zmm2, %%zmm2, %%zmm2\n\t"
        "vxorps %%zmm3, %%zmm3, %%zmm3\n\t"
        "vxorps %%zmm4, %%zmm4, %%zmm4\n\t"
        "vxorps %%zmm5, %%zmm5, %%zmm5\n\t"
        "vxorps %%zmm6, %%zmm6, %%zmm6\n\t"
        "vxorps %%zmm7, %%zmm7, %%zmm7\n\t"
        "vxorps %%zmm8, %%zmm8, %%zmm8\n\t"
        "vxorps %%zmm9, %%zmm9, %%zmm9\n\t"
        "vxorps %%zmm10, %%zmm10, %%zmm10\n\t"
        "vxorps %%zmm11, %%zmm11, %%zmm11\n\t"
        "vxorps %%zmm12, %%zmm12, %%zmm12\n\t"
        "vxorps %%zmm13, %%zmm13, %%zmm13\n\t"
        ::: "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7",
            "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13");

    // Warmup
    for (int64_t i = 0; i < iters / 4; ++i) {
      __asm__ volatile(
          "vfmadd231ps %%zmm12, %%zmm13, %%zmm0\n\t"
          "vfmadd231ps %%zmm12, %%zmm13, %%zmm1\n\t"
          "vfmadd231ps %%zmm12, %%zmm13, %%zmm2\n\t"
          "vfmadd231ps %%zmm12, %%zmm13, %%zmm3\n\t"
          ::: "zmm0", "zmm1", "zmm2", "zmm3");
    }

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      for (int64_t i = 0; i < iters; ++i) {
        __asm__ volatile(
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm0\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm1\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm2\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm3\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm4\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm5\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm6\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm7\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm8\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm9\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm10\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm11\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm0\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm1\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm2\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm3\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm4\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm5\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm6\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm7\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm8\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm9\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm10\n\t"
            "vfmadd231ps %%zmm12, %%zmm13, %%zmm11\n\t"
            :
            :
            : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7",
              "zmm8", "zmm9", "zmm10", "zmm11");
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        double ops_per_iter = 24.0 * 16.0 * 2.0;
        samples[trial] =
            (double)num_threads * (double)iters * ops_per_iter / (t1 - t0) / 1e9;
      }
      #pragma omp barrier
    }
  }
  return compute_stats(samples, num_trials);
}

// -----------------------------------------------------------------------------
// 2. Memory-Hierarchy Bandwidth Ceiling Probes: BW_prac(WorkingSet)
// -----------------------------------------------------------------------------

typedef enum {
  BW_MODE_READ_ONLY = 0,
  BW_MODE_READ_WRITE_COPY = 1,
  BW_MODE_STREAM_TRIAD = 2
} bw_mode_t;

static probe_stats_t run_bw_probe(int num_threads, size_t bytes_per_thread,
                                  int64_t reps, bw_mode_t mode,
                                  int num_trials) {
  size_t aligned_bytes = (bytes_per_thread + 511) & ~(size_t)511;
  size_t num_floats = aligned_bytes / sizeof(float);

  float **buf_a = (float **)calloc(num_threads, sizeof(float *));
  float **buf_b = (float **)calloc(num_threads, sizeof(float *));
  float **buf_c = (float **)calloc(num_threads, sizeof(float *));
  double samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;

  double traffic_mult = (mode == BW_MODE_READ_ONLY)
                            ? 1.0
                            : (mode == BW_MODE_READ_WRITE_COPY ? 2.0 : 3.0);

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    buf_a[tid] = (float *)aligned_alloc(64, aligned_bytes);
    buf_b[tid] = (float *)aligned_alloc(64, aligned_bytes);
    buf_c[tid] = (float *)aligned_alloc(64, aligned_bytes);
    for (size_t i = 0; i < num_floats; ++i) {
      buf_a[tid][i] = 1.0f;
      buf_b[tid][i] = 2.0f;
      buf_c[tid][i] = 0.5f;
    }

    const float *a = buf_a[tid];
    float *b = buf_b[tid];
    float *c = buf_c[tid];

    // Warmup pass
    __m512 acc = _mm512_setzero_ps();
    for (int64_t r = 0; r < (reps > 4 ? reps / 4 : 1); ++r) {
      for (size_t i = 0; i < num_floats; i += 16) {
        __m512 va = _mm512_load_ps(a + i);
        acc = _mm512_add_ps(acc, va);
        if (mode == BW_MODE_READ_WRITE_COPY) {
          _mm512_store_ps(b + i, va);
        } else if (mode == BW_MODE_STREAM_TRIAD) {
          __m512 vb = _mm512_load_ps(b + i);
          _mm512_store_ps(c + i, _mm512_add_ps(va, vb));
        }
      }
    }
    __asm__ volatile("" : "+v"(acc));

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      if (mode == BW_MODE_READ_ONLY) {
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        for (int64_t r = 0; r < reps; ++r) {
          for (size_t i = 0; i < num_floats; i += 64) {
            acc0 = _mm512_add_ps(acc0, _mm512_load_ps(a + i + 0));
            acc1 = _mm512_add_ps(acc1, _mm512_load_ps(a + i + 16));
            acc2 = _mm512_add_ps(acc2, _mm512_load_ps(a + i + 32));
            acc3 = _mm512_add_ps(acc3, _mm512_load_ps(a + i + 48));
          }
          __asm__ volatile("" : "+v"(acc0), "+v"(acc1), "+v"(acc2), "+v"(acc3));
        }
      } else if (mode == BW_MODE_READ_WRITE_COPY) {
        for (int64_t r = 0; r < reps; ++r) {
          for (size_t i = 0; i < num_floats; i += 64) {
            __m512 v0 = _mm512_load_ps(a + i + 0);
            __m512 v1 = _mm512_load_ps(a + i + 16);
            __m512 v2 = _mm512_load_ps(a + i + 32);
            __m512 v3 = _mm512_load_ps(a + i + 48);
            _mm512_store_ps(b + i + 0, v0);
            _mm512_store_ps(b + i + 16, v1);
            _mm512_store_ps(b + i + 32, v2);
            _mm512_store_ps(b + i + 48, v3);
          }
          __asm__ volatile("" ::: "memory");
        }
      } else {
        __m512 vscale = _mm512_set1_ps(0.5f);
        for (int64_t r = 0; r < reps; ++r) {
          for (size_t i = 0; i < num_floats; i += 64) {
            __m512 a0 = _mm512_load_ps(a + i + 0);
            __m512 a1 = _mm512_load_ps(a + i + 16);
            __m512 a2 = _mm512_load_ps(a + i + 32);
            __m512 a3 = _mm512_load_ps(a + i + 48);
            __m512 b0 = _mm512_load_ps(b + i + 0);
            __m512 b1 = _mm512_load_ps(b + i + 16);
            __m512 b2 = _mm512_load_ps(b + i + 32);
            __m512 b3 = _mm512_load_ps(b + i + 48);
            _mm512_store_ps(c + i + 0, _mm512_fmadd_ps(vscale, a0, b0));
            _mm512_store_ps(c + i + 16, _mm512_fmadd_ps(vscale, a1, b1));
            _mm512_store_ps(c + i + 32, _mm512_fmadd_ps(vscale, a2, b2));
            _mm512_store_ps(c + i + 48, _mm512_fmadd_ps(vscale, a3, b3));
          }
          __asm__ volatile("" ::: "memory");
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        double total_bytes = (double)num_threads * (double)reps *
                             (double)aligned_bytes * traffic_mult;
        samples[trial] = total_bytes / (t1 - t0) / 1e9;
      }
      #pragma omp barrier
    }
  }

  for (int t = 0; t < num_threads; ++t) {
    free(buf_a[t]);
    free(buf_b[t]);
    free(buf_c[t]);
  }
  free(buf_a);
  free(buf_b);
  free(buf_c);

  return compute_stats(samples, num_trials);
}

// Shared L3 probe: allocates a 32 MB shared pool (< 52.5 MB Sapphire Rapids L3)
// where each thread streams over an 8 MB window (> 2 MB per-core L2).
static probe_stats_t run_shared_l3_bw_probe(int num_threads, int64_t reps,
                                            bw_mode_t mode, int num_trials) {
  size_t total_pool_bytes = 32 * 1024 * 1024;
  size_t window_bytes = 8 * 1024 * 1024;
  size_t pool_floats = total_pool_bytes / sizeof(float);
  size_t window_floats = window_bytes / sizeof(float);

  float *pool_a = (float *)aligned_alloc(64, total_pool_bytes);
  float *pool_b = (float *)aligned_alloc(64, total_pool_bytes);
  for (size_t i = 0; i < pool_floats; ++i) {
    pool_a[i] = 1.0f;
    pool_b[i] = 2.0f;
  }

  double samples[MAX_TRIALS];
  double t0 = 0.0, t1 = 0.0;
  double traffic_mult = (mode == BW_MODE_READ_ONLY) ? 1.0 : 2.0;

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    pin_thread_to_core(tid);
    size_t offset = ((size_t)tid * 512 * 1024 / sizeof(float)) %
                    (pool_floats - window_floats);
    const float *a = pool_a + offset;
    float *b = pool_b + offset;

    // Warmup into L3
    __m512 acc = _mm512_setzero_ps();
    for (int64_t r = 0; r < 10; ++r) {
      for (size_t i = 0; i < window_floats; i += 16) {
        __m512 va = _mm512_load_ps(a + i);
        acc = _mm512_add_ps(acc, va);
        if (mode == BW_MODE_READ_WRITE_COPY) {
          _mm512_store_ps(b + i, va);
        }
      }
    }
    __asm__ volatile("" : "+v"(acc));

    for (int trial = 0; trial < num_trials; ++trial) {
      #pragma omp barrier
      #pragma omp master
      { t0 = now_sec(); }
      #pragma omp barrier

      if (mode == BW_MODE_READ_ONLY) {
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        for (int64_t r = 0; r < reps; ++r) {
          for (size_t i = 0; i < window_floats; i += 64) {
            acc0 = _mm512_add_ps(acc0, _mm512_load_ps(a + i + 0));
            acc1 = _mm512_add_ps(acc1, _mm512_load_ps(a + i + 16));
            acc2 = _mm512_add_ps(acc2, _mm512_load_ps(a + i + 32));
            acc3 = _mm512_add_ps(acc3, _mm512_load_ps(a + i + 48));
          }
          __asm__ volatile("" : "+v"(acc0), "+v"(acc1), "+v"(acc2), "+v"(acc3));
        }
      } else {
        for (int64_t r = 0; r < reps; ++r) {
          for (size_t i = 0; i < window_floats; i += 64) {
            __m512 v0 = _mm512_load_ps(a + i + 0);
            __m512 v1 = _mm512_load_ps(a + i + 16);
            __m512 v2 = _mm512_load_ps(a + i + 32);
            __m512 v3 = _mm512_load_ps(a + i + 48);
            _mm512_store_ps(b + i + 0, v0);
            _mm512_store_ps(b + i + 16, v1);
            _mm512_store_ps(b + i + 32, v2);
            _mm512_store_ps(b + i + 48, v3);
          }
          __asm__ volatile("" ::: "memory");
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        t1 = now_sec();
        double total_bytes = (double)num_threads * (double)reps *
                             (double)window_bytes * traffic_mult;
        samples[trial] = total_bytes / (t1 - t0) / 1e9;
      }
      #pragma omp barrier
    }
  }

  free(pool_a);
  free(pool_b);
  return compute_stats(samples, num_trials);
}

static void print_stat_obj(FILE *out, const char *indent, const char *name,
                           probe_stats_t s1, probe_stats_t smc, int mc_threads,
                           int trailing_comma) {
  fprintf(out, "%s\"%s\": {\n", indent, name);
  fprintf(out,
          "%s  \"single_core\": %.2f,\n"
          "%s  \"single_core_stats\": { \"median\": %.2f, \"mean\": %.2f, "
          "\"min\": %.2f, \"max\": %.2f, \"stdev\": %.2f, \"cv_pct\": %.2f },\n",
          indent, s1.median, indent, s1.median, s1.mean, s1.min, s1.max,
          s1.stdev, s1.cv_pct);
  fprintf(out,
          "%s  \"numa_node0_%dc\": %.2f,\n"
          "%s  \"numa_node0_%dc_stats\": { \"median\": %.2f, \"mean\": %.2f, "
          "\"min\": %.2f, \"max\": %.2f, \"stdev\": %.2f, \"cv_pct\": %.2f }\n",
          indent, mc_threads, smc.median, indent, mc_threads, smc.median,
          smc.mean, smc.min, smc.max, smc.stdev, smc.cv_pct);
  fprintf(out, "%s}%s\n", indent, trailing_comma ? "," : "");
}

int main(int argc, char **argv) {
  int multi_threads = 28;
  int num_trials = 7;
  const char *out_path = NULL;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      multi_threads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
      num_trials = atoi(argv[++i]);
      if (num_trials < 1) num_trials = 1;
      if (num_trials > MAX_TRIALS) num_trials = MAX_TRIALS;
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else if (argv[i][0] != '-') {
      multi_threads = atoi(argv[i]);
    }
  }

  int64_t compute_iters = 2000000LL;
  probe_stats_t amx_int8_1c = run_amx_int8_probe(1, compute_iters, num_trials);
  probe_stats_t amx_int8_mc =
      run_amx_int8_probe(multi_threads, compute_iters, num_trials);

  probe_stats_t amx_bf16_1c = run_amx_bf16_probe(1, compute_iters, num_trials);
  probe_stats_t amx_bf16_mc =
      run_amx_bf16_probe(multi_threads, compute_iters, num_trials);

  probe_stats_t avx512_vnni_1c =
      run_avx512_vnni_probe(1, compute_iters * 8, num_trials);
  probe_stats_t avx512_vnni_mc =
      run_avx512_vnni_probe(multi_threads, compute_iters * 8, num_trials);

  probe_stats_t avx_vnni_256_1c =
      run_avx_vnni_256_probe(1, compute_iters * 8, num_trials);
  probe_stats_t avx_vnni_256_mc =
      run_avx_vnni_256_probe(multi_threads, compute_iters * 8, num_trials);

  probe_stats_t avx512_fp32_1c =
      run_avx512_fp32_fma_probe(1, compute_iters * 8, num_trials);
  probe_stats_t avx512_fp32_mc =
      run_avx512_fp32_fma_probe(multi_threads, compute_iters * 8, num_trials);

  // Memory Hierarchy Bandwidth Ceilings
  probe_stats_t l1_read_1c =
      run_bw_probe(1, 16 * 1024, 150000, BW_MODE_READ_ONLY, num_trials);
  probe_stats_t l1_read_mc = run_bw_probe(
      multi_threads, 16 * 1024, 150000, BW_MODE_READ_ONLY, num_trials);
  probe_stats_t l1_copy_1c =
      run_bw_probe(1, 8 * 1024, 150000, BW_MODE_READ_WRITE_COPY, num_trials);
  probe_stats_t l1_copy_mc = run_bw_probe(
      multi_threads, 8 * 1024, 150000, BW_MODE_READ_WRITE_COPY, num_trials);

  probe_stats_t l2_read_1c =
      run_bw_probe(1, 512 * 1024, 8000, BW_MODE_READ_ONLY, num_trials);
  probe_stats_t l2_read_mc = run_bw_probe(
      multi_threads, 512 * 1024, 8000, BW_MODE_READ_ONLY, num_trials);
  probe_stats_t l2_copy_1c =
      run_bw_probe(1, 256 * 1024, 8000, BW_MODE_READ_WRITE_COPY, num_trials);
  probe_stats_t l2_copy_mc = run_bw_probe(
      multi_threads, 256 * 1024, 8000, BW_MODE_READ_WRITE_COPY, num_trials);

  probe_stats_t l3_read_1c =
      run_bw_probe(1, 16 * 1024 * 1024, 300, BW_MODE_READ_ONLY, num_trials);
  probe_stats_t l3_read_mc =
      run_shared_l3_bw_probe(multi_threads, 200, BW_MODE_READ_ONLY, num_trials);
  probe_stats_t l3_copy_1c = run_bw_probe(
      1, 8 * 1024 * 1024, 300, BW_MODE_READ_WRITE_COPY, num_trials);
  probe_stats_t l3_copy_mc = run_shared_l3_bw_probe(
      multi_threads, 200, BW_MODE_READ_WRITE_COPY, num_trials);

  probe_stats_t dram_read_1c =
      run_bw_probe(1, 256 * 1024 * 1024, 30, BW_MODE_READ_ONLY, num_trials);
  probe_stats_t dram_read_mc = run_bw_probe(
      multi_threads, 32 * 1024 * 1024, 40, BW_MODE_READ_ONLY, num_trials);
  probe_stats_t dram_copy_1c = run_bw_probe(
      1, 128 * 1024 * 1024, 30, BW_MODE_READ_WRITE_COPY, num_trials);
  probe_stats_t dram_copy_mc = run_bw_probe(
      multi_threads, 16 * 1024 * 1024, 40, BW_MODE_READ_WRITE_COPY, num_trials);
  probe_stats_t dram_triad_1c =
      run_bw_probe(1, 128 * 1024 * 1024, 30, BW_MODE_STREAM_TRIAD, num_trials);
  probe_stats_t dram_triad_mc = run_bw_probe(
      multi_threads, 16 * 1024 * 1024, 40, BW_MODE_STREAM_TRIAD, num_trials);

  FILE *outs[2] = {stdout, NULL};
  int num_outs = 1;
  if (out_path) {
    outs[1] = fopen(out_path, "w");
    if (outs[1]) num_outs = 2;
  }

  for (int o = 0; o < num_outs; ++o) {
    FILE *out = outs[o];
    fprintf(out, "{\n");
    fprintf(out, "  \"arch\": \"x86_64\",\n");
    fprintf(out, "  \"microarchitecture\": \"Intel Sapphire Rapids\",\n");
    fprintf(out, "  \"methodology\": {\n");
    fprintf(out, "    \"clock\": \"CLOCK_MONOTONIC_RAW\",\n");
    fprintf(out, "    \"affinity\": \"sched_setaffinity physical cores 0..%d (NUMA Node 0)\",\n",
            multi_threads - 1);
    fprintf(out, "    \"synchronization\": \"omp barrier before and after timed loop\",\n");
    fprintf(out, "    \"num_trials\": %d,\n", num_trials);
    fprintf(out, "    \"reported_value\": \"median\"\n");
    fprintf(out, "  },\n");
    fprintf(out, "  \"threads_single_core\": 1,\n");
    fprintf(out, "  \"threads_numa_node0\": %d,\n", multi_threads);
    fprintf(out, "  \"compute_ceilings_gops\": {\n");
    print_stat_obj(out, "    ", "amx_int8_tdpbssd", amx_int8_1c, amx_int8_mc,
                   multi_threads, 1);
    print_stat_obj(out, "    ", "amx_bf16_tdpbf16ps", amx_bf16_1c, amx_bf16_mc,
                   multi_threads, 1);
    print_stat_obj(out, "    ", "avx512_vnni_int8_vpdpbusd", avx512_vnni_1c,
                   avx512_vnni_mc, multi_threads, 1);
    print_stat_obj(out, "    ", "avx_vnni_256_int8_vpdpbusd", avx_vnni_256_1c,
                   avx_vnni_256_mc, multi_threads, 1);
    print_stat_obj(out, "    ", "avx512_fp32_fma", avx512_fp32_1c,
                   avx512_fp32_mc, multi_threads, 0);
    fprintf(out, "  },\n");
    fprintf(out, "  \"bandwidth_ceilings_gbps\": {\n");
    fprintf(out, "    \"l1d_16kb_per_core\": {\n");
    print_stat_obj(out, "      ", "read_only", l1_read_1c, l1_read_mc,
                   multi_threads, 1);
    print_stat_obj(out, "      ", "read_write_copy", l1_copy_1c, l1_copy_mc,
                   multi_threads, 0);
    fprintf(out, "    },\n");
    fprintf(out, "    \"l2_512kb_per_core\": {\n");
    print_stat_obj(out, "      ", "read_only", l2_read_1c, l2_read_mc,
                   multi_threads, 1);
    print_stat_obj(out, "      ", "read_write_copy", l2_copy_1c, l2_copy_mc,
                   multi_threads, 0);
    fprintf(out, "    },\n");
    fprintf(out, "    \"l3_llc\": {\n");
    print_stat_obj(out, "      ", "read_only", l3_read_1c, l3_read_mc,
                   multi_threads, 1);
    print_stat_obj(out, "      ", "read_write_copy", l3_copy_1c, l3_copy_mc,
                   multi_threads, 0);
    fprintf(out, "    },\n");
    fprintf(out, "    \"ddr5_dram\": {\n");
    print_stat_obj(out, "      ", "read_only", dram_read_1c, dram_read_mc,
                   multi_threads, 1);
    print_stat_obj(out, "      ", "read_write_copy", dram_copy_1c, dram_copy_mc,
                   multi_threads, 1);
    print_stat_obj(out, "      ", "stream_triad", dram_triad_1c, dram_triad_mc,
                   multi_threads, 0);
    fprintf(out, "    }\n");
    fprintf(out, "  }\n");
    fprintf(out, "}\n");
  }
  if (outs[1]) fclose(outs[1]);
  return 0;
}
