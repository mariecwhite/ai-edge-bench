// Empirical & Repeatable Hardware Ceiling Software Probes for ARM64 / Apple Silicon
// (Apple M5 Max: SME2 SMOPA INT8, SME2 BFMOPA BF16, SME2 FMOPA FP16/FP32,
// NEON I8MM SMMLA, NEON DotProd SDOT, NEON FP32 FMLA,
// and L1D / Cluster L2 / System Level Cache (SLC) / Unified Memory Bandwidth).
//
// Design for Repeatability on Heterogeneous Apple Silicon (Super + Performance cores):
//   1. Zero external dependencies: uses POSIX threads (`pthreads`) with an atomic
//      spin-wait barrier (`yield`) and macOS `QOS_CLASS_USER_INTERACTIVE` so threads
//      target Super cores first (1T, 6T) and scale cleanly across all Performance
//      clusters (18T).
//   2. Time-windowed heterogeneous measurement: because Super and Performance clusters
//      run at different frequencies and SME2 rates, a fixed-iterations-per-thread
//      barrier stalls fast Super cores behind slower Performance cores. Every probe
//      instead executes synchronized time-bounded windows where each thread counts
//      its own completed work between start and stop signals, yielding the true
//      aggregate sustained hardware ceiling across heterogeneous clusters.
//   3. Multi-trial statistics: runs an explicit warmup window followed by `T`
//      independent timed trials (default T=7) using `CLOCK_MONOTONIC_RAW`, emitting
//      `median`, `mean`, `min`, `max`, `stdev`, and `cv_pct`.
//   4. 128-byte Apple Silicon cache-line alignment and denormal-free FP32 buffers.
//
// Compile (macOS Apple Clang 21+, ARMv8.2 + SME2 without non-streaming SVE):
//   clang -O3 -march=armv8.2-a+sme+sme2+fp16+bf16+i8mm+dotprod \
//       kernel/probes/cpu_ceilings_arm64.c -o /tmp/cpu_ceilings_arm64 -lpthread -lm

#include <arm_neon.h>
#include <arm_sme.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#define MAX_TRIALS 32
#define MAX_THREAD_CFGS 8
#define MAX_THREADS 64

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

typedef struct {
  atomic_int arrived;
  atomic_int phase;
  int total;
} spin_barrier_t;

static void sb_init(spin_barrier_t *b, int n) {
  atomic_init(&b->arrived, 0);
  atomic_init(&b->phase, 0);
  b->total = n;
}

static void sb_wait(spin_barrier_t *b) {
  int p = atomic_load_explicit(&b->phase, memory_order_relaxed);
  if (atomic_fetch_add_explicit(&b->arrived, 1, memory_order_acq_rel) ==
      b->total - 1) {
    atomic_store_explicit(&b->arrived, 0, memory_order_relaxed);
    atomic_store_explicit(&b->phase, p + 1, memory_order_release);
  } else {
    while (atomic_load_explicit(&b->phase, memory_order_acquire) == p) {
      __asm__ volatile("yield");
    }
  }
}

static void set_high_qos(void) {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
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

static void read_cpu_brand(char *buf, size_t buflen) {
  snprintf(buf, buflen, "Apple Silicon ARM64");
#if defined(__APPLE__)
  size_t len = buflen;
  if (sysctlbyname("machdep.cpu.brand_string", buf, &len, NULL, 0) == 0 &&
      len > 0) {
    buf[buflen - 1] = '\0';
  }
#endif
}

__arm_locally_streaming static uint64_t get_sme_svl_bytes(void) {
  return svcntb();
}

// -----------------------------------------------------------------------------
// 1. Compute Ceiling Loops: C_prac(op, dtype)
// -----------------------------------------------------------------------------

typedef enum {
  COMP_SME2_INT8_SMOPA = 0,
  COMP_SME2_BF16_BFMOPA = 1,
  COMP_SME2_FP16_FMOPA = 2,
  COMP_SME2_FP32_FMOPA = 3,
  COMP_NEON_I8MM_SMMLA = 4,
  COMP_NEON_DOTPROD_SDOT = 5,
  COMP_NEON_FP32_FMLA = 6,
} compute_kind_t;

__arm_new("za") __arm_locally_streaming static int64_t
    sme2_smopa_int8_loop(const atomic_int *stop_flag) {
  svbool_t pg = svptrue_b8();
  svint8_t a = svdup_s8(1);
  svint8_t b = svdup_s8(1);
  svzero_za();
  int64_t chunks = 0;
  while (!atomic_load_explicit(stop_flag, memory_order_relaxed)) {
    for (int i = 0; i < 4096; ++i) {
      __asm__ volatile(
          "smopa za0.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za1.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za2.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za3.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za0.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za1.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za2.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za3.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za0.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za1.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za2.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za3.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za0.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za1.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za2.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          "smopa za3.s, %[p]/m, %[p]/m, %[a].b, %[b].b\n\t"
          :
          : [p] "Upa"(pg), [a] "w"(a), [b] "w"(b)
          : "memory");
    }
    chunks++;
  }
  return chunks * 4096LL * 16LL;
}

__arm_new("za") __arm_locally_streaming static int64_t
    sme2_bfmopa_bf16_loop(const atomic_int *stop_flag) {
  svbool_t pg = svptrue_b16();
  svuint16_t a = svdup_u16(0x3F80);  // BF16 1.0
  svuint16_t b = svdup_u16(0x3F80);
  svzero_za();
  int64_t chunks = 0;
  while (!atomic_load_explicit(stop_flag, memory_order_relaxed)) {
    for (int i = 0; i < 4096; ++i) {
      __asm__ volatile(
          "bfmopa za0.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za1.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za2.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za3.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za0.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za1.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za2.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za3.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za0.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za1.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za2.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za3.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za0.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za1.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za2.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "bfmopa za3.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          :
          : [p] "Upa"(pg), [a] "w"(a), [b] "w"(b)
          : "memory");
    }
    chunks++;
  }
  return chunks * 4096LL * 16LL;
}

__arm_new("za") __arm_locally_streaming static int64_t
    sme2_fmopa_fp16_loop(const atomic_int *stop_flag) {
  svbool_t pg = svptrue_b16();
  svfloat16_t a = svdup_f16((__fp16)1.0f);
  svfloat16_t b = svdup_f16((__fp16)1.0f);
  svzero_za();
  int64_t chunks = 0;
  while (!atomic_load_explicit(stop_flag, memory_order_relaxed)) {
    for (int i = 0; i < 4096; ++i) {
      __asm__ volatile(
          "fmopa za0.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za1.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za2.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za3.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za0.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za1.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za2.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za3.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za0.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za1.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za2.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za3.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za0.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za1.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za2.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          "fmopa za3.s, %[p]/m, %[p]/m, %[a].h, %[b].h\n\t"
          :
          : [p] "Upa"(pg), [a] "w"(a), [b] "w"(b)
          : "memory");
    }
    chunks++;
  }
  return chunks * 4096LL * 16LL;
}

__arm_new("za") __arm_locally_streaming static int64_t
    sme2_fmopa_fp32_loop(const atomic_int *stop_flag) {
  svbool_t pg = svptrue_b32();
  svfloat32_t a = svdup_f32(1.0f);
  svfloat32_t b = svdup_f32(1.0f);
  svzero_za();
  int64_t chunks = 0;
  while (!atomic_load_explicit(stop_flag, memory_order_relaxed)) {
    for (int i = 0; i < 4096; ++i) {
      __asm__ volatile(
          "fmopa za0.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za1.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za2.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za3.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za0.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za1.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za2.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za3.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za0.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za1.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za2.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za3.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za0.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za1.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za2.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          "fmopa za3.s, %[p]/m, %[p]/m, %[a].s, %[b].s\n\t"
          :
          : [p] "Upa"(pg), [a] "w"(a), [b] "w"(b)
          : "memory");
    }
    chunks++;
  }
  return chunks * 4096LL * 16LL;
}

static int64_t neon_i8mm_smmla_loop(const atomic_int *stop_flag) {
  int64_t chunks = 0;
  __asm__ volatile(
      "movi v0.16b, #0\n\t"
      "movi v1.16b, #0\n\t"
      "movi v2.16b, #0\n\t"
      "movi v3.16b, #0\n\t"
      "movi v4.16b, #0\n\t"
      "movi v5.16b, #0\n\t"
      "movi v6.16b, #0\n\t"
      "movi v7.16b, #0\n\t"
      "movi v16.16b, #1\n\t"
      "movi v17.16b, #1\n\t"
      "movi v18.16b, #0\n\t"
      "movi v19.16b, #0\n\t"
      "movi v20.16b, #0\n\t"
      "movi v21.16b, #0\n\t"
      ::: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18",
          "v19", "v20", "v21");
  while (!atomic_load_explicit(stop_flag, memory_order_relaxed)) {
    for (int i = 0; i < 16384; ++i) {
      __asm__ volatile(
          "smmla v0.4s, v16.16b, v17.16b\n\t"
          "smmla v1.4s, v16.16b, v17.16b\n\t"
          "smmla v2.4s, v16.16b, v17.16b\n\t"
          "smmla v3.4s, v16.16b, v17.16b\n\t"
          "smmla v4.4s, v16.16b, v17.16b\n\t"
          "smmla v5.4s, v16.16b, v17.16b\n\t"
          "smmla v6.4s, v16.16b, v17.16b\n\t"
          "smmla v7.4s, v16.16b, v17.16b\n\t"
          "smmla v18.4s, v16.16b, v17.16b\n\t"
          "smmla v19.4s, v16.16b, v17.16b\n\t"
          "smmla v20.4s, v16.16b, v17.16b\n\t"
          "smmla v21.4s, v16.16b, v17.16b\n\t"
          "smmla v0.4s, v16.16b, v17.16b\n\t"
          "smmla v1.4s, v16.16b, v17.16b\n\t"
          "smmla v2.4s, v16.16b, v17.16b\n\t"
          "smmla v3.4s, v16.16b, v17.16b\n\t"
          "smmla v4.4s, v16.16b, v17.16b\n\t"
          "smmla v5.4s, v16.16b, v17.16b\n\t"
          "smmla v6.4s, v16.16b, v17.16b\n\t"
          "smmla v7.4s, v16.16b, v17.16b\n\t"
          "smmla v18.4s, v16.16b, v17.16b\n\t"
          "smmla v19.4s, v16.16b, v17.16b\n\t"
          "smmla v20.4s, v16.16b, v17.16b\n\t"
          "smmla v21.4s, v16.16b, v17.16b\n\t"
          ::: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v18", "v19",
              "v20", "v21");
    }
    chunks++;
  }
  return chunks * 16384LL * 24LL;
}

static int64_t neon_dotprod_sdot_loop(const atomic_int *stop_flag) {
  int64_t chunks = 0;
  __asm__ volatile(
      "movi v0.16b, #0\n\t"
      "movi v1.16b, #0\n\t"
      "movi v2.16b, #0\n\t"
      "movi v3.16b, #0\n\t"
      "movi v4.16b, #0\n\t"
      "movi v5.16b, #0\n\t"
      "movi v6.16b, #0\n\t"
      "movi v7.16b, #0\n\t"
      "movi v16.16b, #1\n\t"
      "movi v17.16b, #1\n\t"
      "movi v18.16b, #0\n\t"
      "movi v19.16b, #0\n\t"
      "movi v20.16b, #0\n\t"
      "movi v21.16b, #0\n\t"
      ::: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18",
          "v19", "v20", "v21");
  while (!atomic_load_explicit(stop_flag, memory_order_relaxed)) {
    for (int i = 0; i < 16384; ++i) {
      __asm__ volatile(
          "sdot v0.4s, v16.16b, v17.16b\n\t"
          "sdot v1.4s, v16.16b, v17.16b\n\t"
          "sdot v2.4s, v16.16b, v17.16b\n\t"
          "sdot v3.4s, v16.16b, v17.16b\n\t"
          "sdot v4.4s, v16.16b, v17.16b\n\t"
          "sdot v5.4s, v16.16b, v17.16b\n\t"
          "sdot v6.4s, v16.16b, v17.16b\n\t"
          "sdot v7.4s, v16.16b, v17.16b\n\t"
          "sdot v18.4s, v16.16b, v17.16b\n\t"
          "sdot v19.4s, v16.16b, v17.16b\n\t"
          "sdot v20.4s, v16.16b, v17.16b\n\t"
          "sdot v21.4s, v16.16b, v17.16b\n\t"
          "sdot v0.4s, v16.16b, v17.16b\n\t"
          "sdot v1.4s, v16.16b, v17.16b\n\t"
          "sdot v2.4s, v16.16b, v17.16b\n\t"
          "sdot v3.4s, v16.16b, v17.16b\n\t"
          "sdot v4.4s, v16.16b, v17.16b\n\t"
          "sdot v5.4s, v16.16b, v17.16b\n\t"
          "sdot v6.4s, v16.16b, v17.16b\n\t"
          "sdot v7.4s, v16.16b, v17.16b\n\t"
          "sdot v18.4s, v16.16b, v17.16b\n\t"
          "sdot v19.4s, v16.16b, v17.16b\n\t"
          "sdot v20.4s, v16.16b, v17.16b\n\t"
          "sdot v21.4s, v16.16b, v17.16b\n\t"
          ::: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v18", "v19",
              "v20", "v21");
    }
    chunks++;
  }
  return chunks * 16384LL * 24LL;
}

static int64_t neon_fp32_fmla_loop(const atomic_int *stop_flag) {
  int64_t chunks = 0;
  __asm__ volatile(
      "movi v0.4s, #0\n\t"
      "movi v1.4s, #0\n\t"
      "movi v2.4s, #0\n\t"
      "movi v3.4s, #0\n\t"
      "movi v4.4s, #0\n\t"
      "movi v5.4s, #0\n\t"
      "movi v6.4s, #0\n\t"
      "movi v7.4s, #0\n\t"
      "fmov v16.4s, #1.0\n\t"
      "fmov v17.4s, #1.0\n\t"
      "movi v18.4s, #0\n\t"
      "movi v19.4s, #0\n\t"
      "movi v20.4s, #0\n\t"
      "movi v21.4s, #0\n\t"
      ::: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18",
          "v19", "v20", "v21");
  while (!atomic_load_explicit(stop_flag, memory_order_relaxed)) {
    for (int i = 0; i < 16384; ++i) {
      __asm__ volatile(
          "fmla v0.4s, v16.4s, v17.4s\n\t"
          "fmla v1.4s, v16.4s, v17.4s\n\t"
          "fmla v2.4s, v16.4s, v17.4s\n\t"
          "fmla v3.4s, v16.4s, v17.4s\n\t"
          "fmla v4.4s, v16.4s, v17.4s\n\t"
          "fmla v5.4s, v16.4s, v17.4s\n\t"
          "fmla v6.4s, v16.4s, v17.4s\n\t"
          "fmla v7.4s, v16.4s, v17.4s\n\t"
          "fmla v18.4s, v16.4s, v17.4s\n\t"
          "fmla v19.4s, v16.4s, v17.4s\n\t"
          "fmla v20.4s, v16.4s, v17.4s\n\t"
          "fmla v21.4s, v16.4s, v17.4s\n\t"
          "fmla v0.4s, v16.4s, v17.4s\n\t"
          "fmla v1.4s, v16.4s, v17.4s\n\t"
          "fmla v2.4s, v16.4s, v17.4s\n\t"
          "fmla v3.4s, v16.4s, v17.4s\n\t"
          "fmla v4.4s, v16.4s, v17.4s\n\t"
          "fmla v5.4s, v16.4s, v17.4s\n\t"
          "fmla v6.4s, v16.4s, v17.4s\n\t"
          "fmla v7.4s, v16.4s, v17.4s\n\t"
          "fmla v18.4s, v16.4s, v17.4s\n\t"
          "fmla v19.4s, v16.4s, v17.4s\n\t"
          "fmla v20.4s, v16.4s, v17.4s\n\t"
          "fmla v21.4s, v16.4s, v17.4s\n\t"
          ::: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v18", "v19",
              "v20", "v21");
    }
    chunks++;
  }
  return chunks * 16384LL * 24LL;
}

typedef struct {
  int tid;
  compute_kind_t kind;
  int num_trials;
  spin_barrier_t *bar;
  const atomic_int *stop_flag;
  int64_t counts[MAX_TRIALS];
} comp_worker_t;

static int64_t dispatch_compute_loop(compute_kind_t kind,
                                     const atomic_int *stop_flag) {
  switch (kind) {
    case COMP_SME2_INT8_SMOPA:
      return sme2_smopa_int8_loop(stop_flag);
    case COMP_SME2_BF16_BFMOPA:
      return sme2_bfmopa_bf16_loop(stop_flag);
    case COMP_SME2_FP16_FMOPA:
      return sme2_fmopa_fp16_loop(stop_flag);
    case COMP_SME2_FP32_FMOPA:
      return sme2_fmopa_fp32_loop(stop_flag);
    case COMP_NEON_I8MM_SMMLA:
      return neon_i8mm_smmla_loop(stop_flag);
    case COMP_NEON_DOTPROD_SDOT:
      return neon_dotprod_sdot_loop(stop_flag);
    case COMP_NEON_FP32_FMLA:
      return neon_fp32_fmla_loop(stop_flag);
  }
  return 0;
}

static void *comp_worker_fn(void *arg) {
  comp_worker_t *w = (comp_worker_t *)arg;
  set_high_qos();
  // Warmup window
  sb_wait(w->bar);
  (void)dispatch_compute_loop(w->kind, w->stop_flag);
  sb_wait(w->bar);
  // Timed trials
  for (int t = 0; t < w->num_trials; ++t) {
    sb_wait(w->bar);
    w->counts[t] = dispatch_compute_loop(w->kind, w->stop_flag);
    sb_wait(w->bar);
  }
  return NULL;
}

static void sleep_sec(double s) {
  struct timespec ts;
  ts.tv_sec = (time_t)s;
  ts.tv_nsec = (long)((s - (double)ts.tv_sec) * 1e9);
  nanosleep(&ts, NULL);
}

static probe_stats_t run_compute_probe(int num_threads, compute_kind_t kind,
                                       double ops_per_inst, int num_trials,
                                       double trial_sec) {
  spin_barrier_t bar;
  sb_init(&bar, num_threads + 1);
  atomic_int stop_flag = 0;
  pthread_t th[MAX_THREADS];
  comp_worker_t w[MAX_THREADS];
  double samples[MAX_TRIALS];

  for (int i = 0; i < num_threads; ++i) {
    w[i].tid = i;
    w[i].kind = kind;
    w[i].num_trials = num_trials;
    w[i].bar = &bar;
    w[i].stop_flag = &stop_flag;
    pthread_create(&th[i], NULL, comp_worker_fn, &w[i]);
  }

  // Warmup
  atomic_store_explicit(&stop_flag, 0, memory_order_release);
  sb_wait(&bar);
  sleep_sec(trial_sec * 0.5);
  atomic_store_explicit(&stop_flag, 1, memory_order_release);
  sb_wait(&bar);

  for (int t = 0; t < num_trials; ++t) {
    atomic_store_explicit(&stop_flag, 0, memory_order_release);
    sb_wait(&bar);
    double t0 = now_sec();
    sleep_sec(trial_sec);
    atomic_store_explicit(&stop_flag, 1, memory_order_release);
    sb_wait(&bar);
    double t1 = now_sec();
    int64_t total_insts = 0;
    for (int i = 0; i < num_threads; ++i) total_insts += w[i].counts[t];
    samples[t] = (double)total_insts * ops_per_inst / (t1 - t0) / 1e9;
  }

  for (int i = 0; i < num_threads; ++i) pthread_join(th[i], NULL);
  return compute_stats(samples, num_trials);
}

// -----------------------------------------------------------------------------
// 2. Memory-Hierarchy Bandwidth Ceiling Probes: BW_prac(WorkingSet)
// -----------------------------------------------------------------------------

typedef enum {
  BW_MODE_READ_ONLY = 0,
  BW_MODE_READ_WRITE_COPY = 1,
  BW_MODE_STREAM_TRIAD = 2,
} bw_mode_t;

typedef struct {
  int tid;
  bw_mode_t mode;
  size_t num_floats;
  int batch_passes;
  const float *a;
  float *b;
  float *c;
  int num_trials;
  spin_barrier_t *bar;
  const atomic_int *stop_flag;
  int64_t passes[MAX_TRIALS];
} bw_worker_t;

static int64_t run_bw_window(const bw_worker_t *w) {
  const size_t n = w->num_floats;
  const float *a = w->a;
  float *b = w->b;
  float *c = w->c;
  const int bp = w->batch_passes;
  int64_t done = 0;

  if (w->mode == BW_MODE_READ_ONLY) {
    const uint32_t *au = (const uint32_t *)a;
    uint32x4_t acc0 = vdupq_n_u32(0);
    uint32x4_t acc1 = vdupq_n_u32(0);
    uint32x4_t acc2 = vdupq_n_u32(0);
    uint32x4_t acc3 = vdupq_n_u32(0);
    uint32x4_t acc4 = vdupq_n_u32(0);
    uint32x4_t acc5 = vdupq_n_u32(0);
    uint32x4_t acc6 = vdupq_n_u32(0);
    uint32x4_t acc7 = vdupq_n_u32(0);
    while (!atomic_load_explicit(w->stop_flag, memory_order_relaxed)) {
      for (int p = 0; p < bp; ++p) {
        for (size_t i = 0; i < n; i += 32) {
          acc0 = vorrq_u32(acc0, vld1q_u32(au + i + 0));
          acc1 = vorrq_u32(acc1, vld1q_u32(au + i + 4));
          acc2 = vorrq_u32(acc2, vld1q_u32(au + i + 8));
          acc3 = vorrq_u32(acc3, vld1q_u32(au + i + 12));
          acc4 = vorrq_u32(acc4, vld1q_u32(au + i + 16));
          acc5 = vorrq_u32(acc5, vld1q_u32(au + i + 20));
          acc6 = vorrq_u32(acc6, vld1q_u32(au + i + 24));
          acc7 = vorrq_u32(acc7, vld1q_u32(au + i + 28));
        }
        __asm__ volatile(""
                         : "+w"(acc0), "+w"(acc1), "+w"(acc2), "+w"(acc3),
                           "+w"(acc4), "+w"(acc5), "+w"(acc6), "+w"(acc7));
      }
      done += bp;
    }
  } else if (w->mode == BW_MODE_READ_WRITE_COPY) {
    while (!atomic_load_explicit(w->stop_flag, memory_order_relaxed)) {
      for (int p = 0; p < bp; ++p) {
        for (size_t i = 0; i < n; i += 32) {
          float32x4_t v0 = vld1q_f32(a + i + 0);
          float32x4_t v1 = vld1q_f32(a + i + 4);
          float32x4_t v2 = vld1q_f32(a + i + 8);
          float32x4_t v3 = vld1q_f32(a + i + 12);
          float32x4_t v4 = vld1q_f32(a + i + 16);
          float32x4_t v5 = vld1q_f32(a + i + 20);
          float32x4_t v6 = vld1q_f32(a + i + 24);
          float32x4_t v7 = vld1q_f32(a + i + 28);
          vst1q_f32(b + i + 0, v0);
          vst1q_f32(b + i + 4, v1);
          vst1q_f32(b + i + 8, v2);
          vst1q_f32(b + i + 12, v3);
          vst1q_f32(b + i + 16, v4);
          vst1q_f32(b + i + 20, v5);
          vst1q_f32(b + i + 24, v6);
          vst1q_f32(b + i + 28, v7);
        }
        __asm__ volatile("" ::: "memory");
      }
      done += bp;
    }
  } else {
    float32x4_t vscale = vdupq_n_f32(0.5f);
    while (!atomic_load_explicit(w->stop_flag, memory_order_relaxed)) {
      for (int p = 0; p < bp; ++p) {
        for (size_t i = 0; i < n; i += 16) {
          float32x4_t a0 = vld1q_f32(a + i + 0);
          float32x4_t a1 = vld1q_f32(a + i + 4);
          float32x4_t a2 = vld1q_f32(a + i + 8);
          float32x4_t a3 = vld1q_f32(a + i + 12);
          float32x4_t b0 = vld1q_f32(b + i + 0);
          float32x4_t b1 = vld1q_f32(b + i + 4);
          float32x4_t b2 = vld1q_f32(b + i + 8);
          float32x4_t b3 = vld1q_f32(b + i + 12);
          vst1q_f32(c + i + 0, vfmaq_f32(b0, vscale, a0));
          vst1q_f32(c + i + 4, vfmaq_f32(b1, vscale, a1));
          vst1q_f32(c + i + 8, vfmaq_f32(b2, vscale, a2));
          vst1q_f32(c + i + 12, vfmaq_f32(b3, vscale, a3));
        }
        __asm__ volatile("" ::: "memory");
      }
      done += bp;
    }
  }
  return done;
}

static void *bw_worker_fn(void *arg) {
  bw_worker_t *w = (bw_worker_t *)arg;
  set_high_qos();
  // Warmup
  sb_wait(w->bar);
  (void)run_bw_window(w);
  sb_wait(w->bar);
  // Timed trials
  for (int t = 0; t < w->num_trials; ++t) {
    sb_wait(w->bar);
    w->passes[t] = run_bw_window(w);
    sb_wait(w->bar);
  }
  return NULL;
}

static probe_stats_t run_bw_probe(int num_threads, size_t bytes_per_thread,
                                  int batch_passes, bw_mode_t mode,
                                  int num_trials, double trial_sec) {
  size_t aligned_bytes = (bytes_per_thread + 255) & ~(size_t)255;
  size_t num_floats = aligned_bytes / sizeof(float);
  // Non-zero 16K Apple Silicon page-offset delta (+1024B) between a/b/c
  size_t stride_bytes =
      aligned_bytes + (((aligned_bytes & 16383) == 0) ? 1024 : 0);
  size_t total_thread_bytes = 3 * stride_bytes + 128;

  uint8_t *raw_bufs[MAX_THREADS] = {0};
  spin_barrier_t bar;
  sb_init(&bar, num_threads + 1);
  atomic_int stop_flag = 0;
  pthread_t th[MAX_THREADS];
  bw_worker_t w[MAX_THREADS];

  for (int i = 0; i < num_threads; ++i) {
    posix_memalign((void **)&raw_bufs[i], 128, total_thread_bytes);
    float *a = (float *)(raw_bufs[i] + 0);
    float *b = (float *)(raw_bufs[i] + stride_bytes);
    float *c = (float *)(raw_bufs[i] + 2 * stride_bytes);
    for (size_t j = 0; j < num_floats; ++j) {
      a[j] = 1.0f;
      b[j] = 2.0f;
      c[j] = 0.5f;
    }
    w[i].tid = i;
    w[i].mode = mode;
    w[i].num_floats = num_floats;
    w[i].batch_passes = batch_passes;
    w[i].a = a;
    w[i].b = b;
    w[i].c = c;
    w[i].num_trials = num_trials;
    w[i].bar = &bar;
    w[i].stop_flag = &stop_flag;
    pthread_create(&th[i], NULL, bw_worker_fn, &w[i]);
  }

  double traffic_mult = (mode == BW_MODE_READ_ONLY)
                            ? 1.0
                            : (mode == BW_MODE_READ_WRITE_COPY ? 2.0 : 3.0);

  // Warmup
  atomic_store_explicit(&stop_flag, 0, memory_order_release);
  sb_wait(&bar);
  sleep_sec(trial_sec * 0.4);
  atomic_store_explicit(&stop_flag, 1, memory_order_release);
  sb_wait(&bar);

  double samples[MAX_TRIALS];
  for (int t = 0; t < num_trials; ++t) {
    atomic_store_explicit(&stop_flag, 0, memory_order_release);
    sb_wait(&bar);
    double t0 = now_sec();
    sleep_sec(trial_sec);
    atomic_store_explicit(&stop_flag, 1, memory_order_release);
    sb_wait(&bar);
    double t1 = now_sec();
    int64_t total_passes = 0;
    for (int i = 0; i < num_threads; ++i) total_passes += w[i].passes[t];
    double total_bytes =
        (double)total_passes * (double)aligned_bytes * traffic_mult;
    samples[t] = total_bytes / (t1 - t0) / 1e9;
  }

  for (int i = 0; i < num_threads; ++i) {
    pthread_join(th[i], NULL);
    free(raw_bufs[i]);
  }
  return compute_stats(samples, num_trials);
}

// Shared System Level Cache (SLC) probe: streams staggered windows across a
// 48 MB shared pool (> 16 MB Super L2 / 8 MB Perf L2, < ~48-64 MB SLC).
static probe_stats_t run_shared_slc_bw_probe(int num_threads, bw_mode_t mode,
                                             int num_trials, double trial_sec) {
  size_t pool_bytes =
      (mode == BW_MODE_READ_ONLY) ? (48ULL * 1024 * 1024) : (24ULL * 1024 * 1024);
  size_t window_bytes =
      (mode == BW_MODE_READ_ONLY) ? (24ULL * 1024 * 1024) : (12ULL * 1024 * 1024);
  size_t pool_floats = pool_bytes / sizeof(float);
  size_t window_floats = window_bytes / sizeof(float);

  float *pool_a = NULL, *pool_b = NULL;
  posix_memalign((void **)&pool_a, 128, pool_bytes);
  posix_memalign((void **)&pool_b, 128, pool_bytes);
  for (size_t i = 0; i < pool_floats; ++i) {
    pool_a[i] = 1.0f;
    pool_b[i] = 2.0f;
  }

  spin_barrier_t bar;
  sb_init(&bar, num_threads + 1);
  atomic_int stop_flag = 0;
  pthread_t th[MAX_THREADS];
  bw_worker_t w[MAX_THREADS];

  size_t max_off = pool_floats - window_floats;
  for (int i = 0; i < num_threads; ++i) {
    size_t off = (max_off > 0)
                     ? (((size_t)i * 512 * 1024 / sizeof(float)) % max_off)
                     : 0;
    off &= ~(size_t)31;
    w[i].tid = i;
    w[i].mode = mode;
    w[i].num_floats = window_floats;
    w[i].batch_passes = 1;
    w[i].a = pool_a + off;
    w[i].b = pool_b + off;
    w[i].c = pool_b + off;
    w[i].num_trials = num_trials;
    w[i].bar = &bar;
    w[i].stop_flag = &stop_flag;
    pthread_create(&th[i], NULL, bw_worker_fn, &w[i]);
  }

  double traffic_mult = (mode == BW_MODE_READ_ONLY) ? 1.0 : 2.0;

  atomic_store_explicit(&stop_flag, 0, memory_order_release);
  sb_wait(&bar);
  sleep_sec(trial_sec * 0.4);
  atomic_store_explicit(&stop_flag, 1, memory_order_release);
  sb_wait(&bar);

  double samples[MAX_TRIALS];
  for (int t = 0; t < num_trials; ++t) {
    atomic_store_explicit(&stop_flag, 0, memory_order_release);
    sb_wait(&bar);
    double t0 = now_sec();
    sleep_sec(trial_sec);
    atomic_store_explicit(&stop_flag, 1, memory_order_release);
    sb_wait(&bar);
    double t1 = now_sec();
    int64_t total_passes = 0;
    for (int i = 0; i < num_threads; ++i) total_passes += w[i].passes[t];
    double total_bytes =
        (double)total_passes * (double)window_bytes * traffic_mult;
    samples[t] = total_bytes / (t1 - t0) / 1e9;
  }

  for (int i = 0; i < num_threads; ++i) pthread_join(th[i], NULL);
  free(pool_a);
  free(pool_b);
  return compute_stats(samples, num_trials);
}

static void print_stat_obj(FILE *out, const char *indent, const char *name,
                           probe_stats_t s1, const probe_stats_t *smc,
                           const int *mc_threads, int num_mc,
                           int trailing_comma) {
  fprintf(out, "%s\"%s\": {\n", indent, name);
  fprintf(out,
          "%s  \"single_core\": %.2f,\n"
          "%s  \"single_core_stats\": { \"median\": %.2f, \"mean\": %.2f, "
          "\"min\": %.2f, \"max\": %.2f, \"stdev\": %.2f, \"cv_pct\": %.2f }%s\n",
          indent, s1.median, indent, s1.median, s1.mean, s1.min, s1.max,
          s1.stdev, s1.cv_pct, num_mc > 0 ? "," : "");
  for (int i = 0; i < num_mc; ++i) {
    int comma = (i + 1 < num_mc);
    fprintf(out,
            "%s  \"threads_%dc\": %.2f,\n"
            "%s  \"threads_%dc_stats\": { \"median\": %.2f, \"mean\": %.2f, "
            "\"min\": %.2f, \"max\": %.2f, \"stdev\": %.2f, \"cv_pct\": %.2f }%s\n",
            indent, mc_threads[i], smc[i].median, indent, mc_threads[i],
            smc[i].median, smc[i].mean, smc[i].min, smc[i].max, smc[i].stdev,
            smc[i].cv_pct, comma ? "," : "");
  }
  fprintf(out, "%s}%s\n", indent, trailing_comma ? "," : "");
}

static int parse_thread_list(const char *arg, int *out, int max_count) {
  int count = 0;
  const char *p = arg;
  while (*p && count < max_count) {
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p) break;
    if (v > 1 && v <= MAX_THREADS) out[count++] = (int)v;
    if (*end == ',') {
      p = end + 1;
    } else {
      break;
    }
  }
  return count;
}

int main(int argc, char **argv) {
  int mc_threads[MAX_THREAD_CFGS] = {6, 18};
  int num_mc = 2;
  int num_trials = 7;
  double trial_sec = 0.25;
  const char *out_path = NULL;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      int n = parse_thread_list(argv[++i], mc_threads, MAX_THREAD_CFGS);
      if (n > 0) num_mc = n;
    } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
      num_trials = atoi(argv[++i]);
      if (num_trials < 1) num_trials = 1;
      if (num_trials > MAX_TRIALS) num_trials = MAX_TRIALS;
    } else if (strcmp(argv[i], "--trial-sec") == 0 && i + 1 < argc) {
      trial_sec = atof(argv[++i]);
      if (trial_sec < 0.05) trial_sec = 0.05;
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    }
  }

  set_high_qos();
  char cpu_brand[128];
  read_cpu_brand(cpu_brand, sizeof(cpu_brand));
  uint64_t svl_bytes = get_sme_svl_bytes();
  double svl_w = (double)(svl_bytes / 4);  // 16 lanes for 64-byte SVL

  int max_mc = mc_threads[0];
  for (int i = 1; i < num_mc; ++i) {
    if (mc_threads[i] > max_mc) max_mc = mc_threads[i];
  }

  double ops_smopa_i8 = svl_w * svl_w * 4.0 * 2.0;    // 2048 OPs/inst
  double ops_bfmopa_bf16 = svl_w * svl_w * 2.0 * 2.0; // 1024 FLOPs/inst
  double ops_fmopa_fp16 = svl_w * svl_w * 2.0 * 2.0;  // 1024 FLOPs/inst
  double ops_fmopa_fp32 = svl_w * svl_w * 1.0 * 2.0;  // 512 FLOPs/inst
  double ops_i8mm_smmla = 32.0 * 2.0;                 // 64 OPs/inst
  double ops_dot_sdot = 16.0 * 2.0;                   // 32 OPs/inst
  double ops_neon_fmla = 4.0 * 2.0;                   // 8 FLOPs/inst

  fprintf(stderr, "[cpu_ceilings_arm64] Host: %s (SME SVL=%llu B, trials=%d)\n",
          cpu_brand, (unsigned long long)svl_bytes, num_trials);

  // Single-core (1T) probes
  fprintf(stderr, "[cpu_ceilings_arm64] Running 1T probes...\n");
  probe_stats_t sme_i8_1c = run_compute_probe(1, COMP_SME2_INT8_SMOPA,
                                              ops_smopa_i8, num_trials, trial_sec);
  probe_stats_t sme_bf16_1c = run_compute_probe(
      1, COMP_SME2_BF16_BFMOPA, ops_bfmopa_bf16, num_trials, trial_sec);
  probe_stats_t sme_fp16_1c = run_compute_probe(
      1, COMP_SME2_FP16_FMOPA, ops_fmopa_fp16, num_trials, trial_sec);
  probe_stats_t sme_fp32_1c = run_compute_probe(
      1, COMP_SME2_FP32_FMOPA, ops_fmopa_fp32, num_trials, trial_sec);
  probe_stats_t i8mm_1c = run_compute_probe(1, COMP_NEON_I8MM_SMMLA,
                                            ops_i8mm_smmla, num_trials, trial_sec);
  probe_stats_t sdot_1c = run_compute_probe(1, COMP_NEON_DOTPROD_SDOT,
                                            ops_dot_sdot, num_trials, trial_sec);
  probe_stats_t fmla_1c = run_compute_probe(1, COMP_NEON_FP32_FMLA,
                                            ops_neon_fmla, num_trials, trial_sec);

  probe_stats_t l1_rd_1c = run_bw_probe(1, 32 * 1024, 256, BW_MODE_READ_ONLY,
                                        num_trials, trial_sec);
  probe_stats_t l1_cp_1c = run_bw_probe(
      1, 16 * 1024, 256, BW_MODE_READ_WRITE_COPY, num_trials, trial_sec);
  probe_stats_t l2_rd_1c = run_bw_probe(1, 1024 * 1024, 16, BW_MODE_READ_ONLY,
                                        num_trials, trial_sec);
  probe_stats_t l2_cp_1c = run_bw_probe(
      1, 512 * 1024, 16, BW_MODE_READ_WRITE_COPY, num_trials, trial_sec);
  probe_stats_t slc_rd_1c =
      run_bw_probe(1, 32 * 1024 * 1024, 1, BW_MODE_READ_ONLY, num_trials, trial_sec);
  probe_stats_t slc_cp_1c = run_bw_probe(
      1, 16 * 1024 * 1024, 1, BW_MODE_READ_WRITE_COPY, num_trials, trial_sec);
  probe_stats_t mem_rd_1c = run_bw_probe(
      1, 256 * 1024 * 1024, 1, BW_MODE_READ_ONLY, num_trials, trial_sec);
  probe_stats_t mem_cp_1c = run_bw_probe(
      1, 128 * 1024 * 1024, 1, BW_MODE_READ_WRITE_COPY, num_trials, trial_sec);
  probe_stats_t mem_tr_1c = run_bw_probe(
      1, 128 * 1024 * 1024, 1, BW_MODE_STREAM_TRIAD, num_trials, trial_sec);

  probe_stats_t sme_i8_mc[MAX_THREAD_CFGS], sme_bf16_mc[MAX_THREAD_CFGS];
  probe_stats_t sme_fp16_mc[MAX_THREAD_CFGS], sme_fp32_mc[MAX_THREAD_CFGS];
  probe_stats_t i8mm_mc[MAX_THREAD_CFGS], sdot_mc[MAX_THREAD_CFGS],
      fmla_mc[MAX_THREAD_CFGS];
  probe_stats_t l1_rd_mc[MAX_THREAD_CFGS], l1_cp_mc[MAX_THREAD_CFGS];
  probe_stats_t l2_rd_mc[MAX_THREAD_CFGS], l2_cp_mc[MAX_THREAD_CFGS];
  probe_stats_t slc_rd_mc[MAX_THREAD_CFGS], slc_cp_mc[MAX_THREAD_CFGS];
  probe_stats_t mem_rd_mc[MAX_THREAD_CFGS], mem_cp_mc[MAX_THREAD_CFGS],
      mem_tr_mc[MAX_THREAD_CFGS];

  for (int t = 0; t < num_mc; ++t) {
    int nt = mc_threads[t];
    fprintf(stderr, "[cpu_ceilings_arm64] Running %dT probes...\n", nt);
    sme_i8_mc[t] = run_compute_probe(nt, COMP_SME2_INT8_SMOPA, ops_smopa_i8,
                                     num_trials, trial_sec);
    sme_bf16_mc[t] = run_compute_probe(nt, COMP_SME2_BF16_BFMOPA,
                                       ops_bfmopa_bf16, num_trials, trial_sec);
    sme_fp16_mc[t] = run_compute_probe(nt, COMP_SME2_FP16_FMOPA, ops_fmopa_fp16,
                                       num_trials, trial_sec);
    sme_fp32_mc[t] = run_compute_probe(nt, COMP_SME2_FP32_FMOPA, ops_fmopa_fp32,
                                       num_trials, trial_sec);
    i8mm_mc[t] = run_compute_probe(nt, COMP_NEON_I8MM_SMMLA, ops_i8mm_smmla,
                                   num_trials, trial_sec);
    sdot_mc[t] = run_compute_probe(nt, COMP_NEON_DOTPROD_SDOT, ops_dot_sdot,
                                   num_trials, trial_sec);
    fmla_mc[t] = run_compute_probe(nt, COMP_NEON_FP32_FMLA, ops_neon_fmla,
                                   num_trials, trial_sec);

    l1_rd_mc[t] = run_bw_probe(nt, 32 * 1024, 256, BW_MODE_READ_ONLY, num_trials,
                               trial_sec);
    l1_cp_mc[t] = run_bw_probe(nt, 16 * 1024, 256, BW_MODE_READ_WRITE_COPY,
                               num_trials, trial_sec);
    l2_rd_mc[t] = run_bw_probe(nt, 1024 * 1024, 16, BW_MODE_READ_ONLY,
                               num_trials, trial_sec);
    l2_cp_mc[t] = run_bw_probe(nt, 512 * 1024, 16, BW_MODE_READ_WRITE_COPY,
                               num_trials, trial_sec);
    slc_rd_mc[t] =
        run_shared_slc_bw_probe(nt, BW_MODE_READ_ONLY, num_trials, trial_sec);
    slc_cp_mc[t] = run_shared_slc_bw_probe(nt, BW_MODE_READ_WRITE_COPY,
                                           num_trials, trial_sec);

    size_t mem_per_t = (1024ULL * 1024 * 1024) / (size_t)nt;
    if (mem_per_t < 32ULL * 1024 * 1024) mem_per_t = 32ULL * 1024 * 1024;
    mem_rd_mc[t] =
        run_bw_probe(nt, mem_per_t, 1, BW_MODE_READ_ONLY, num_trials, trial_sec);
    mem_cp_mc[t] = run_bw_probe(nt, mem_per_t / 2, 1, BW_MODE_READ_WRITE_COPY,
                                num_trials, trial_sec);
    mem_tr_mc[t] = run_bw_probe(nt, mem_per_t / 2, 1, BW_MODE_STREAM_TRIAD,
                                num_trials, trial_sec);
  }

  FILE *outs[2] = {stdout, NULL};
  int num_outs = 1;
  if (out_path) {
    outs[1] = fopen(out_path, "w");
    if (outs[1]) num_outs = 2;
  }

  for (int o = 0; o < num_outs; ++o) {
    FILE *out = outs[o];
    fprintf(out, "{\n");
    fprintf(out, "  \"arch\": \"arm64\",\n");
    fprintf(out, "  \"microarchitecture\": \"%s\",\n", cpu_brand);
    fprintf(out, "  \"sme_svl_bytes\": %llu,\n", (unsigned long long)svl_bytes);
    fprintf(out, "  \"topology\": {\n");
    fprintf(out, "    \"total_cores\": 18,\n");
    fprintf(out, "    \"super_cores\": 6,\n");
    fprintf(out, "    \"performance_cores\": 12,\n");
    fprintf(out, "    \"l1d_super_kb\": 128,\n");
    fprintf(out, "    \"l1d_perf_kb\": 64,\n");
    fprintf(out, "    \"l2_super_cluster_mb\": 16,\n");
    fprintf(out, "    \"l2_perf_cluster_mb\": 8,\n");
    fprintf(out, "    \"l2_perf_clusters\": 2,\n");
    fprintf(out, "    \"cache_line_bytes\": 128,\n");
    fprintf(out, "    \"unified_memory_gb\": 36\n");
    fprintf(out, "  },\n");
    fprintf(out, "  \"methodology\": {\n");
    fprintf(out, "    \"clock\": \"CLOCK_MONOTONIC_RAW\",\n");
    fprintf(out,
            "    \"scheduling\": \"pthreads + QOS_CLASS_USER_INTERACTIVE "
            "(Super cores prioritized at 1T/6T; full 3 clusters at 18T)\",\n");
    fprintf(out,
            "    \"synchronization\": \"atomic spin-wait barrier + time-windowed "
            "heterogeneous work accounting\",\n");
    fprintf(out, "    \"num_trials\": %d,\n", num_trials);
    fprintf(out, "    \"trial_window_sec\": %.2f,\n", trial_sec);
    fprintf(out, "    \"reported_value\": \"median\"\n");
    fprintf(out, "  },\n");
    fprintf(out, "  \"threads_single_core\": 1,\n");
    fprintf(out, "  \"threads_max\": %d,\n", max_mc);
    fprintf(out, "  \"thread_configs\": [1");
    for (int t = 0; t < num_mc; ++t) fprintf(out, ", %d", mc_threads[t]);
    fprintf(out, "],\n");
    fprintf(out, "  \"compute_ceilings_gops\": {\n");
    print_stat_obj(out, "    ", "sme2_int8_smopa", sme_i8_1c, sme_i8_mc,
                   mc_threads, num_mc, 1);
    print_stat_obj(out, "    ", "sme2_bf16_bfmopa", sme_bf16_1c, sme_bf16_mc,
                   mc_threads, num_mc, 1);
    print_stat_obj(out, "    ", "sme2_fp16_fmopa", sme_fp16_1c, sme_fp16_mc,
                   mc_threads, num_mc, 1);
    print_stat_obj(out, "    ", "sme2_fp32_fmopa", sme_fp32_1c, sme_fp32_mc,
                   mc_threads, num_mc, 1);
    print_stat_obj(out, "    ", "neon_i8mm_smmla", i8mm_1c, i8mm_mc, mc_threads,
                   num_mc, 1);
    print_stat_obj(out, "    ", "neon_dotprod_sdot", sdot_1c, sdot_mc,
                   mc_threads, num_mc, 1);
    print_stat_obj(out, "    ", "neon_fp32_fmla", fmla_1c, fmla_mc, mc_threads,
                   num_mc, 0);
    fprintf(out, "  },\n");
    fprintf(out, "  \"bandwidth_ceilings_gbps\": {\n");
    fprintf(out, "    \"l1d_32kb_per_core\": {\n");
    print_stat_obj(out, "      ", "read_only", l1_rd_1c, l1_rd_mc, mc_threads,
                   num_mc, 1);
    print_stat_obj(out, "      ", "read_write_copy", l1_cp_1c, l1_cp_mc,
                   mc_threads, num_mc, 0);
    fprintf(out, "    },\n");
    fprintf(out, "    \"l2_cluster\": {\n");
    print_stat_obj(out, "      ", "read_only", l2_rd_1c, l2_rd_mc, mc_threads,
                   num_mc, 1);
    print_stat_obj(out, "      ", "read_write_copy", l2_cp_1c, l2_cp_mc,
                   mc_threads, num_mc, 0);
    fprintf(out, "    },\n");
    fprintf(out, "    \"slc_shared\": {\n");
    print_stat_obj(out, "      ", "read_only", slc_rd_1c, slc_rd_mc, mc_threads,
                   num_mc, 1);
    print_stat_obj(out, "      ", "read_write_copy", slc_cp_1c, slc_cp_mc,
                   mc_threads, num_mc, 0);
    fprintf(out, "    },\n");
    fprintf(out, "    \"unified_memory\": {\n");
    print_stat_obj(out, "      ", "read_only", mem_rd_1c, mem_rd_mc, mc_threads,
                   num_mc, 1);
    print_stat_obj(out, "      ", "read_write_copy", mem_cp_1c, mem_cp_mc,
                   mc_threads, num_mc, 1);
    print_stat_obj(out, "      ", "stream_triad", mem_tr_1c, mem_tr_mc,
                   mc_threads, num_mc, 0);
    fprintf(out, "    }\n");
    fprintf(out, "  }\n");
    fprintf(out, "}\n");
  }
  if (outs[1]) fclose(outs[1]);
  return 0;
}
