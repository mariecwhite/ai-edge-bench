// Copyright 2026 Google LLC
//
// Reproducible LiteRT C-API LD_PRELOAD profiler hook for benchmarking and
// extracting TFLite/XNNPACK/YNNPACK operator & microkernel profiles from
// multimodal subgraphs (`vision_encoder`, `vision_adapter`, `audio_encoder`,
// `audio_adapter`) embedded inside `.litertlm` flatbuffer bundles using
// `/opt/litert-lm/bin/litert_lm_advanced_main`.
//
// Build:
//   gcc -O2 -shared -fPIC -o /tmp/liblitert_subgraph_profiler.so \
//       kernel/scripts/litert_subgraph_profiler.c -ldl
//
// Usage (inside aeb/litert-lm:cpu container or host):
//   AEB_SUBGRAPH_FILE=/run/aeb_models/gemma-4-31B-it-vision-280.litertlm \
//   AEB_SUBGRAPH_SPEC="vision_encoder:176715776:428625920,vision_adapter:169934848:6780672" \
//   AEB_YNNPACK=false AEB_THREADS=28 AEB_WARMUP=1 AEB_RUNS=3 \
//   LD_PRELOAD=/tmp/liblitert_subgraph_profiler.so \
//   /opt/litert-lm/bin/litert_lm_advanced_main

#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef int (*CreateEnvFn)(int, void*, void**);
typedef int (*CreateModelFn)(void*, const void*, size_t, void**);
typedef int (*CreateOptionsFn)(void**);
typedef int (*SetHwAccelFn)(void*, int);
typedef int (*CreateOpaqueFn)(const char*, void*, void (*)(void*), void**);
typedef int (*AddOpaqueFn)(void*, void*);
typedef int (*CreateCmFn)(void*, void*, void*, void**);
typedef int (*GetSigFn)(void*, size_t, void**);
typedef int (*GetNumFn)(void*, size_t*);
typedef int (*GetReqFn)(void*, size_t, size_t, void**);
typedef int (*GetTensorFn)(void*, size_t, void**);
typedef int (*GetTypeFn)(void*, void*);
typedef int (*CreateBufFn)(void*, void*, void*, void**);
typedef int (*GetProfFn)(void*, void**);
typedef int (*ProfCtrlFn)(void*);
typedef int (*GetSummaryFn)(void*, void*, const char**);
typedef int (*RunCmFn)(void*, size_t, size_t, void**, size_t, void**);

static void noop_free(void* p) { (void)p; }

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

typedef struct {
  char name[128];
  size_t offset;
  size_t size;
  void* model;
  void* cm;
  void* prof;
  size_t num_in;
  size_t num_out;
  void** in_bufs;
  void** out_bufs;
  double elapsed_ms;
} SubgraphCtx;

static int custom_main(int argc, char** argv, char** envp) {
  (void)argc;
  (void)argv;
  (void)envp;

  const char* file_path = getenv("AEB_SUBGRAPH_FILE");
  const char* spec_str = getenv("AEB_SUBGRAPH_SPEC");
  if (!file_path || !spec_str) {
    return 0;
  }
  const char* ynn_env = getenv("AEB_YNNPACK");
  int use_ynn = (ynn_env && strcmp(ynn_env, "true") == 0) ? 1 : 0;
  const char* thr_env = getenv("AEB_THREADS");
  int threads = thr_env ? atoi(thr_env) : 28;
  const char* warm_env = getenv("AEB_WARMUP");
  int warmup = warm_env ? atoi(warm_env) : 1;
  const char* runs_env = getenv("AEB_RUNS");
  int runs = runs_env ? atoi(runs_env) : 1;

  CreateEnvFn LiteRtCreateEnvironment =
      (CreateEnvFn)dlsym(RTLD_DEFAULT, "LiteRtCreateEnvironment");
  CreateModelFn LiteRtCreateModelFromBuffer =
      (CreateModelFn)dlsym(RTLD_DEFAULT, "LiteRtCreateModelFromBuffer");
  CreateOptionsFn LiteRtCreateOptions =
      (CreateOptionsFn)dlsym(RTLD_DEFAULT, "LiteRtCreateOptions");
  SetHwAccelFn LiteRtSetOptionsHardwareAccelerators = (SetHwAccelFn)dlsym(
      RTLD_DEFAULT, "LiteRtSetOptionsHardwareAccelerators");
  CreateOpaqueFn LiteRtCreateOpaqueOptions =
      (CreateOpaqueFn)dlsym(RTLD_DEFAULT, "LiteRtCreateOpaqueOptions");
  AddOpaqueFn LiteRtAddOpaqueOptions =
      (AddOpaqueFn)dlsym(RTLD_DEFAULT, "LiteRtAddOpaqueOptions");
  CreateCmFn LiteRtCreateCompiledModel =
      (CreateCmFn)dlsym(RTLD_DEFAULT, "LiteRtCreateCompiledModel");
  GetSigFn LiteRtGetModelSignature =
      (GetSigFn)dlsym(RTLD_DEFAULT, "LiteRtGetModelSignature");
  GetNumFn LiteRtGetNumSignatureInputs =
      (GetNumFn)dlsym(RTLD_DEFAULT, "LiteRtGetNumSignatureInputs");
  GetNumFn LiteRtGetNumSignatureOutputs =
      (GetNumFn)dlsym(RTLD_DEFAULT, "LiteRtGetNumSignatureOutputs");
  GetReqFn LiteRtGetCompiledModelInputBufferRequirements = (GetReqFn)dlsym(
      RTLD_DEFAULT, "LiteRtGetCompiledModelInputBufferRequirements");
  GetReqFn LiteRtGetCompiledModelOutputBufferRequirements = (GetReqFn)dlsym(
      RTLD_DEFAULT, "LiteRtGetCompiledModelOutputBufferRequirements");
  GetTensorFn LiteRtGetSignatureInputTensorByIndex = (GetTensorFn)dlsym(
      RTLD_DEFAULT, "LiteRtGetSignatureInputTensorByIndex");
  GetTensorFn LiteRtGetSignatureOutputTensorByIndex = (GetTensorFn)dlsym(
      RTLD_DEFAULT, "LiteRtGetSignatureOutputTensorByIndex");
  GetTypeFn LiteRtGetRankedTensorType =
      (GetTypeFn)dlsym(RTLD_DEFAULT, "LiteRtGetRankedTensorType");
  CreateBufFn LiteRtCreateManagedTensorBufferFromRequirements =
      (CreateBufFn)dlsym(RTLD_DEFAULT,
                         "LiteRtCreateManagedTensorBufferFromRequirements");
  GetProfFn LiteRtCompiledModelGetProfiler =
      (GetProfFn)dlsym(RTLD_DEFAULT, "LiteRtCompiledModelGetProfiler");
  ProfCtrlFn LiteRtStartProfiler =
      (ProfCtrlFn)dlsym(RTLD_DEFAULT, "LiteRtStartProfiler");
  ProfCtrlFn LiteRtStopProfiler =
      (ProfCtrlFn)dlsym(RTLD_DEFAULT, "LiteRtStopProfiler");
  GetSummaryFn LiteRtGetProfileSummary =
      (GetSummaryFn)dlsym(RTLD_DEFAULT, "LiteRtGetProfileSummary");
  RunCmFn LiteRtRunCompiledModel =
      (RunCmFn)dlsym(RTLD_DEFAULT, "LiteRtRunCompiledModel");

  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "Failed to open %s\n", file_path);
    _exit(1);
  }
  struct stat st;
  fstat(fd, &st);
  uint8_t* base =
      (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED) {
    fprintf(stderr, "mmap failed\n");
    _exit(1);
  }

  void* env = NULL;
  LiteRtCreateEnvironment(0, NULL, &env);

  SubgraphCtx subgraphs[8];
  int num_sg = 0;

  char spec_copy[1024];
  strncpy(spec_copy, spec_str, sizeof(spec_copy) - 1);
  spec_copy[sizeof(spec_copy) - 1] = '\0';

  char* saveptr = NULL;
  for (char* tok = strtok_r(spec_copy, ",", &saveptr);
       tok != NULL && num_sg < 8; tok = strtok_r(NULL, ",", &saveptr)) {
    SubgraphCtx* sg = &subgraphs[num_sg];
    memset(sg, 0, sizeof(*sg));
    if (sscanf(tok, "%127[^:]:%zu:%zu", sg->name, &sg->offset, &sg->size) !=
        3) {
      continue;
    }
    int sg_ynn = use_ynn;
    if (use_ynn && strcmp(sg->name, "vision_encoder") == 0) {
      fprintf(stderr,
              "NOTE: vision_encoder falls back to XNNPACK in LiteRT-LM v0.19.0 "
              "due to YNNPackDelegate int32 REDUCE_MAX limitation.\n");
      sg_ynn = 0;
    }
    fprintf(stderr,
            "=== COMPILING SUBGRAPH: %s (offset=%zu, size=%zu, ynnpack=%d, "
            "threads=%d) ===\n",
            sg->name, sg->offset, sg->size, sg_ynn, threads);

    int rc = LiteRtCreateModelFromBuffer(env, base + sg->offset, sg->size,
                                         &sg->model);
    if (rc != 0) {
      fprintf(stderr, "LiteRtCreateModelFromBuffer failed: %d\n", rc);
      _exit(2);
    }

    void* sig = NULL;
    LiteRtGetModelSignature(sg->model, 0, &sig);
    LiteRtGetNumSignatureInputs(sig, &sg->num_in);
    LiteRtGetNumSignatureOutputs(sig, &sg->num_out);

    int32_t in_types[16][64];
    int32_t out_types[16][64];
    for (size_t i = 0; i < sg->num_in; i++) {
      void* t = NULL;
      LiteRtGetSignatureInputTensorByIndex(sig, i, &t);
      memset(in_types[i], 0, sizeof(in_types[i]));
      LiteRtGetRankedTensorType(t, in_types[i]);
    }
    for (size_t i = 0; i < sg->num_out; i++) {
      void* t = NULL;
      LiteRtGetSignatureOutputTensorByIndex(sig, i, &t);
      memset(out_types[i], 0, sizeof(out_types[i]));
      LiteRtGetRankedTensorType(t, out_types[i]);
    }

    void* opts = NULL;
    LiteRtCreateOptions(&opts);
    LiteRtSetOptionsHardwareAccelerators(opts, 1);

    char* toml_cpu = (char*)malloc(256);
    snprintf(toml_cpu, 256,
             "num_threads = %d\nflags = 75\nenable_ynnpack = %s\n", threads,
             sg_ynn ? "true" : "false");
    void* op_cpu = NULL;
    LiteRtCreateOpaqueOptions("cpu_options", toml_cpu, noop_free, &op_cpu);
    LiteRtAddOpaqueOptions(opts, op_cpu);

    char* toml_rt = strdup("enable_profiling = true\n");
    void* op_rt = NULL;
    LiteRtCreateOpaqueOptions("runtime_options_string", toml_rt, noop_free,
                              &op_rt);
    LiteRtAddOpaqueOptions(opts, op_rt);

    rc = LiteRtCreateCompiledModel(env, sg->model, opts, &sg->cm);
    if (rc != 0) {
      fprintf(stderr, "LiteRtCreateCompiledModel failed: %d\n", rc);
      _exit(3);
    }

    sg->in_bufs = (void**)calloc(sg->num_in, sizeof(void*));
    for (size_t i = 0; i < sg->num_in; i++) {
      void* req = NULL;
      LiteRtGetCompiledModelInputBufferRequirements(sg->cm, 0, i, &req);
      LiteRtCreateManagedTensorBufferFromRequirements(env, in_types[i], req,
                                                      &sg->in_bufs[i]);
    }

    sg->out_bufs = (void**)calloc(sg->num_out, sizeof(void*));
    for (size_t i = 0; i < sg->num_out; i++) {
      void* req = NULL;
      LiteRtGetCompiledModelOutputBufferRequirements(sg->cm, 0, i, &req);
      LiteRtCreateManagedTensorBufferFromRequirements(env, out_types[i], req,
                                                      &sg->out_bufs[i]);
    }

    for (int w = 0; w < warmup; w++) {
      LiteRtRunCompiledModel(sg->cm, 0, sg->num_in, sg->in_bufs, sg->num_out,
                             sg->out_bufs);
    }

    LiteRtCompiledModelGetProfiler(sg->cm, &sg->prof);
    num_sg++;
  }

  for (int i = 0; i < num_sg; i++) {
    LiteRtStartProfiler(subgraphs[i].prof);
  }

  for (int r = 0; r < runs; r++) {
    for (int i = 0; i < num_sg; i++) {
      SubgraphCtx* sg = &subgraphs[i];
      double t0 = now_ms();
      LiteRtRunCompiledModel(sg->cm, 0, sg->num_in, sg->in_bufs, sg->num_out,
                             sg->out_bufs);
      double t1 = now_ms();
      sg->elapsed_ms += (t1 - t0);
    }
  }

  double total_ms = 0.0;
  for (int i = 0; i < num_sg; i++) {
    SubgraphCtx* sg = &subgraphs[i];
    LiteRtStopProfiler(sg->prof);
    double avg_ms = sg->elapsed_ms / (double)runs;
    total_ms += avg_ms;
    fprintf(stderr, "SUBGRAPH_LATENCY_MS[%s]: %.3f\n", sg->name, avg_ms);

    const char* summary = NULL;
    LiteRtGetProfileSummary(sg->prof, sg->cm, &summary);
    if (summary) {
      fprintf(stdout, "%s\n", summary);
      fflush(stdout);
    }
  }

  fprintf(stderr, "TOTAL_MULTIMODAL_LATENCY_MS: %.3f\n", total_ms);
  fprintf(stderr, "Prefill Speed: %.6f tokens/sec\n",
          1000.0 / (total_ms > 0.0 ? total_ms : 1.0));
  fprintf(stderr, "Decode Speed: 0.000000 tokens/sec\n");
  fflush(stderr);
  _exit(0);
}

typedef int (*libc_start_main_fn)(int (*)(int, char**, char**), int, char**,
                                  void (*)(void), void (*)(void),
                                  void (*)(void), void*);

int __libc_start_main(int (*main_fn)(int, char**, char**), int argc,
                      char** ubp_av, void (*init)(void), void (*fini)(void),
                      void (*rtld_fini)(void), void* stack_end) {
  (void)main_fn;
  libc_start_main_fn orig =
      (libc_start_main_fn)dlsym(RTLD_NEXT, "__libc_start_main");
  return orig(custom_main, argc, ubp_av, init, fini, rtld_fini, stack_end);
}
