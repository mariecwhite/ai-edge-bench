// aeb-llama-driver: in-process llama.cpp driver for the AI Edge Bench harness.
//
// Speaks the harness JSON-lines protocol on stdin/stdout (see
// harness/aeb/protocol.md). The model is loaded once; every request starts
// from an empty KV cache and is decoded greedily. Timestamps are taken from
// CLOCK_MONOTONIC (std::chrono::steady_clock on Linux) so the orchestrator can
// line them up with its own resource samples.
//
// Build (inside the llama.cpp image):
//   g++ -O2 -std=c++17 aeb-llama-driver.cpp -Iinclude -Iggml/include -Ivendor \
//       -Lbuild/bin -lllama -lggml -lggml-base -o aeb-llama-driver

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace {

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void emit(const json& j) {
  std::cout << j.dump() << "\n" << std::flush;
}

ggml_type parse_cache_type(const std::string& s) {
  if (s == "f16") return GGML_TYPE_F16;
  if (s == "bf16") return GGML_TYPE_BF16;
  if (s == "f32") return GGML_TYPE_F32;
  if (s == "q8_0") return GGML_TYPE_Q8_0;
  if (s == "q4_0") return GGML_TYPE_Q4_0;
  std::fprintf(stderr, "unsupported cache type: %s\n", s.c_str());
  std::exit(2);
}

struct Args {
  std::string model;
  int threads = 4;
  int threads_batch = 0;  // 0 = same as threads
  int ctx = 4096;
  int batch = 2048;
  int ubatch = 512;
  std::string flash_attn = "auto";
  std::string cache_type_k = "f16";
  std::string cache_type_v = "f16";
  bool mmap = true;
  bool repack = true;
  bool swa_full = false;  // llama-bench and llama-cli default
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", k.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--model") a.model = val();
    else if (k == "--threads") a.threads = std::stoi(val());
    else if (k == "--threads-batch") a.threads_batch = std::stoi(val());
    else if (k == "--ctx") a.ctx = std::stoi(val());
    else if (k == "--batch") a.batch = std::stoi(val());
    else if (k == "--ubatch") a.ubatch = std::stoi(val());
    else if (k == "--flash-attn") a.flash_attn = val();
    else if (k == "--cache-type-k") a.cache_type_k = val();
    else if (k == "--cache-type-v") a.cache_type_v = val();
    else if (k == "--no-mmap") a.mmap = false;
    else if (k == "--no-repack") a.repack = false;
    else if (k == "--swa-full") a.swa_full = true;
    else {
      std::fprintf(stderr, "unknown argument: %s\n", k.c_str());
      std::exit(2);
    }
  }
  if (a.model.empty()) {
    std::fprintf(stderr, "--model is required\n");
    std::exit(2);
  }
  if (a.threads_batch <= 0) a.threads_batch = a.threads;
  return a;
}

std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& text, bool add_bos) {
  int n = -llama_tokenize(vocab, text.data(), (int32_t)text.size(), nullptr, 0,
                          add_bos, /*parse_special=*/true);
  std::vector<llama_token> out(n);
  int got = llama_tokenize(vocab, text.data(), (int32_t)text.size(),
                           out.data(), n, add_bos, true);
  if (got < 0) throw std::runtime_error("tokenization failed");
  out.resize(got);
  return out;
}

std::string piece(const llama_vocab* vocab, llama_token t) {
  char buf[256];
  int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0,
                               /*special=*/false);
  if (n < 0) return std::string();
  return std::string(buf, n);
}

llama_token argmax(const float* logits, int n_vocab) {
  return (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  llama_log_set([](ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
  }, nullptr);

  const int64_t t_load0 = now_ns();
  ggml_backend_load_all();
  llama_backend_init();

  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  mp.use_extra_bufts = args.repack;
  if (!args.mmap) mp.load_mode = LLAMA_LOAD_MODE_NONE;
  llama_model* model = llama_model_load_from_file(args.model.c_str(), mp);
  if (!model) {
    emit({{"event", "error"}, {"error", "failed to load model"}});
    return 1;
  }

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = args.ctx;
  cp.n_batch = args.batch;
  cp.n_ubatch = args.ubatch;
  cp.n_seq_max = 1;
  cp.n_threads = args.threads;
  cp.n_threads_batch = args.threads_batch;
  cp.flash_attn_type = args.flash_attn == "on"    ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : args.flash_attn == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                                  : LLAMA_FLASH_ATTN_TYPE_AUTO;
  cp.type_k = parse_cache_type(args.cache_type_k);
  cp.type_v = parse_cache_type(args.cache_type_v);
  cp.no_perf = false;
  cp.swa_full = args.swa_full;
  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) {
    emit({{"event", "error"}, {"error", "failed to create context"}});
    return 1;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);
  const int64_t t_load1 = now_ns();

  char desc[256];
  llama_model_desc(model, desc, sizeof(desc));
  emit({{"event", "ready"},
        {"framework", "llama.cpp"},
        {"load_ns", t_load1 - t_load0},
        {"info",
         {{"model_desc", desc},
          {"model_size_bytes", llama_model_size(model)},
          {"n_params", llama_model_n_params(model)},
          {"n_ctx", llama_n_ctx(ctx)},
          {"n_vocab", n_vocab},
          {"threads", args.threads},
          {"threads_batch", args.threads_batch},
          {"batch", args.batch},
          {"ubatch", args.ubatch},
          {"flash_attn", args.flash_attn},
          {"cache_type_k", args.cache_type_k},
          {"cache_type_v", args.cache_type_v},
          {"mmap", args.mmap},
          {"repack", args.repack},
          {"swa_full", args.swa_full},
          {"system_info", llama_print_system_info()}}}});

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    json req;
    try {
      req = json::parse(line);
    } catch (const std::exception& e) {
      emit({{"ok", false}, {"error", std::string("bad json: ") + e.what()}});
      continue;
    }
    const std::string op = req.value("op", "");
    const json id = req.value("id", json());
    try {
      if (op == "quit") {
        emit({{"id", id}, {"ok", true}});
        break;
      } else if (op == "tokenize") {
        auto ids = tokenize(vocab, req.at("text").get<std::string>(),
                            req.value("add_bos", true));
        emit({{"id", id}, {"ok", true}, {"ids", ids}});
      } else if (op == "generate") {
        const std::string prompt = req.at("prompt").get<std::string>();
        const int max_tokens = req.value("max_tokens", 256);
        const bool ignore_eos = req.value("ignore_eos", false);
        const bool return_text = req.value("return_text", true);
        // Optional seeded sampling; absent or temperature <= 0 means greedy.
        const json sampling = req.value("sampling", json::object());
        const float temperature = sampling.value("temperature", 0.0f);
        llama_sampler* smpl = nullptr;
        if (temperature > 0.0f) {
          smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
          const int top_k = sampling.value("top_k", 0);
          const float top_p = sampling.value("top_p", 1.0f);
          if (top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
          if (top_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
          llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
          llama_sampler_chain_add(smpl, llama_sampler_init_dist(sampling.value("seed", 0u)));
        }
        std::set<llama_token> stop_ids;
        for (auto& s : req.value("stop_ids", json::array())) stop_ids.insert(s.get<int>());

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_perf_context_reset(ctx);

        // Timed region starts before tokenization, matching LiteRT-LM where
        // tokenization happens inside the prefill call.
        const int64_t t_req = now_ns();
        std::vector<llama_token> prompt_ids = tokenize(vocab, prompt, true);
        if ((int)prompt_ids.size() + max_tokens > (int)llama_n_ctx(ctx)) {
          throw std::runtime_error("prompt + max_tokens exceeds context");
        }
        // Prefill in n_batch chunks; llama_decode splits into ubatches.
        for (size_t i = 0; i < prompt_ids.size(); i += args.batch) {
          int n = (int)std::min<size_t>(args.batch, prompt_ids.size() - i);
          llama_batch b = llama_batch_get_one(prompt_ids.data() + i, n);
          if (llama_decode(ctx, b) != 0) throw std::runtime_error("prefill decode failed");
        }
        const int64_t t_prefill_done = now_ns();

        std::vector<llama_token> gen;
        std::vector<int64_t> token_ns;
        std::string text;
        gen.reserve(max_tokens);
        token_ns.reserve(max_tokens);
        bool stopped = false;
        while ((int)gen.size() < max_tokens) {
          llama_token t = smpl ? llama_sampler_sample(smpl, ctx, -1)
                               : argmax(llama_get_logits_ith(ctx, -1), n_vocab);
          const int64_t ts = now_ns();
          const bool is_stop = stop_ids.count(t) > 0 || llama_vocab_is_eog(vocab, t);
          if (is_stop && !ignore_eos) {
            stopped = true;
            break;
          }
          gen.push_back(t);
          token_ns.push_back(ts);
          if ((int)gen.size() >= max_tokens) break;
          llama_batch b = llama_batch_get_one(&gen.back(), 1);
          if (llama_decode(ctx, b) != 0) throw std::runtime_error("decode failed");
        }
        const int64_t t_end = now_ns();
        if (smpl) llama_sampler_free(smpl);
        if (return_text) {
          for (auto t : gen) text += piece(vocab, t);
        }
        auto perf = llama_perf_context(ctx);
        json resp = {{"id", id},
                     {"ok", true},
                     {"n_prompt", prompt_ids.size()},
                     {"n_gen", gen.size()},
                     {"stopped_on_eos", stopped},
                     {"t_req_ns", t_req},
                     {"t_prefill_done_ns", t_prefill_done},
                     {"t_first_ns", token_ns.empty() ? 0 : token_ns.front()},
                     {"t_end_ns", t_end},
                     {"token_ns", token_ns},
                     {"gen_ids", gen},
                     {"native",
                      {{"t_p_eval_ms", perf.t_p_eval_ms},
                       {"t_eval_ms", perf.t_eval_ms},
                       {"n_p_eval", perf.n_p_eval},
                       {"n_eval", perf.n_eval}}}};
        if (req.value("return_prompt_ids", false)) resp["prompt_ids"] = prompt_ids;
        if (return_text) resp["text"] = text;
        emit(resp);
      } else {
        emit({{"id", id}, {"ok", false}, {"error", "unknown op: " + op}});
      }
    } catch (const std::exception& e) {
      emit({{"id", id}, {"ok", false}, {"error", e.what()}});
    }
  }

  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
