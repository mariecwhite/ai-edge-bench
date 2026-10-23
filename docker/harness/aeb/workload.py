"""Deterministic workload construction.

The performance workload is a real-text prompt (public-domain passage, see
data/alice_ch1-3.txt) rendered with the Gemma 4 chat template and trimmed
character-by-character until the framework's own tokenizer reports exactly
`prompt_tokens` tokens including <bos>. Both frameworks use the same Gemma 4
SentencePiece vocabulary; the harness records the SHA-256 of the token ids
each framework produced so a report can prove the inputs were identical.

Real text is used instead of random token ids (llama-bench) or zero padding
(LiteRT-LM's --benchmark_prefill_tokens) because attention/activation value
ranges, and therefore some kernels' fast paths, depend on the content.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib import resources

# Gemma 4 chat template, single user turn, thinking disabled (no <|think|> in
# a system turn). <bos> is added by the framework, never by the harness.
USER_PREFIX = "<|turn>user\n"
TURN_SUFFIX = "<turn|>\n<|turn>model\n"
# Generation-prompt variants of the Gemma 4 chat templates with thinking
# disabled. The E2B template ends at "<|turn>model\n"; the 26B A4B (and 31B)
# templates also emit an empty thought channel. Selected per suite
# (workload.chat_template) through AEB_CHAT_TEMPLATE and recorded in every run.
TEMPLATES = {"gemma4": "", "gemma4-empty-thought": "<|channel>thought\n<channel|>"}
CHAT_TEMPLATE = os.environ.get("AEB_CHAT_TEMPLATE", "gemma4")
if CHAT_TEMPLATE not in TEMPLATES:
  raise ValueError(f"AEB_CHAT_TEMPLATE={CHAT_TEMPLATE!r}; expected one of {sorted(TEMPLATES)}")
# Stop tokens from the .litertlm LlmMetadata (<eos>, <|tool_response>, <turn|>);
# llama.cpp is given the same set explicitly.
STOP_IDS = [1, 50, 106]

PERF_INSTRUCTION = ("Read the following passage, then write a detailed continuation "
                    "of the story in the same style.\n\n")


def chat(user_text: str) -> str:
  return f"{USER_PREFIX}{user_text}{TURN_SUFFIX}{TEMPLATES[CHAT_TEMPLATE]}"


def passage() -> str:
  return resources.files("aeb.data").joinpath("alice_ch1-3.txt").read_text()


def ids_sha256(ids: list[int]) -> str:
  return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def build_perf_prompt(tokenize, prompt_tokens: int) -> tuple[str, list[int]]:
  """Return (prompt_text_without_bos, token_ids_with_bos) of exactly
  prompt_tokens tokens. `tokenize(text) -> ids` must add <bos>."""
  text = passage()

  def render(n_chars: int) -> str:
    return chat(PERF_INSTRUCTION + text[:n_chars].rstrip())

  # Repeat only for lengths beyond the corpus; existing prompts stay unchanged.
  while len(tokenize(render(len(text)))) < prompt_tokens:
    text = text + "\n\n" + text
  lo, hi = 0, len(text)
  while lo < hi:  # smallest prefix with >= prompt_tokens tokens
    mid = (lo + hi) // 2
    if len(tokenize(render(mid))) >= prompt_tokens:
      hi = mid
    else:
      lo = mid + 1
  for n in range(lo, max(lo - 64, 0), -1):
    ids = tokenize(render(n))
    if len(ids) == prompt_tokens:
      return render(n), ids
  raise ValueError(f"could not hit exactly {prompt_tokens} tokens")


def build_prefix_cache_prompt(
    tokenize,
    prompt_tokens: int,
    prefix_tokens: int,
    variant: int = 0,
) -> tuple[str, str, str, list[int], list[int]]:
  """Build a prompt of `prompt_tokens` tokens split into a shared prefix of
  `prefix_tokens` tokens (including <bos>) and a request-specific suffix of
  `prompt_tokens - prefix_tokens` tokens ending with `TURN_SUFFIX`.

  Returns `(full_prompt, prefix_text, suffix_text, full_ids, prefix_ids)`.
  Varying `variant` (0, 1, 2, ...) changes the suffix text while keeping
  `prefix_text` and `prefix_ids` identical across requests.
  """
  if prefix_tokens < 8 or prefix_tokens >= prompt_tokens:
    raise ValueError(
        f"prefix_tokens ({prefix_tokens}) must be in [8, prompt_tokens ({prompt_tokens}) - 1]"
    )
  base_text = passage()
  while len(tokenize(f"{USER_PREFIX}{PERF_INSTRUCTION}{base_text}")) < prompt_tokens * 2:
    base_text = base_text + "\n\n" + base_text

  def render_prefix(n_chars: int) -> str:
    return f"{USER_PREFIX}{PERF_INSTRUCTION}{base_text[:n_chars]}"

  lo, hi = 0, len(base_text)
  while lo < hi:
    mid = (lo + hi) // 2
    if len(tokenize(render_prefix(mid))) >= prefix_tokens:
      hi = mid
    else:
      lo = mid + 1
  prefix_text = None
  prefix_ids = None
  prefix_char_end = lo
  for n in range(min(lo + 32, len(base_text)), max(lo - 128, 0), -1):
    cand = render_prefix(n)
    ids = tokenize(cand)
    if len(ids) == prefix_tokens:
      # Verify that appending text after `cand` does not alter `ids` via BPE merge.
      probe_ids = tokenize(cand + " The")
      if probe_ids[:prefix_tokens] == ids:
        prefix_text = cand
        prefix_ids = ids
        prefix_char_end = n
        break
  if prefix_text is None or prefix_ids is None:
    raise ValueError(f"could not hit exact prefix_tokens={prefix_tokens}")

  # Select a distinct slice of the corpus for each variant so consecutive
  # requests do not share suffix tokens beyond prefix_tokens.
  rem_corpus = base_text[prefix_char_end:] + "\n\n" + base_text
  stride = max(256, (prompt_tokens - prefix_tokens) * 6)
  offset = (variant * stride) % max(1, len(base_text) // 2)
  suffix_corpus = " " + rem_corpus[offset:].lstrip()
  while len(tokenize(prefix_text + suffix_corpus + TURN_SUFFIX)) < prompt_tokens + 64:
    suffix_corpus = suffix_corpus + "\n\n" + base_text

  def render_full(n_chars: int) -> tuple[str, str]:
    suf = suffix_corpus[:n_chars] + TURN_SUFFIX
    return prefix_text + suf, suf

  lo, hi = 1, len(suffix_corpus)
  while lo < hi:
    mid = (lo + hi) // 2
    full_cand, _ = render_full(mid)
    if len(tokenize(full_cand)) >= prompt_tokens:
      hi = mid
    else:
      lo = mid + 1
  for n in range(min(lo + 32, len(suffix_corpus)), max(lo - 128, 1), -1):
    full_cand, suf_cand = render_full(n)
    ids = tokenize(full_cand)
    if len(ids) == prompt_tokens and ids[:prefix_tokens] == prefix_ids:
      return full_cand, prefix_text, suf_cand, ids, prefix_ids
  raise ValueError(
      f"could not hit exact prompt_tokens={prompt_tokens} with prefix_tokens={prefix_tokens}"
  )

