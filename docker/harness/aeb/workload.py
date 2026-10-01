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
from importlib import resources

# Gemma 4 chat template, single user turn, thinking disabled (no <|think|> in
# a system turn). <bos> is added by the framework, never by the harness.
USER_PREFIX = "<|turn>user\n"
TURN_SUFFIX = "<turn|>\n<|turn>model\n"
# Stop tokens from the .litertlm LlmMetadata (<eos>, <|tool_response>, <turn|>);
# llama.cpp is given the same set explicitly.
STOP_IDS = [1, 50, 106]

PERF_INSTRUCTION = ("Read the following passage, then write a detailed continuation "
                    "of the story in the same style.\n\n")


def chat(user_text: str) -> str:
  return f"{USER_PREFIX}{user_text}{TURN_SUFFIX}"


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

  lo, hi = 0, len(text)
  if len(tokenize(render(hi))) < prompt_tokens:
    raise ValueError(f"passage too short for {prompt_tokens} prompt tokens")
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
