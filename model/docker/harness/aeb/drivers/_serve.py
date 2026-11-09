"""JSON-lines serve loop shared by the Python drivers added for the Apple GPU
suite (see docs/harness.md, "Driver protocol")."""

from __future__ import annotations

import json
import sys


def emit(obj) -> None:
  sys.stdout.write(json.dumps(obj) + "\n")
  sys.stdout.flush()


def serve(driver, ready: dict) -> None:
  """`driver` implements tokenize(text, add_bos) -> ids and generate(req) -> dict."""
  emit(ready)
  for line in sys.stdin:
    line = line.strip()
    if not line:
      continue
    try:
      req = json.loads(line)
    except json.JSONDecodeError as e:
      emit({"ok": False, "error": f"bad json: {e}"})
      continue
    rid, op = req.get("id"), req.get("op")
    try:
      if op == "quit":
        emit({"id": rid, "ok": True})
        break
      if op == "tokenize":
        emit({"id": rid, "ok": True, "ids": driver.tokenize(req["text"], req.get("add_bos", True))})
      elif op == "generate":
        out = driver.generate(req)
        out["id"] = rid
        emit(out)
      else:
        emit({"id": rid, "ok": False, "error": f"unknown op: {op}"})
    except Exception as e:  # report and keep serving
      emit({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})


def run(make_driver, ready_info) -> int:
  """Construct the driver (load the model) and serve; report load failures
  on the protocol instead of a bare traceback."""
  try:
    driver = make_driver()
  except Exception as e:
    emit({"event": "error", "error": f"{type(e).__name__}: {e}"})
    return 1
  serve(driver, ready_info(driver))
  return 0
