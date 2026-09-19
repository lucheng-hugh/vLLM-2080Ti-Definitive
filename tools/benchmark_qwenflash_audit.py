#!/usr/bin/env python3
"""Measure streaming prompt/decode throughput for a vLLM completion endpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8046/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-chars", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "prompt": "x" * args.prompt_chars,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token = None
    token_events = 0
    usage = None
    with requests.post(args.url, json=payload, stream=True, timeout=1800) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or line == b"data: [DONE]" or not line.startswith(b"data: "):
                continue
            event_time = time.perf_counter()
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            if event.get("choices"):
                token_events += len(event["choices"])
                first_token = first_token or event_time
    finished = time.perf_counter()
    if first_token is None or usage is None:
        raise RuntimeError("stream did not contain token and usage events")
    prefill_s = first_token - started
    decode_s = finished - first_token
    result = {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_s": finished - started,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tok_s": usage["prompt_tokens"] / prefill_s,
        "decode_tok_s": usage["completion_tokens"] / decode_s,
        "token_events": token_events,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
