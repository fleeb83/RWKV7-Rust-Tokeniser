"""Run one validated tokeniser implementation for an explicit wall window."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def child_result(command: list[str], output: Path) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"child failed with exit {completed.returncode}: {' '.join(command)}\n"
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"child did not produce valid JSON at {output}: {error}") from error
    if result.get("pass") is not True:
        raise RuntimeError(f"child reported failure: {result!r}")
    return result


def observed(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("observed")
    if isinstance(value, dict):
        return value
    return {"token_ids": row.get("token_ids"), "decoded_bytes": row.get("decoded_bytes")}


def timing_samples(row: dict[str, Any]) -> tuple[list[int], list[int]]:
    timing = row.get("timings_ns")
    if isinstance(timing, dict):
        encode = timing.get("encode")
        decode = timing.get("decode")
    else:
        encode = row.get("encode_nanos")
        decode = row.get("decode_nanos")
    if not isinstance(encode, list) or not isinstance(decode, list) or not encode or not decode:
        raise RuntimeError(f"case_id {row.get('case_id')!r} has missing timing samples")
    if len(encode) != len(decode):
        raise RuntimeError(f"case_id {row.get('case_id')!r} has unequal encode/decode sample counts")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in encode + decode):
        raise RuntimeError(f"case_id {row.get('case_id')!r} has invalid timing sample")
    return list(encode), list(decode)


def merge_cycle_rows(
    rows: list[dict[str, Any]],
    final_rows: dict[str, dict[str, Any]],
    encode_samples: list[int],
    decode_samples: list[int],
    cycle: int,
) -> None:
    if not rows:
        raise RuntimeError(f"cycle {cycle} returned no cases")
    seen: set[str] = set()
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"cycle {cycle} has invalid case_id: {case_id!r}")
        if case_id in seen:
            raise RuntimeError(f"cycle {cycle} duplicates case_id {case_id!r}")
        seen.add(case_id)
        current = observed(row)
        if current["token_ids"] is None or current["decoded_bytes"] is None:
            raise RuntimeError(f"case_id {case_id!r} has no retained IDs or decoded bytes")
        encode, decode = timing_samples(row)
        prior = final_rows.get(case_id)
        if prior is not None and observed(prior) != current:
            raise RuntimeError(f"case_id {case_id!r} changed IDs or bytes at cycle {cycle}")
        accumulated = prior.get("timings_ns", {"encode": [], "decode": []}) if prior else {"encode": [], "decode": []}
        if not isinstance(accumulated, dict):
            raise RuntimeError(f"case_id {case_id!r} has malformed retained timings")
        prior_encode = accumulated.get("encode", [])
        prior_decode = accumulated.get("decode", [])
        if not isinstance(prior_encode, list) or not isinstance(prior_decode, list):
            raise RuntimeError(f"case_id {case_id!r} has malformed retained timing samples")
        retained = dict(row)
        retained["timings_ns"] = {"encode": list(prior_encode) + encode, "decode": list(prior_decode) + decode}
        final_rows[case_id] = retained
        encode_samples.extend(encode)
        decode_samples.extend(decode)


def collect_window(
    duration_seconds: float,
    warmups: int,
    run_child: Any,
    clock: Any = time.perf_counter,
) -> tuple[dict[str, dict[str, Any]], list[int], list[int], int, int, float]:
    for warmup in range(1, warmups + 1):
        run_child(f"warmup-{warmup}")
    start = clock()
    deadline = start + duration_seconds
    cycles = 0
    completed_cases = 0
    final_rows: dict[str, dict[str, Any]] = {}
    encode_samples: list[int] = []
    decode_samples: list[int] = []
    while clock() < deadline or cycles == 0:
        result = run_child(f"cycle-{cycles}")
        rows = result.get("cases")
        if not isinstance(rows, list):
            raise RuntimeError(f"cycle {cycles + 1} returned malformed cases")
        cycles += 1
        completed_cases += len(rows)
        merge_cycle_rows(rows, final_rows, encode_samples, decode_samples, cycles)
    elapsed = clock() - start
    if elapsed < duration_seconds:
        raise RuntimeError(f"measured wall window ended early: {elapsed:.9f} < {duration_seconds}")
    if not encode_samples or not decode_samples:
        raise RuntimeError("measured window retained no encode or decode samples")
    return final_rows, encode_samples, decode_samples, cycles, completed_cases, elapsed


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not math.isfinite(args.duration_seconds) or args.duration_seconds < 300:
        raise ValueError("--duration-seconds must be at least 300")
    if args.warmups < 1:
        raise ValueError("--warmups must be positive")

    root = Path(__file__).resolve().parents[1]
    vocab = required_file(args.vocab, "vocabulary")
    fixture = required_file(args.fixture, "fixture")
    reference_source = required_file(args.reference_source, "reference source")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output: {output}")
    if output in {vocab, fixture, reference_source}:
        raise ValueError(f"output aliases an input: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source = required_file(root / "src" / "lib.rs", "Rust source")
    tool = required_file(Path(__file__).resolve().parent / "bench_tokenizer_reference.py", "Python benchmark driver")
    generator = required_file(Path(__file__).resolve().parent / "tokenizer_reference.py", "fixture generator")
    corpus = required_file(Path(__file__).resolve().parent / "corpus-extended.jsonl", "benchmark corpus")
    duration_wrapper = required_file(Path(__file__).resolve(), "duration wrapper")
    runner: Path | None = None
    if args.implementation == "rust":
        if args.rust_runner is None:
            raise ValueError("--rust-runner is required for --implementation rust")
        runner = required_file(args.rust_runner, "Rust fixture runner")

    inputs = {
        "vocab": {"path": str(vocab), "sha256": sha256_file(vocab), "size_bytes": vocab.stat().st_size},
        "fixture": {"path": str(fixture), "sha256": sha256_file(fixture), "size_bytes": fixture.stat().st_size},
        "reference_source": {"path": str(reference_source), "sha256": sha256_file(reference_source), "size_bytes": reference_source.stat().st_size},
        "rust_source": {"path": "embedded:src/lib.rs", "sha256": sha256_file(source), "size_bytes": source.stat().st_size},
        "tool": {"path": str(tool), "sha256": sha256_file(tool), "size_bytes": tool.stat().st_size},
        "generator": {"path": str(generator), "sha256": sha256_file(generator), "size_bytes": generator.stat().st_size},
        "corpus": {"path": str(corpus), "sha256": sha256_file(corpus), "size_bytes": corpus.stat().st_size},
        "duration_wrapper": {"path": str(duration_wrapper), "sha256": sha256_file(duration_wrapper), "size_bytes": duration_wrapper.stat().st_size},
    }
    if runner is not None:
        inputs["rust_runner"] = {"path": str(runner), "sha256": sha256_file(runner), "size_bytes": runner.stat().st_size}
    for name, value in inputs.items():
        print(f"input {name} path={value['path']} sha256={value['sha256']}", flush=True)

    def command_for(temp_output: Path) -> list[str]:
        if args.implementation == "rust":
            assert runner is not None
            return [
                str(runner), "--vocab", str(vocab), "--fixture", str(fixture), "--output", str(temp_output),
                "--mode", "sequential", "--workers", "1", "--iterations", "1",
            ]
        return [
            sys.executable, str(tool), "--reference-source", str(reference_source), "--vocab", str(vocab),
            "--fixture", str(fixture), "--output", str(temp_output), "--implementation", "trie", "--iterations", "1",
        ]

    child_provenance: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="rwkv-tokeniser-duration-") as temporary:
        temporary_path = Path(temporary)
        def run_one(label: str) -> dict[str, Any]:
            nonlocal child_provenance
            output_path = temporary_path / f"{label}.json"
            result = child_result(command_for(output_path), output_path)
            if child_provenance is None:
                child_provenance = {
                    key: result[key]
                    for key in (
                        "host", "runtime", "git", "inputs", "resource", "process_usage",
                        "timing_scope", "timings", "validation",
                    )
                    if key in result
                }
            return result

        final_rows, encode_samples, decode_samples, cycles, completed_cases, elapsed = collect_window(
            args.duration_seconds, args.warmups, run_one
        )
    if elapsed < args.duration_seconds:
        raise RuntimeError(f"measured wall window ended early: {elapsed:.9f} < {args.duration_seconds}")
    manifest = {
        "schema_version": "rwkv-tokeniser-duration-v1",
        "pass": True,
        "implementation": args.implementation,
        "duration_seconds_requested": args.duration_seconds,
        "duration_seconds_measured": elapsed,
        "warmups": args.warmups,
        "completed_corpus_cycles": cycles,
        "completed_cases": completed_cases,
        "case_count": len(final_rows),
        "inputs": inputs,
        "timing": {
            "encode_sample_count": len(encode_samples),
            "decode_sample_count": len(decode_samples),
            "encode_scope": "inner tokenizer call reported by the child driver",
            "decode_scope": "inner tokeniser call reported by the child driver",
            "wall_scope": "whole wrapper window, including child startup, validation, and orchestration",
        },
        "validation": "each child validates every case's IDs and decoded bytes before retaining timing samples",
        "child_provenance": child_provenance,
        "resource_scope": "child manifest resource fields only; wrapper aggregate resources are not measured",
        "cases": list(final_rows.values()),
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run one RWKV World tokeniser for an explicit measured wall window.")
    result.add_argument("--implementation", choices=("rust", "trie"), required=True)
    result.add_argument("--vocab", type=Path, required=True)
    result.add_argument("--fixture", type=Path, required=True)
    result.add_argument("--reference-source", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--rust-runner", type=Path)
    result.add_argument("--duration-seconds", type=float, required=True)
    result.add_argument("--warmups", type=int, required=True)
    return result


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
