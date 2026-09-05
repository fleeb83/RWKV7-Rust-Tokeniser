"""Benchmark the pinned RWKV tokenizer references with validation gates."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from tokenizer_reference import HarnessError, _git_revision, load_reference_classes, sha256_file


class BenchmarkError(Exception):
    """An expected benchmark input, validation, or measurement failure."""


IMPLEMENTATION_CLASSES = {
    "naive": "RWKV_TOKENIZER",
    "trie": "TRIE_TOKENIZER",
}


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BenchmarkError("%s does not exist or is not a regular file: %s" % (label, resolved))
    return resolved


def _reject_aliases(paths: Mapping[str, Path]) -> None:
    seen: Dict[Path, str] = {}
    for label, path in paths.items():
        previous = seen.get(path)
        if previous is not None:
            raise BenchmarkError("%s aliases %s: %s" % (label, previous, path))
        seen[path] = label


def _load_fixture(path: Path, implementation: str) -> List[Dict[str, Any]]:
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise BenchmarkError("fixture is not valid UTF-8: %s" % exc) from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError("fixture is not valid JSON: %s" % exc) from exc
    except OSError as exc:
        raise BenchmarkError("cannot read fixture %s: %s" % (path, exc)) from exc

    if not isinstance(fixture, dict) or not isinstance(fixture.get("cases"), list):
        raise BenchmarkError("fixture must contain a cases array")
    cases = fixture["cases"]
    if not cases:
        raise BenchmarkError("fixture cases array is empty")
    expected_class = IMPLEMENTATION_CLASSES[implementation]
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise BenchmarkError("fixture case %d is not an object" % index)
        case_id = case.get("case_id")
        text = case.get("text")
        if not isinstance(case_id, str) or not case_id:
            raise BenchmarkError("fixture case %d has invalid case_id %r" % (index, case_id))
        if case_id in seen:
            raise BenchmarkError("fixture case %d duplicates case_id %r" % (index, case_id))
        if not isinstance(text, str):
            raise BenchmarkError("fixture case_id %r has non-string text" % case_id)
        references = case.get("references")
        if not isinstance(references, dict) or not isinstance(references.get(expected_class), dict):
            raise BenchmarkError("fixture case_id %r lacks %s reference" % (case_id, expected_class))
        reference = references[expected_class]
        token_ids = reference.get("token_ids")
        decoded_bytes = reference.get("decoded_bytes")
        decoded_text = reference.get("decoded_text")
        if not isinstance(token_ids, list) or not all(
            isinstance(token, int) and not isinstance(token, bool) for token in token_ids
        ):
            raise BenchmarkError("fixture case_id %r has invalid token_ids" % case_id)
        if not isinstance(decoded_bytes, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 255
            for value in decoded_bytes
        ):
            raise BenchmarkError("fixture case_id %r has invalid decoded_bytes" % case_id)
        if not isinstance(decoded_text, str):
            raise BenchmarkError("fixture case_id %r has invalid decoded_text" % case_id)
        expected_hex = reference.get("decoded_bytes_hex")
        actual_hex = bytes(decoded_bytes).hex()
        if expected_hex != actual_hex:
            raise BenchmarkError(
                "fixture case_id %r decoded_bytes_hex mismatch: %r != %r"
                % (case_id, expected_hex, actual_hex)
            )
        input_row = case.get("input_row")
        if not isinstance(input_row, dict):
            input_row = {
                key: value
                for key, value in case.items()
                if key not in ("references", "input_row")
            }
        rows.append(
            {
                "case_id": case_id,
                "input_row": input_row,
                "text": text,
                "expected_token_ids": token_ids,
                "expected_decoded_bytes": decoded_bytes,
                "expected_decoded_text": decoded_text,
            }
        )
        seen.add(case_id)
    declared_count = fixture.get("case_count")
    if declared_count != len(rows):
        raise BenchmarkError(
            "fixture case_count mismatch: declared %r, actual %d" % (declared_count, len(rows))
        )
    return rows


def _token_ids(value: Any, case_id: str, operation: str) -> List[int]:
    try:
        values = list(value)
    except Exception as exc:
        raise BenchmarkError("case_id %r %s did not return an iterable: %s" % (case_id, operation, exc)) from exc
    if any(isinstance(token, bool) or not isinstance(token, int) for token in values):
        raise BenchmarkError("case_id %r %s returned a non-integer token" % (case_id, operation))
    return values


def _validate_iteration(
    case: Mapping[str, Any],
    encoded: Sequence[int],
    decoded_text: Any,
    decoded_bytes: Any,
    iteration: int,
) -> None:
    case_id = case["case_id"]
    expected_tokens = case["expected_token_ids"]
    if list(encoded) != expected_tokens:
        raise BenchmarkError(
            "case_id %r iteration %d token ID mismatch: %r != %r"
            % (case_id, iteration, list(encoded), expected_tokens)
        )
    expected_bytes = bytes(case["expected_decoded_bytes"])
    if not isinstance(decoded_bytes, bytes) or decoded_bytes != expected_bytes:
        actual = decoded_bytes.hex() if isinstance(decoded_bytes, bytes) else repr(decoded_bytes)
        raise BenchmarkError(
            "case_id %r iteration %d decoded bytes mismatch: %s != %s"
            % (case_id, iteration, actual, expected_bytes.hex())
        )
    if decoded_text != case["expected_decoded_text"]:
        raise BenchmarkError(
            "case_id %r iteration %d decoded text mismatch: %r != %r"
            % (case_id, iteration, decoded_text, case["expected_decoded_text"])
        )


def _process_usage() -> Dict[str, Any]:
    usage: Dict[str, Any] = {
        "measured": False,
        "scope": "whole benchmark process, including reference loading and fixture validation; not tokenizer allocation exclusively",
        "platform": platform.system(),
    }
    try:
        import psutil  # type: ignore
    except ImportError:
        usage["psutil_available"] = False
    else:
        usage["psutil_available"] = True
        try:
            process = psutil.Process(os.getpid())
            cpu_times = process.cpu_times()
            memory_info = process.memory_info()
            usage["cpu_times_seconds"] = {
                field: getattr(cpu_times, field)
                for field in ("user", "system")
                if hasattr(cpu_times, field)
            }
            usage["memory_info_bytes"] = {
                field: getattr(memory_info, field)
                for field in ("rss", "vms", "peak_wset")
                if hasattr(memory_info, field)
            }
            usage["measured"] = bool(usage["cpu_times_seconds"] and usage["memory_info_bytes"])
        except (OSError, psutil.Error) as exc:
            usage["error"] = "psutil measurement failed: %s" % exc

    if sys.platform.startswith("linux"):
        try:
            import resource

            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except (ImportError, OSError):
            pass
        else:
            usage["resource_maxrss_bytes"] = int(maxrss) * 1024
            usage["peak_memory_source"] = "resource.getrusage(RUSAGE_SELF).ru_maxrss"
            usage["measured"] = True
    elif os.name == "nt" and "peak_wset" in usage.get("memory_info_bytes", {}):
        usage["peak_memory_source"] = "psutil.Process.memory_info().peak_wset"
    return usage


def run_benchmark(
    reference_source: Path,
    vocab: Path,
    fixture: Path,
    output: Path,
    iterations: int,
    implementation: str,
) -> Dict[str, Any]:
    """Run validated encode/decode timings and write one benchmark manifest."""

    if implementation not in IMPLEMENTATION_CLASSES:
        raise BenchmarkError("implementation must be naive or trie: %r" % implementation)
    if iterations <= 0:
        raise BenchmarkError("--iterations must be positive: %d" % iterations)
    reference_source = _require_file(reference_source, "reference source")
    vocab = _require_file(vocab, "vocabulary")
    fixture = _require_file(fixture, "fixture")
    output = output.expanduser().resolve()
    _reject_aliases(
        {
            "reference source": reference_source,
            "vocabulary": vocab,
            "fixture": fixture,
            "output": output,
        }
    )
    tool = Path(__file__).resolve()
    source_hash, source_size = sha256_file(reference_source)
    vocab_hash, vocab_size = sha256_file(vocab)
    fixture_hash, fixture_size = sha256_file(fixture)
    tool_hash, tool_size = sha256_file(tool)
    for label, path, digest in (
        ("reference_source", reference_source, source_hash),
        ("vocab", vocab, vocab_hash),
        ("fixture", fixture, fixture_hash),
        ("tool", tool, tool_hash),
    ):
        print("input %s path=%s sha256=%s" % (label, path, digest), flush=True)

    cases = _load_fixture(fixture, implementation)
    try:
        classes = load_reference_classes(reference_source, vocab)
        tokenizer = classes[IMPLEMENTATION_CLASSES[implementation]](str(vocab))
    except (HarnessError, BenchmarkError):
        raise
    except Exception as exc:
        raise BenchmarkError("failed to load %s tokenizer reference: %s: %s" % (implementation, type(exc).__name__, exc)) from exc

    benchmark_cases: List[Dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        source_bytes = case["text"].encode("utf-8")
        encode_timings: List[int] = []
        decode_timings: List[int] = []
        observed_tokens: List[int] | None = None
        observed_bytes: bytes | None = None
        for iteration in range(1, iterations + 1):
            try:
                encode_start = time.perf_counter_ns()
                encoded_raw = tokenizer.encodeBytes(source_bytes)
                encode_elapsed = time.perf_counter_ns() - encode_start
            except BenchmarkError:
                raise
            except Exception as exc:
                raise BenchmarkError(
                    "case_id %r iteration %d encode failed: %s: %s"
                    % (case_id, iteration, type(exc).__name__, exc)
                ) from exc
            encoded = _token_ids(encoded_raw, case_id, "encodeBytes")

            try:
                decode_start = time.perf_counter_ns()
                decoded_bytes = tokenizer.decodeBytes(encoded)
                decode_elapsed = time.perf_counter_ns() - decode_start
            except Exception as exc:
                raise BenchmarkError(
                    "case_id %r iteration %d decode failed: %s: %s"
                    % (case_id, iteration, type(exc).__name__, exc)
                ) from exc
            try:
                decoded_text = tokenizer.decode(encoded)
            except Exception as exc:
                raise BenchmarkError(
                    "case_id %r iteration %d decode text validation failed: %s: %s"
                    % (case_id, iteration, type(exc).__name__, exc)
                ) from exc

            # Validation is deliberately complete before either duration enters
            # the retained timing arrays. A failed iteration contributes no time.
            _validate_iteration(case, encoded, decoded_text, decoded_bytes, iteration)
            encode_timings.append(encode_elapsed)
            decode_timings.append(decode_elapsed)
            observed_tokens = list(encoded)
            observed_bytes = decoded_bytes

        if observed_tokens is None or observed_bytes is None:
            raise BenchmarkError("case_id %r produced no validated iterations" % case_id)
        benchmark_cases.append(
            {
                "case_id": case_id,
                "input_row": case["input_row"],
                "observed": {
                    "token_ids": observed_tokens,
                    "decoded_bytes": list(observed_bytes),
                    "decoded_bytes_hex": observed_bytes.hex(),
                    "decoded_text": case["expected_decoded_text"],
                },
                "expected": {
                    "token_ids": case["expected_token_ids"],
                    "decoded_bytes": case["expected_decoded_bytes"],
                    "decoded_bytes_hex": bytes(case["expected_decoded_bytes"]).hex(),
                    "decoded_text": case["expected_decoded_text"],
                },
                "timings_ns": {
                    "encode": encode_timings,
                    "decode": decode_timings,
                    "iterations_validated": iterations,
                    "clock": "time.perf_counter_ns",
                    "encode_scope": "tokenizer.encodeBytes(precomputed UTF-8 input bytes) call only",
                    "decode_scope": "tokenizer.decodeBytes(validated token IDs) call only",
                },
            }
        )

    try:
        git = _git_revision(Path(__file__).resolve().parents[1])
    except HarnessError as error:
        git = {"available": False, "error": str(error)}
    manifest = {
        "schema_version": "b16a-tokenizer-reference-benchmark-v1",
        "pass": True,
        "implementation": implementation,
        "implementation_class": IMPLEMENTATION_CLASSES[implementation],
        "iterations": iterations,
        "validation": {
            "all_iterations_compared_before_timing_count": True,
            "validation_outside_timed_sections": True,
            "top_level_reference_code_executed": False,
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "encode_scope": "tokenizer.encodeBytes(precomputed UTF-8 input bytes) call only",
            "decode_scope": "tokenizer.decodeBytes(validated token IDs) call only",
            "validation_scope": "token conversion, expected ID/byte/text comparison, and fixture validation are outside timed sections",
        },
        "inputs": {
            "reference_source": {"path": str(reference_source), "sha256": source_hash, "size_bytes": source_size},
            "vocab": {"path": str(vocab), "sha256": vocab_hash, "size_bytes": vocab_size},
            "fixture": {"path": str(fixture), "sha256": fixture_hash, "size_bytes": fixture_size},
            "tool": {"path": str(tool), "sha256": tool_hash, "size_bytes": tool_size},
        },
        "runtime": {
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "sys_platform": sys.platform,
            "git": git,
        },
        "process_usage": _process_usage(),
        "case_count": len(benchmark_cases),
        "cases": benchmark_cases,
    }
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise BenchmarkError("cannot write benchmark output %s: %s" % (output, exc)) from exc
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a validated RWKV tokenizer reference.")
    parser.add_argument("--reference-source", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--implementation", required=True, choices=tuple(IMPLEMENTATION_CLASSES))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        manifest = run_benchmark(
            args.reference_source,
            args.vocab,
            args.fixture,
            args.output,
            args.iterations,
            args.implementation,
        )
    except BenchmarkError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    print(
        "wrote %s implementation=%s cases=%d iterations=%d"
        % (args.output, manifest["implementation"], manifest["case_count"], manifest["iterations"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
