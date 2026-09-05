"""B16a fixture generator for the two public RWKV tokenizer references.

The public ``rwkv_tokenizer.py`` file contains executable demonstrations after
the tokenizer classes. This harness parses the complete source as an AST and
executes only the expected class definitions.

The reference source and vocabulary are trusted local inputs. AST extraction
is not a security sandbox: selected method bodies and the reference's ``eval``
can execute arbitrary code. The extraction boundary only prevents unrelated
top-level demos, benchmarks, and tests from executing.

The reference source is a required local input. This module does not download
it, search for it, or fall back to another tokenizer.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


REFERENCE_CLASS_NAMES = ("RWKV_TOKENIZER", "TRIE", "TRIE_TOKENIZER")
PUBLIC_TOKENIZER_NAMES = ("RWKV_TOKENIZER", "TRIE_TOKENIZER")


class HarnessError(Exception):
    """An expected input, extraction, reference, or fixture invariant error."""


def sha256_file(path: Path) -> Tuple[str, int]:
    """Return the SHA-256 and byte size of one file."""

    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise HarnessError("cannot read input file %r: %s" % (str(path), exc)) from exc
    return digest.hexdigest(), size


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise HarnessError("%s does not exist or is not a regular file: %s" % (label, resolved))
    return resolved


def _read_utf8(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HarnessError("%s is not valid UTF-8: %s" % (label, exc)) from exc
    except OSError as exc:
        raise HarnessError("cannot read %s %r: %s" % (label, str(path), exc)) from exc


def _class_body_is_declarative(node: ast.ClassDef) -> bool:
    """Reject executable class-body statements while retaining public classes."""

    allowed = (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.AsyncFunctionDef, ast.Pass)
    for statement in node.body:
        if isinstance(statement, ast.Expr) and isinstance(getattr(statement, "value", None), ast.Constant):
            if isinstance(statement.value.value, str):
                continue  # A class docstring is harmless.
        if not isinstance(statement, allowed):
            return False
    return True


def _extract_class_nodes(source: str) -> List[ast.ClassDef]:
    try:
        tree = ast.parse(source, filename="rwkv_tokenizer.py", mode="exec")
    except SyntaxError as exc:
        raise HarnessError("reference source has a syntax error: %s" % exc) from exc

    found: Dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in REFERENCE_CLASS_NAMES:
            if node.name in found:
                raise HarnessError("reference source defines %s more than once" % node.name)
            if node.decorator_list or node.bases or node.keywords:
                raise HarnessError("reference class %s has unsupported bases or decorators" % node.name)
            if not _class_body_is_declarative(node):
                raise HarnessError("reference class %s has executable class-body statements" % node.name)
            found[node.name] = node

    missing = [name for name in REFERENCE_CLASS_NAMES if name not in found]
    if missing:
        raise HarnessError("reference source is missing class definitions: %s" % ", ".join(missing))

    # Preserve source order: this is the closest representation of the public
    # module while still compiling no imports, demos, benchmarks, or tests.
    return [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in found]


def _controlled_builtins(vocab_path: Path) -> Dict[str, Any]:
    allowed_names = (
        "AssertionError",
        "Exception",
        "ValueError",
        "__build_class__",
        "bytes",
        "dict",
        "enumerate",
        "eval",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "next",
        "open",
        "print",
        "range",
        "repr",
        "reversed",
        "set",
        "str",
        "type",
    )
    namespace = {name: getattr(builtins, name) for name in allowed_names if name != "open"}
    expected = vocab_path.resolve()

    def controlled_open(file_name: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        try:
            requested = Path(file_name).expanduser().resolve()
        except (TypeError, OSError) as exc:
            raise HarnessError("reference attempted to open an invalid path %r: %s" % (file_name, exc)) from exc
        if requested != expected:
            raise HarnessError(
                "reference attempted to open %s; only the explicit vocabulary path %s is allowed"
                % (requested, expected)
            )
        if mode not in ("r", "rt"):
            raise HarnessError("reference attempted to open vocabulary in unsupported mode %r" % mode)
        return builtins.open(expected, mode, *args, **kwargs)

    namespace["open"] = controlled_open
    namespace["__builtins__"] = namespace
    return namespace


def load_reference_classes(source_path: Path, vocab_path: Path) -> Dict[str, type]:
    """Extract and load only the three expected public tokenizer definitions."""

    source = _read_utf8(source_path, "reference source")
    nodes = _extract_class_nodes(source)
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = _controlled_builtins(vocab_path)
    namespace["__name__"] = "b16a_tokenizer_reference_extracted"
    try:
        code = compile(module, str(source_path), "exec")
        exec(code, namespace, namespace)
    except HarnessError:
        raise
    except Exception as exc:
        raise HarnessError("failed to load extracted reference classes: %s: %s" % (type(exc).__name__, exc)) from exc

    classes: Dict[str, type] = {}
    for name in REFERENCE_CLASS_NAMES:
        value = namespace.get(name)
        if not isinstance(value, type):
            raise HarnessError("extracted %s is not a class" % name)
        classes[name] = value
    return classes


def _git_revision(repo_root: Path) -> Dict[str, Any]:
    """Read revision and dirty state with exact-root safe.directory settings."""

    safe_directory = "safe.directory=%s" % repo_root

    def run_git(arguments: Sequence[str]) -> str:
        try:
            result = subprocess.run(
                ["git", "-c", safe_directory, *arguments],
                cwd=str(repo_root),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise HarnessError(
                "git provenance command failed for %s: %s" % (repo_root, exc)
            ) from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
            raise HarnessError(
                "git provenance command failed for %s: %s"
                % (repo_root, detail)
            )
        return result.stdout

    revision = run_git(["rev-parse", "--verify", "HEAD"]).strip()
    if not revision:
        raise HarnessError("git provenance returned an empty HEAD revision")

    status = run_git(["status", "--porcelain=v1", "--untracked-files=all"])
    return {
        "revision": revision,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def _input_metadata(path: Path, digest: str, size: int) -> Dict[str, Any]:
    return {"path": str(path), "sha256": digest, "size_bytes": size}


def _load_corpus(path: Path) -> List[Dict[str, Any]]:
    text = _read_utf8(path, "corpus JSONL")
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise HarnessError("corpus JSONL line %d is blank" % line_number)
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarnessError("corpus JSONL line %d is invalid JSON: %s" % (line_number, exc)) from exc
        if not isinstance(row, dict):
            raise HarnessError("corpus JSONL line %d must be a JSON object" % line_number)
        missing = [key for key in ("case_id", "text", "seed") if key not in row]
        if missing:
            raise HarnessError("corpus JSONL line %d is missing required field(s): %s" % (line_number, ", ".join(missing)))
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise HarnessError("corpus JSONL line %d has invalid case_id %r" % (line_number, case_id))
        if case_id in seen:
            raise HarnessError("corpus JSONL line %d duplicates case_id %r" % (line_number, case_id))
        if not isinstance(row["text"], str):
            raise HarnessError("corpus JSONL line %d case_id %r has non-string text" % (line_number, case_id))
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            raise HarnessError("corpus JSONL line %d case_id %r has non-integer seed" % (line_number, case_id))
        if "references" in row:
            raise HarnessError("corpus JSONL line %d case_id %r uses reserved field 'references'" % (line_number, case_id))
        seen.add(case_id)
        rows.append(row)
    if not rows:
        raise HarnessError("corpus JSONL is empty: %s" % path)
    return rows


def _token_ids(value: Any, class_name: str, case_id: str, method_name: str) -> List[int]:
    try:
        values = list(value)
    except Exception as exc:
        raise HarnessError("case_id %r %s.%s did not return an iterable: %s" % (case_id, class_name, method_name, exc)) from exc
    for index, token in enumerate(values):
        if isinstance(token, bool) or not isinstance(token, int):
            raise HarnessError(
                "case_id %r %s.%s returned non-integer token at index %d: %r"
                % (case_id, class_name, method_name, index, token)
            )
    return values


def _first_difference(left: Sequence[Any], right: Sequence[Any]) -> Tuple[int, Any, Any] | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index, left[index], right[index]
    if len(left) != len(right):
        index = limit
        left_value = left[index] if index < len(left) else "<end>"
        right_value = right[index] if index < len(right) else "<end>"
        return index, left_value, right_value
    return None


def _assert_equal_sequence(left: Sequence[Any], right: Sequence[Any], message: str) -> None:
    difference = _first_difference(left, right)
    if difference is not None:
        index, left_value, right_value = difference
        raise HarnessError("%s at index %d: %r != %r" % (message, index, left_value, right_value))


def _run_one_tokenizer(tokenizer: Any, class_name: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    case_id = row["case_id"]
    text = row["text"]
    try:
        source_bytes = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise HarnessError(
            "case_id %r text cannot be encoded as UTF-8: %s" % (case_id, exc)
        ) from exc
    try:
        encoded = _token_ids(tokenizer.encode(text), class_name, case_id, "encode")
        encoded_bytes = _token_ids(tokenizer.encodeBytes(source_bytes), class_name, case_id, "encodeBytes")
    except HarnessError:
        raise
    except Exception as exc:
        raise HarnessError("case_id %r %s encode failed: %s: %s" % (case_id, class_name, type(exc).__name__, exc)) from exc
    _assert_equal_sequence(encoded, encoded_bytes, "case_id %r %s encode/encodeBytes mismatch" % (case_id, class_name))

    try:
        decoded_bytes = tokenizer.decodeBytes(encoded)
        decoded_text = tokenizer.decode(encoded)
    except Exception as exc:
        raise HarnessError("case_id %r %s decode failed: %s: %s" % (case_id, class_name, type(exc).__name__, exc)) from exc
    if not isinstance(decoded_bytes, bytes):
        raise HarnessError("case_id %r %s decodeBytes returned %s, not bytes" % (case_id, class_name, type(decoded_bytes).__name__))
    if decoded_bytes != source_bytes:
        _assert_equal_sequence(
            list(decoded_bytes),
            list(source_bytes),
            "case_id %r %s exact byte roundtrip mismatch" % (case_id, class_name),
        )
    if decoded_text != text:
        raise HarnessError("case_id %r %s exact text roundtrip mismatch: %r != %r" % (case_id, class_name, decoded_text, text))

    return {
        "token_ids": encoded,
        "decoded_bytes": list(decoded_bytes),
        "decoded_bytes_hex": decoded_bytes.hex(),
        "decoded_text": decoded_text,
        "roundtrip_exact": True,
    }


def run_fixture(reference_source: Path, vocab: Path, corpus: Path, output: Path) -> Dict[str, Any]:
    """Generate one B16a fixture, raising ``HarnessError`` on the first failure."""

    reference_source = _require_file(reference_source, "reference source")
    vocab = _require_file(vocab, "vocabulary")
    corpus = _require_file(corpus, "corpus JSONL")
    output = output.expanduser().resolve()
    if output in (reference_source, vocab, corpus):
        raise HarnessError("output path must be distinct from every input path: %s" % output)
    source_hash, source_size = sha256_file(reference_source)
    vocab_hash, vocab_size = sha256_file(vocab)
    corpus_hash, corpus_size = sha256_file(corpus)

    print(
        "input reference_source path=%s sha256=%s"
        % (reference_source, source_hash),
        flush=True,
    )
    print(
        "input vocab path=%s sha256=%s" % (vocab, vocab_hash),
        flush=True,
    )
    print(
        "input corpus path=%s sha256=%s" % (corpus, corpus_hash),
        flush=True,
    )

    rows = _load_corpus(corpus)
    classes = load_reference_classes(reference_source, vocab)
    try:
        tokenizers = {name: classes[name](str(vocab)) for name in PUBLIC_TOKENIZER_NAMES}
    except Exception as exc:
        raise HarnessError("failed to construct tokenizer reference: %s: %s" % (type(exc).__name__, exc)) from exc

    cases: List[Dict[str, Any]] = []
    for row in rows:
        results = {name: _run_one_tokenizer(tokenizers[name], name, row) for name in PUBLIC_TOKENIZER_NAMES}
        left = results[PUBLIC_TOKENIZER_NAMES[0]]["token_ids"]
        right = results[PUBLIC_TOKENIZER_NAMES[1]]["token_ids"]
        _assert_equal_sequence(
            left,
            right,
            "case_id %r public tokenizer token ID mismatch" % row["case_id"],
        )
        _assert_equal_sequence(
            results[PUBLIC_TOKENIZER_NAMES[0]]["decoded_bytes"],
            results[PUBLIC_TOKENIZER_NAMES[1]]["decoded_bytes"],
            "case_id %r public tokenizer decoded-byte mismatch" % row["case_id"],
        )
        case_output = dict(row)
        case_output["references"] = results
        case_output["input_row"] = row
        cases.append(case_output)

    repo_root = Path(__file__).resolve().parents[1]
    git = _git_revision(repo_root)
    manifest = {
        "schema_version": "b16a-tokenizer-reference-fixture-v1",
        "reference_extraction": {
            "selected_class_definitions": list(REFERENCE_CLASS_NAMES),
            "executed_class_definitions": list(REFERENCE_CLASS_NAMES),
            "public_tokenizers_compared": list(PUBLIC_TOKENIZER_NAMES),
            "method": "AST parse of the full source; compile only selected class definitions in a controlled namespace",
            "trusted_source_and_vocab_required": True,
            "namespace_is_not_a_security_sandbox": True,
            "top_level_code_executed": False,
            "full_source_sha256": source_hash,
        },
        "inputs": {
            "reference_source": _input_metadata(reference_source, source_hash, source_size),
            "vocab": _input_metadata(vocab, vocab_hash, vocab_size),
            "corpus": _input_metadata(corpus, corpus_hash, corpus_size),
        },
        "input_hashes": {
            "reference_source": source_hash,
            "vocab": vocab_hash,
            "corpus": corpus_hash,
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
        "case_count": len(cases),
        "cases": cases,
    }
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise HarnessError("cannot write output fixture %r: %s" % (str(output), exc)) from exc
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an independent B16a RWKV tokenizer reference fixture.")
    parser.add_argument("--reference-source", "--reference-source-path", dest="reference_source", required=True, type=Path)
    parser.add_argument("--vocab", "--vocab-path", dest="vocab", required=True, type=Path)
    parser.add_argument("--corpus", "--corpus-jsonl", dest="corpus", required=True, type=Path)
    parser.add_argument("--output", "--output-path", dest="output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        manifest = run_fixture(args.reference_source, args.vocab, args.corpus, args.output)
    except HarnessError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    print("wrote %s cases=%d reference_source_sha256=%s" % (args.output, manifest["case_count"], manifest["input_hashes"]["reference_source"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
