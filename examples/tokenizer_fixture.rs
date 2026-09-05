#![warn(clippy::undocumented_unsafe_blocks)]

use rwkv_tokenizer::{DecodeError, EncodeError, RwkvTokenizer, PARALLEL_MIN_BYTES};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

const MAX_WORKERS: usize = 16;
const EMBEDDED_SOURCE: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"));

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Sequential,
    Parallel,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "sequential" => Ok(Self::Sequential),
            "parallel" => Ok(Self::Parallel),
            other => Err(format!(
                "invalid --mode {other:?}; expected sequential or parallel"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
        }
    }
}

#[derive(Debug)]
struct Config {
    vocab: PathBuf,
    fixture: PathBuf,
    output: PathBuf,
    mode: Mode,
    workers: usize,
    iterations: usize,
}

fn parse_args<I>(args: I) -> Result<Config, String>
where
    I: IntoIterator<Item = String>,
{
    let mut vocab = None;
    let mut fixture = None;
    let mut output = None;
    let mut mode = None;
    let mut workers = None;
    let mut iterations = None;
    let mut args = args.into_iter();

    while let Some(option) = args.next() {
        let mut value = |name: &str, slot: &mut Option<String>| -> Result<(), String> {
            if slot.is_some() {
                return Err(format!("duplicate option {name}"));
            }
            *slot = Some(
                args.next()
                    .ok_or_else(|| format!("missing value for {name}"))?,
            );
            Ok(())
        };

        match option.as_str() {
            "--vocab" => value("--vocab", &mut vocab)?,
            "--fixture" => value("--fixture", &mut fixture)?,
            "--output" => value("--output", &mut output)?,
            "--mode" => value("--mode", &mut mode)?,
            "--workers" => value("--workers", &mut workers)?,
            "--iterations" => value("--iterations", &mut iterations)?,
            _ => return Err(format!("unknown option {option:?}")),
        }
    }

    let required = |name: &str, slot: Option<String>| {
        slot.ok_or_else(|| format!("missing required option {name}"))
    };
    let mode = Mode::parse(&required("--mode", mode)?)?;
    let workers: usize = required("--workers", workers)?
        .parse()
        .map_err(|_| "--workers must be a positive integer".to_owned())?;
    if workers == 0 || workers > MAX_WORKERS {
        return Err(format!(
            "--workers must be between 1 and {MAX_WORKERS}, got {workers}"
        ));
    }
    match mode {
        Mode::Sequential if workers != 1 => {
            return Err(format!(
                "sequential mode requires --workers 1, got {workers}"
            ));
        }
        Mode::Parallel if workers < 2 => {
            return Err(format!(
                "parallel mode requires --workers >= 2, got {workers}"
            ));
        }
        _ => {}
    }
    let iterations: usize = required("--iterations", iterations)?
        .parse()
        .map_err(|_| "--iterations must be a positive integer".to_owned())?;
    if iterations == 0 {
        return Err("--iterations must be greater than zero".to_owned());
    }

    Ok(Config {
        vocab: PathBuf::from(required("--vocab", vocab)?),
        fixture: PathBuf::from(required("--fixture", fixture)?),
        output: PathBuf::from(required("--output", output)?),
        mode,
        workers,
        iterations,
    })
}

#[derive(Debug)]
struct Case {
    case_id: String,
    text: String,
    seed: u64,
    expected_tokens: Vec<u32>,
    expected_bytes: Vec<u8>,
}

#[derive(Debug)]
struct Fixture {
    cases: Vec<Case>,
    vocab_sha256: String,
}

fn member<'a>(value: &'a Value, name: &str, context: &str) -> Result<&'a Value, String> {
    value
        .get(name)
        .ok_or_else(|| format!("{context} missing required field {name:?}"))
}

fn string_member(value: &Value, name: &str, context: &str) -> Result<String, String> {
    member(value, name, context)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{context}.{name} must be a string"))
}

fn number_array<T, F>(value: &Value, context: &str, convert: F) -> Result<Vec<T>, String>
where
    F: Fn(u64) -> Option<T>,
{
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let number = value
                .as_u64()
                .ok_or_else(|| format!("{context}[{index}] must be a non-negative integer"))?;
            convert(number).ok_or_else(|| format!("{context}[{index}] is out of range: {number}"))
        })
        .collect()
}

fn parse_fixture(bytes: &[u8]) -> Result<Fixture, String> {
    let root: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid fixture JSON: {error}"))?;
    let vocab_sha256 = string_member(
        member(&root, "input_hashes", "fixture")?,
        "vocab",
        "fixture.input_hashes",
    )?;
    let values = member(&root, "cases", "fixture")?
        .as_array()
        .ok_or_else(|| "fixture.cases must be an array".to_owned())?;
    if values.is_empty() {
        return Err("fixture.cases must not be empty".to_owned());
    }
    if let Some(case_count) = root.get("case_count") {
        let expected = case_count
            .as_u64()
            .ok_or_else(|| "fixture.case_count must be an integer".to_owned())?;
        if expected != values.len() as u64 {
            return Err(format!(
                "fixture.case_count mismatch: declared {expected}, actual {}",
                values.len()
            ));
        }
    }

    let mut seen_case_ids = HashSet::with_capacity(values.len());
    let mut cases = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let context = format!("fixture.cases[{index}]");
        let case_id = string_member(value, "case_id", &context)?;
        if case_id.is_empty() {
            return Err(format!("{context}.case_id must not be empty"));
        }
        if !seen_case_ids.insert(case_id.clone()) {
            return Err(format!("{context} duplicates case_id {case_id:?}"));
        }
        let text = string_member(value, "text", &context)?;
        let seed = member(value, "seed", &context)?
            .as_u64()
            .ok_or_else(|| format!("{context}.seed must be a non-negative integer"))?;
        let reference = member(value, "references", &context)?;
        let reference = member(reference, "RWKV_TOKENIZER", &context)?;
        let expected_tokens = number_array(
            member(reference, "token_ids", &context)?,
            &format!("{context}.references.RWKV_TOKENIZER.token_ids"),
            |number| u32::try_from(number).ok(),
        )?;
        let expected_bytes = number_array(
            member(reference, "decoded_bytes", &context)?,
            &format!("{context}.references.RWKV_TOKENIZER.decoded_bytes"),
            |number| u8::try_from(number).ok(),
        )?;
        cases.push(Case {
            case_id,
            text,
            seed,
            expected_tokens,
            expected_bytes,
        });
    }
    Ok(Fixture {
        cases,
        vocab_sha256,
    })
}

fn compare_sequence<T: PartialEq + std::fmt::Debug>(
    case_id: &str,
    kind: &str,
    expected: &[T],
    actual: &[T],
) -> Result<(), String> {
    let common = expected.len().min(actual.len());
    for index in 0..common {
        if expected[index] != actual[index] {
            return Err(format!(
                "case_id {case_id:?} {kind} index {index}: expected {:?}, actual {:?}",
                expected[index], actual[index]
            ));
        }
    }
    if expected.len() != actual.len() {
        let index = common;
        return Err(format!(
            "case_id {case_id:?} {kind} index {index}: length mismatch, expected {}, actual {}",
            expected.len(),
            actual.len()
        ));
    }
    Ok(())
}

fn compare_case(case: &Case, actual_tokens: &[u32], actual_bytes: &[u8]) -> Result<(), String> {
    compare_sequence(&case.case_id, "token", &case.expected_tokens, actual_tokens)?;
    compare_sequence(
        &case.case_id,
        "decoded byte",
        &case.expected_bytes,
        actual_bytes,
    )
}

fn encode_case(
    tokenizer: &RwkvTokenizer,
    case: &Case,
    mode: Mode,
    workers: usize,
) -> Result<(Vec<u32>, &'static str, usize), String> {
    let input = case.text.as_bytes();
    match mode {
        Mode::Sequential => tokenizer
            .encode_sequential(input)
            .map(|tokens| (tokens, "sequential", 1))
            .map_err(|error: EncodeError| {
                format!("case_id {:?} encode failed: {error}", case.case_id)
            }),
        Mode::Parallel => {
            let fallback = input.len() < PARALLEL_MIN_BYTES;
            tokenizer
                .encode_parallel_with_workers(input, workers)
                .map(|tokens| {
                    if fallback {
                        (tokens, "sequential-fallback", 1)
                    } else {
                        (tokens, "parallel", workers)
                    }
                })
                .map_err(|error: EncodeError| {
                    format!("case_id {:?} encode failed: {error}", case.case_id)
                })
        }
    }
}

fn verify_case(tokenizer: &RwkvTokenizer, case: &Case, config: &Config) -> Result<Value, String> {
    let mut iteration_nanos = Vec::with_capacity(config.iterations);
    let mut encode_nanos = Vec::with_capacity(config.iterations);
    let mut decode_nanos = Vec::with_capacity(config.iterations);
    let mut token_count = 0;
    let mut decoded_byte_count = 0;
    let mut actual_path = "";
    let mut actual_workers = 0;
    let mut final_tokens = Vec::new();
    let mut final_decoded = Vec::new();
    let started = Instant::now();
    for _ in 0..config.iterations {
        let iteration_started = Instant::now();
        let encode_started = Instant::now();
        let (tokens, path, workers) = encode_case(tokenizer, case, config.mode, config.workers)?;
        encode_nanos.push(encode_started.elapsed().as_nanos());
        let decode_started = Instant::now();
        let decoded = tokenizer
            .decode_bytes(&tokens)
            .map_err(|error: DecodeError| {
                format!("case_id {:?} decode failed: {error}", case.case_id)
            })?;
        decode_nanos.push(decode_started.elapsed().as_nanos());
        compare_case(case, &tokens, &decoded)?;
        token_count = tokens.len();
        decoded_byte_count = decoded.len();
        actual_path = path;
        actual_workers = workers;
        final_tokens = tokens;
        final_decoded = decoded;
        iteration_nanos.push(iteration_started.elapsed().as_nanos());
    }
    Ok(json!({
        "case_id": case.case_id,
        "seed": case.seed,
        "text_bytes": case.text.len(),
        "requested_mode": config.mode.as_str(),
        "actual_path": actual_path,
        "actual_workers": actual_workers,
        "fallback": actual_path == "sequential-fallback",
        "outcome": "pass",
        "pass": true,
        "token_count": token_count,
        "decoded_byte_count": decoded_byte_count,
        "token_ids": final_tokens,
        "decoded_bytes": final_decoded,
        "elapsed_nanos_including_validation": started.elapsed().as_nanos(),
        "iteration_nanos_including_validation": iteration_nanos,
        "encode_nanos": encode_nanos,
        "decode_nanos": decode_nanos,
    }))
}

#[derive(Debug)]
struct InputInfo {
    path: PathBuf,
    sha256: String,
    size_bytes: usize,
}

struct Inputs<'a> {
    fixture: &'a InputInfo,
    vocab: &'a InputInfo,
    source: &'a InputInfo,
    binary: &'a InputInfo,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_input(path: PathBuf, label: &str) -> Result<(InputInfo, Vec<u8>), String> {
    let bytes = fs::read(&path).map_err(|error| format!("read {label} {:?}: {error}", path))?;
    let info = InputInfo {
        path,
        sha256: sha256_hex(&bytes),
        size_bytes: bytes.len(),
    };
    Ok((info, bytes))
}

fn resolve_existing(path: &Path, label: &str) -> Result<PathBuf, String> {
    fs::canonicalize(path).map_err(|error| format!("resolve {label} {:?}: {error}", path))
}

fn input_json(info: &InputInfo) -> Value {
    json!({
        "path": info.path,
        "sha256": info.sha256,
        "size_bytes": info.size_bytes,
    })
}

fn git_output(root: &Path, arguments: &[&str]) -> Result<String, String> {
    let safe_directory = format!("safe.directory={}", root.display());
    let output = Command::new("git")
        .arg("-c")
        .arg(safe_directory)
        .arg("-C")
        .arg(root)
        .args(arguments)
        .output()
        .map_err(|error| format!("run git {:?}: {error}", arguments))?;
    if !output.status.success() {
        return Err(format!(
            "git {:?} failed: {}",
            arguments,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn git_provenance(root: &Path) -> Result<Value, String> {
    let revision = git_output(root, &["rev-parse", "--verify", "HEAD"])?
        .trim()
        .to_owned();
    if revision.is_empty() {
        return Err("git returned an empty revision".to_owned());
    }
    let status = git_output(root, &["status", "--porcelain=v1", "--untracked-files=all"])?;
    Ok(json!({
        "revision": revision,
        "dirty": !status.is_empty(),
        "status_porcelain": status.lines().collect::<Vec<_>>(),
    }))
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(env::current_dir()
            .map_err(|error| format!("resolve current directory: {error}"))?
            .join(path))
    }
}

fn guard_output_path(output: &Path, inputs: &[&Path]) -> Result<(), String> {
    if output.exists() {
        return Err(format!(
            "refusing to overwrite existing output {:?}",
            output
        ));
    }
    let file_name = output
        .file_name()
        .ok_or_else(|| format!("output path has no file name: {:?}", output))?;
    let parent = output
        .parent()
        .ok_or_else(|| format!("output path has no parent: {:?}", output))?;
    let candidate = fs::canonicalize(parent)
        .map_err(|error| format!("resolve output parent {:?}: {error}", parent))?
        .join(file_name);
    if let Some(input) = inputs.iter().find(|input| candidate == **input) {
        return Err(format!("output {:?} aliases input {:?}", output, input));
    }
    Ok(())
}

fn run(config: Config) -> Result<(), String> {
    let vocab_path = resolve_existing(&config.vocab, "vocab")?;
    let fixture_path = resolve_existing(&config.fixture, "fixture")?;
    let output_path = absolute_path(&config.output)?;
    let binary_path = resolve_existing(
        &env::current_exe().map_err(|error| format!("resolve example binary: {error}"))?,
        "example binary",
    )?;
    guard_output_path(&output_path, &[&vocab_path, &fixture_path, &binary_path])?;

    let (fixture_info, fixture_bytes) = read_input(fixture_path, "fixture")?;
    let (vocab_info, vocab_bytes) = read_input(vocab_path, "vocab")?;
    let (binary_info, _) = read_input(binary_path, "example binary")?;
    let source_info = InputInfo {
        path: PathBuf::from("embedded:src/lib.rs"),
        sha256: sha256_hex(EMBEDDED_SOURCE),
        size_bytes: EMBEDDED_SOURCE.len(),
    };
    let git = package_git_provenance();

    println!(
        "input fixture path={} sha256={}",
        fixture_info.path.display(),
        fixture_info.sha256
    );
    println!(
        "input vocab path={} sha256={}",
        vocab_info.path.display(),
        vocab_info.sha256
    );
    println!(
        "input source path={} sha256={}",
        source_info.path.display(),
        source_info.sha256
    );
    println!(
        "input binary path={} sha256={}",
        binary_info.path.display(),
        binary_info.sha256
    );
    println!(
        "mode={} workers={} iterations={}",
        config.mode.as_str(),
        config.workers,
        config.iterations
    );

    let fixture = parse_fixture(&fixture_bytes)?;
    if fixture.vocab_sha256 != vocab_info.sha256 {
        return Err(format!(
            "fixture vocab SHA-256 mismatch: fixture declares {}, supplied vocab is {}",
            fixture.vocab_sha256, vocab_info.sha256
        ));
    }
    let cases = fixture.cases;
    let vocab_text =
        String::from_utf8(vocab_bytes).map_err(|error| format!("vocab is not UTF-8: {error}"))?;
    let tokenizer = RwkvTokenizer::from_vocab_str(&vocab_text)
        .map_err(|error| format!("load vocab {:?}: {error}", vocab_info.path))?;

    let mut outcomes = Vec::with_capacity(cases.len());
    let inputs = Inputs {
        fixture: &fixture_info,
        vocab: &vocab_info,
        source: &source_info,
        binary: &binary_info,
    };
    for case in &cases {
        match verify_case(&tokenizer, case, &config) {
            Ok(outcome) => outcomes.push(outcome),
            Err(error) => {
                outcomes.push(json!({
                    "case_id": case.case_id,
                    "seed": case.seed,
                    "requested_mode": config.mode.as_str(),
                    "outcome": "fail",
                    "pass": false,
                    "error": error,
                }));
                let manifest =
                    manifest(&config, &inputs, git.clone(), outcomes, cases.len(), false);
                write_manifest(&output_path, &manifest)?;
                return Err(error);
            }
        }
    }

    let manifest = manifest(&config, &inputs, git, outcomes, cases.len(), true);
    write_manifest(&output_path, &manifest)
}

fn package_git_provenance() -> Value {
    let package_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    match git_provenance(&package_root) {
        Ok(value) => value,
        Err(error) => json!({
            "available": false,
            "error": error,
        }),
    }
}

fn manifest(
    config: &Config,
    inputs: &Inputs<'_>,
    git: Value,
    outcomes: Vec<Value>,
    case_count: usize,
    pass: bool,
) -> Value {
    json!({
        "schema_version": "b16a-tokenizer-rust-verification-v1",
        "pass": pass,
        "case_count": case_count,
        "mode": config.mode.as_str(),
        "workers": config.workers,
        "iterations": config.iterations,
        "inputs": {
            "fixture": input_json(inputs.fixture),
            "vocab": input_json(inputs.vocab),
            "source": input_json(inputs.source),
            "binary": input_json(inputs.binary),
        },
        "runtime": { "git": git },
        "cases": outcomes,
    })
}

fn write_manifest(path: &Path, manifest: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|error| format!("serialize output {:?}: {error}", path))?;
    fs::write(path, bytes).map_err(|error| format!("write output {:?}: {error}", path))
}

fn main() {
    let result = env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .pipe(parse_args)
        .and_then(run);
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

trait Pipe: Sized {
    fn pipe<T>(self, function: impl FnOnce(Self) -> T) -> T {
        function(self)
    }
}

impl<T> Pipe for T {}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(mode: &str, workers: &str) -> Vec<String> {
        [
            "--vocab",
            "vocab.txt",
            "--fixture",
            "fixture.json",
            "--output",
            "output.json",
            "--mode",
            mode,
            "--workers",
            workers,
            "--iterations",
            "1",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn explicit_worker_modes_are_validated() {
        assert!(parse_args(args("sequential", "1")).is_ok());
        assert!(parse_args(args("parallel", "2")).is_ok());
        for (mode, workers) in [
            ("sequential", "0"),
            ("parallel", "0"),
            ("sequential", "2"),
            ("parallel", "1"),
            ("parallel", "17"),
        ] {
            assert!(parse_args(args(mode, workers)).is_err(), "{mode} {workers}");
        }
    }

    #[test]
    fn mutated_expected_token_is_a_failing_positive_control() {
        let case = Case {
            case_id: "mutated-expected-token".to_owned(),
            text: "a".to_owned(),
            seed: 1,
            expected_tokens: vec![2],
            expected_bytes: vec![b'a'],
        };
        let error = compare_case(&case, &[1], b"a").unwrap_err();
        assert!(error.contains("case_id \"mutated-expected-token\" token index 0"));
    }

    #[test]
    fn duplicate_fixture_case_ids_are_rejected() {
        let fixture = br#"{
            "case_count": 2,
            "input_hashes": {"vocab": "fixture-vocab"},
            "cases": [
                {"case_id": "same", "text": "", "seed": 1,
                 "references": {"RWKV_TOKENIZER": {"token_ids": [], "decoded_bytes": []}}},
                {"case_id": "same", "text": "", "seed": 2,
                 "references": {"RWKV_TOKENIZER": {"token_ids": [], "decoded_bytes": []}}}
            ]
        }"#;
        let error = parse_fixture(fixture).unwrap_err();
        assert!(error.contains("duplicates case_id \"same\""));
    }

    #[test]
    fn existing_output_is_rejected_before_write() {
        let output = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let error = guard_output_path(&output, &[]).unwrap_err();
        assert!(error.contains("refusing to overwrite existing output"));
    }
}
