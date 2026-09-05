# Reproducible tokeniser benchmark inputs

This directory contains the small, standalone benchmark drivers and the
fixed 76-row corpus. It does not contain the expanded fixture, vocabulary, or
model files. The corpus is deterministic input data with SHA-256
`3daea831cef996f46ca9db69ee8b6686d8c24c48cd0c6103b8b842c1dc9d43b6`.

## Prepare pinned inputs

The reference source is pinned to a public commit. Use the bundled vocabulary
in `vocab/`; both inputs are checked against their recorded hashes.

Linux:

```sh
set -eu
mkdir -p benchmark/input
curl --fail --location --silent --show-error \
  --output benchmark/input/rwkv_tokenizer.py \
  https://raw.githubusercontent.com/BlinkDL/ChatRWKV/2e2bb1cd390cefbedae0a89f4a343f6f754d6621/tokenizer/rwkv_tokenizer.py
test "$(sha256sum benchmark/input/rwkv_tokenizer.py | cut -d' ' -f1)" = \
  f4447dc3adae838e4af443b7cece53217d3ccde9d7f4fbfade42c6100e4362da
test "$(sha256sum vocab/rwkv_vocab_v20230424.txt | cut -d' ' -f1)" = \
  e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89
```

Windows PowerShell:

```powershell
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force .\benchmark\input | Out-Null
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/BlinkDL/ChatRWKV/2e2bb1cd390cefbedae0a89f4a343f6f754d6621/tokenizer/rwkv_tokenizer.py' -OutFile .\benchmark\input\rwkv_tokenizer.py
if ((Get-FileHash .\benchmark\input\rwkv_tokenizer.py -Algorithm SHA256).Hash -ne 'F4447DC3ADAE838E4AF443B7CECE53217D3CCDE9D7F4FBFADE42C6100E4362DA') { throw 'reference source hash mismatch' }
if ((Get-FileHash .\vocab\rwkv_vocab_v20230424.txt -Algorithm SHA256).Hash -ne 'E6DEE3D4E31B4D5C40AC99508AC6C701CEEF4BED681BF2167CE9A908552BCA89') { throw 'vocabulary hash mismatch' }
```

Stop if either input hash changes; do not benchmark a replacement silently.

## Generate and verify a fixture

From the package root, with Python 3.10 or newer:

```sh
python benchmark/tokenizer_reference.py \
  --reference-source benchmark/input/rwkv_tokenizer.py \
  --vocab vocab/rwkv_vocab_v20230424.txt \
  --corpus benchmark/corpus-extended.jsonl \
  --output benchmark/tokenizer-fixture-extended.json
sha256sum benchmark/tokenizer-fixture-extended.json
```

The generated fixture has 76 cases and contains both public reference
implementations. The Rust runner consumes the `RWKV_TOKENIZER` rows; the
Python driver below consumes its `TRIE_TOKENIZER` rows. Both drivers reject
duplicate case IDs and validate every ID and decoded byte before timing.
The generated JSON includes runtime metadata, so its complete-file SHA-256 is
run-local. Reproduction compares the recorded corpus, source, and vocabulary
hashes plus semantic rows; record the newly generated fixture hash in the
result rather than requiring the historical full-file hash across platforms.

Windows PowerShell uses the same generator:

```powershell
$ErrorActionPreference = 'Stop'
python .\benchmark\tokenizer_reference.py --reference-source .\benchmark\input\rwkv_tokenizer.py --vocab .\vocab\rwkv_vocab_v20230424.txt --corpus .\benchmark\corpus-extended.jsonl --output .\benchmark\tokenizer-fixture-extended.json
if ($LASTEXITCODE -ne 0) { throw 'fixture generation failed' }
Get-FileHash .\benchmark\tokenizer-fixture-extended.json -Algorithm SHA256
```

## Run the drivers

The Rust example is independent of any sibling repository and embeds the
library source hash at compile time:

```sh
cargo run --release --example tokenizer_fixture -- \
  --vocab vocab/rwkv_vocab_v20230424.txt \
  --fixture benchmark/tokenizer-fixture-extended.json \
  --output benchmark/rust-sequential.json \
  --mode sequential --workers 1 --iterations 3
```

The public Python TRIE driver uses the same fixture and separates encode and
decode timings:

```sh
python benchmark/bench_tokenizer_reference.py \
  --reference-source benchmark/input/rwkv_tokenizer.py \
  --vocab vocab/rwkv_vocab_v20230424.txt \
  --fixture benchmark/tokenizer-fixture-extended.json \
  --output benchmark/python-trie.json \
  --implementation trie --iterations 3
```

For the release gate, use the package-local measured-window wrapper. It
requires the five-minute duration explicitly, refuses an existing output, and
uses unique temporary child outputs. Build the Rust example once before the
window, then run these commands serially.

Linux:

```sh
set -eu
cargo build --release --example tokenizer_fixture --locked
python benchmark/run_duration.py --implementation rust \
  --rust-runner target/release/examples/tokenizer_fixture \
  --reference-source benchmark/input/rwkv_tokenizer.py \
  --vocab vocab/rwkv_vocab_v20230424.txt \
  --fixture benchmark/tokenizer-fixture-extended.json \
  --output benchmark/rust-five-minute.json --duration-seconds 300 --warmups 3
python benchmark/run_duration.py --implementation trie \
  --reference-source benchmark/input/rwkv_tokenizer.py \
  --vocab vocab/rwkv_vocab_v20230424.txt \
  --fixture benchmark/tokenizer-fixture-extended.json \
  --output benchmark/python-trie-five-minute.json --duration-seconds 300 --warmups 3
```

Windows PowerShell:

```powershell
$ErrorActionPreference = 'Stop'
cargo build --release --example tokenizer_fixture --locked
if ($LASTEXITCODE -ne 0) { throw 'Rust example build failed' }
python .\benchmark\run_duration.py --implementation rust --rust-runner .\target\release\examples\tokenizer_fixture.exe --reference-source .\benchmark\input\rwkv_tokenizer.py --vocab .\vocab\rwkv_vocab_v20230424.txt --fixture .\benchmark\tokenizer-fixture-extended.json --output .\benchmark\rust-five-minute.json --duration-seconds 300 --warmups 3
if ($LASTEXITCODE -ne 0) { throw 'Rust measured window failed' }
python .\benchmark\run_duration.py --implementation trie --reference-source .\benchmark\input\rwkv_tokenizer.py --vocab .\vocab\rwkv_vocab_v20230424.txt --fixture .\benchmark\tokenizer-fixture-extended.json --output .\benchmark\python-trie-five-minute.json --duration-seconds 300 --warmups 3
if ($LASTEXITCODE -ne 0) { throw 'Python measured window failed' }
```

The wrapper's wall duration includes child startup, fixture validation, and
orchestration. Its retained encode/decode arrays are the inner call timings;
use those fields for implementation comparisons and do not derive speed or
memory claims from the outer window or mismatched process-resource scopes.
