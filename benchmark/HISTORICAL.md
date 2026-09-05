# Historical measurement (before vocabulary bundling)

These tentative results describe an earlier run, not a fresh benchmark of
this package. Private checkout identifiers and machine labels have been
omitted from this public summary. The measurements below are unchanged.

The run used the official vocabulary identified in [NOTICE](../NOTICE) and
BlinkDL's Python TRIE_TOKENIZER. Reference source SHA-256:
`f4447dc3adae838e4af443b7cece53217d3ccde9d7f4fbfade42c6100e4362da`.
See [reproduction instructions](README.md) for the pinned public source.

## Recorded measurement

This is one measured CPU-only pair, not a universal performance claim or a
speedup headline. Both implementations validated all 76 cases exactly.

| implementation | measured wall | complete corpus cycles | validated cases | encode samples | decode samples |
|---|---:|---:|---:|---:|---:|
| pure Rust | 300.033216988 s | 1,562 | 118,712 | 118,712 | 118,712 |
| public Python TRIE | 300.016014682 s | 5,479 | 416,404 | 416,404 | 416,404 |

Inner operation distributions, in nanoseconds:

| implementation | encode median / p95 | decode median / p95 |
|---|---:|---:|
| pure Rust | 2,695 / 1,042,541 | 461 / 100,820.45 |
| public Python TRIE | 12,413 / 6,042,009 | 2,866 / 805,807.5 |

Measured host context:

| field | recorded value |
|---|---|
| CPU | AMD Ryzen 7 9700X 8-Core Processor; 16 logical CPUs |
| OS | x86_64 Linux under WSL2, kernel `6.18.33.2-microsoft-standard-WSL2`, glibc 2.43 |
| Python | 3.12.14 |
| Rust compiler | unavailable in the retained manifest; not guessed |

The outer wall duration includes corpus validation and, for Rust, repeated
fixture-runner child startup/orchestration. Rust resource CPU/RSS came from
`RUSAGE_CHILDREN`; Python resource CPU/RSS came from in-process
`RUSAGE_SELF`. Those outer resource and cycle values are intentionally not
treated as apples-to-apples comparisons, and no RAM-saving claim is made.

The retained manifests and raw samples are release-gate artifacts outside
this package. Re-run the gate after changing the source, vocabulary, fixture,
reference, or benchmark harness.
