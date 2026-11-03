#![forbid(unsafe_code)]

//! Standalone RWKV World tokenizer: a greedy longest-match trie matcher in pure Rust
//! with no runtime dependencies.
//!
//! [`RwkvTokenizer::bundled`] loads the official `rwkv_vocab_v20230424` vocabulary;
//! [`RwkvTokenizer::from_vocab_str`] and [`RwkvTokenizer::from_file`] load your own.
//! Encoding turns text or bytes into token IDs, decoding turns IDs back. Heavy work
//! shares a process-wide CPU budget, see [`cooperation_stats`].
//!
//! ```
//! use rwkv_tokenizer::RwkvTokenizer;
//!
//! let tokenizer = RwkvTokenizer::bundled().expect("valid bundled vocabulary");
//! let ids = tokenizer.encode("Hello, world!").expect("encodable text");
//! assert_eq!(tokenizer.decode_utf8(&ids).unwrap(), "Hello, world!");
//! ```

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

use std::{fmt, path::Path, sync::{Arc, mpsc, atomic::{AtomicUsize, Ordering}}};

mod cooperate;
pub use cooperate::Stats as CooperationStats;
/// Process-wide activity for this crate. Peaks and counters are cumulative.
pub fn cooperation_stats() -> CooperationStats { cooperate::stats() }

/// Number of token IDs (0..65536); ID 0 is reserved.
pub const VOCAB_SIZE: usize = 65_536;
const _: () = assert!(VOCAB_SIZE == 1 << 16, "token IDs are stored as u16");
/// Token ID 0 has no text; treat it as end-of-text.
pub const RESERVED_TOKEN_ID: u32 = 0;
/// Input length below which parallel encoding stays sequential by default.
pub const PARALLEL_MIN_BYTES: usize = 1 << 17;
/// Total batch size below which `encode_batch_auto` stays sequential.
const BATCH_MIN_BYTES: usize = 1 << 16;
/// Batch bytes per worker chosen by `encode_batch_auto`.
const BATCH_BYTES_PER_WORKER: usize = 1 << 15;

/// A byte with no matching token (partial vocabularies only).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct EncodeError {
    /// Offset of the unencodable byte in the input.
    pub byte_offset: usize,
    /// The unencodable byte.
    pub byte: u8,
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no token for byte 0x{:02x} at byte offset {}", self.byte, self.byte_offset)
    }
}
impl std::error::Error for EncodeError {}

/// An [`EncodeError`] tagged with the failing document.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct BatchEncodeError {
    /// Index of the first failing document.
    pub document_index: usize,
    /// The underlying error.
    pub source: EncodeError,
}
impl fmt::Display for BatchEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "document {}: {}", self.document_index, self.source)
    }
}
impl std::error::Error for BatchEncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { Some(&self.source) }
}

/// Why decoding failed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// The ID is out of range or has no entry (including reserved ID 0).
    MissingToken {
        /// Position in the input slice.
        token_index: usize,
        /// The offending ID.
        token_id: u32,
    },
    /// The decoded bytes are not valid UTF-8.
    InvalidUtf8(std::string::FromUtf8Error),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingToken { token_index, token_id } =>
                write!(f, "missing token ID {token_id} at token index {token_index}"),
            Self::InvalidUtf8(error) => write!(f, "decoded bytes are not UTF-8: {error}"),
        }
    }
}
impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidUtf8(error) => Some(error),
            Self::MissingToken { .. } => None,
        }
    }
}

/// Why a vocabulary could not be loaded.
#[derive(Debug)]
#[non_exhaustive]
pub enum VocabError {
    /// The vocabulary file could not be read.
    Io {
        /// The file that failed to load.
        path: std::path::PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// The vocabulary text is malformed; the message names the line and problem.
    Invalid(String),
}

impl fmt::Display for VocabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{path:?}: {source}"),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for VocabError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Invalid(_) => None,
        }
    }
}

/// Parse an unsigned decimal made only of ASCII digits (no sign, no spaces).
fn digits<T: std::str::FromStr>(text: &str) -> Option<T> {
    text.bytes().all(|byte| byte.is_ascii_digit()).then(|| text.parse().ok()).flatten()
}

/// Parse the Python-style literal subset used by `rwkv_vocab_v20230424.txt`.
#[cfg(test)]
fn parse_python_literal(s: &str) -> Option<Vec<u8>> {
    let mut result = Vec::new();
    parse_python_literal_into(s, &mut result)?;
    Some(result)
}

/// Same grammar, writing into a reusable buffer (cleared first).
fn parse_python_literal_into(s: &str, result: &mut Vec<u8>) -> Option<()> {
    let (is_bytes, s) = match s.strip_prefix('b') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let raw = s.as_bytes();
    if raw.len() < 2 || (raw[0] != b'\'' && raw[0] != b'"') || raw[0] != raw[raw.len() - 1] {
        return None;
    }
    let quote = raw[0];
    let inner = &raw[1..raw.len() - 1];
    result.clear();
    if !inner.iter().any(|&byte| byte == b'\\' || byte == quote || (is_bytes && !byte.is_ascii())) {
        result.extend_from_slice(inner);
        return Some(());
    }
    let mut utf8 = [0u8; 4];
    let mut i = 0;
    while i < inner.len() {
        if inner[i] != b'\\' {
            if inner[i] == quote { return None; }
            if is_bytes && !inner[i].is_ascii() { return None; }
            result.push(inner[i]);
            i += 1;
            continue;
        }
        if i + 1 >= inner.len() { return None; }
        match inner[i + 1] {
            b'n' => result.push(b'\n'),
            b't' => result.push(b'\t'),
            b'r' => result.push(b'\r'),
            b'\\' => result.push(b'\\'),
            b'\'' => result.push(b'\''),
            b'"' => result.push(b'"'),
            b'0' => result.push(0),
            b'x' => {
                if i + 4 > inner.len() { return None; }
                let hex = std::str::from_utf8(&inner[i + 2..i + 4]).ok().filter(|hex| hex.bytes().all(|b| b.is_ascii_hexdigit()))?;
                let value = u8::from_str_radix(hex, 16).ok()?;
                if is_bytes {
                    result.push(value);
                } else {
                    result.extend_from_slice(char::from_u32(u32::from(value))?.encode_utf8(&mut utf8).as_bytes());
                }
                i += 4;
                continue;
            }
            b'u' => {
                if is_bytes || i + 6 > inner.len() { return None; }
                let hex = std::str::from_utf8(&inner[i + 2..i + 6]).ok().filter(|hex| hex.bytes().all(|b| b.is_ascii_hexdigit()))?;
                let value = u32::from_str_radix(hex, 16).ok()?;
                result.extend_from_slice(char::from_u32(value)?.encode_utf8(&mut utf8).as_bytes());
                i += 6;
                continue;
            }
            _ => return None,
        }
        i += 2;
    }
    Some(())
}

/// Greedy longest-match RWKV World tokenizer. Cloning shares the vocabulary.
#[derive(Debug, Clone)]
pub struct RwkvTokenizer { data: Arc<TokenizerData> }

/// Construction-time trie node; the edge from its parent is `label`.
#[derive(Clone, Copy, Default)]
struct Node { head: u32, tail: u32, next: u32, term: u16, count: u16, label: u8 }

#[derive(Debug)]
struct TokenizerData {
    tok_blob: Vec<u8>,
    tok_off: Vec<u32>,
    single: [i32; 256],
    trie_roots: Vec<u64>,
    transitions: Vec<u64>,
    partial: bool,
}

impl RwkvTokenizer {
    /// Load the bundled official RWKV World vocabulary. No file or download needed.
    pub fn bundled() -> Result<Self, VocabError> {
        Self::from_vocab_str(include_str!("../vocab/rwkv_vocab_v20230424.txt"))
    }

    /// Load a vocabulary file; same rules as [`Self::from_vocab_str`].
    pub fn from_file(path: &Path) -> Result<Self, VocabError> {
        let text = std::fs::read_to_string(path).map_err(|source| VocabError::Io { path: path.to_path_buf(), source })?;
        Self::from_vocab_str(&text)
    }

    /// Load a normal vocabulary and require every byte to have a singleton.
    pub fn from_vocab_str(text: &str) -> Result<Self, VocabError> {
        Self::from_vocab_str_mode(text, false).map_err(VocabError::Invalid)
    }

    /// Explicit partial-vocabulary mode. Missing IDs and singleton bytes are
    /// allowed, but checked encoding reports the first missing byte.
    pub fn from_vocab_str_partial(text: &str) -> Result<Self, VocabError> {
        Self::from_vocab_str_mode(text, true).map_err(VocabError::Invalid)
    }

    fn from_vocab_str_mode(text: &str, partial: bool) -> Result<Self, String> {
        let permit = cooperate::acquire();
        // Tokens are stored in ID order as they are parsed (IDs are strictly ascending),
        // and inserted into the trie immediately: a terminal that is already set is a
        // duplicate, so no HashSet<Vec<u8>> or per-ID Option<Vec<u8>> is needed.
        let mut tok_blob: Vec<u8> = Vec::with_capacity(text.len() / 2);
        let mut tok_off: Vec<u32> = Vec::with_capacity(VOCAB_SIZE + 1);
        let mut single = [-1i32; 256];
        let mut trie_roots = vec![0u32; 256 * 256];
        // Flat trie: the edge into node `n` lives in `nodes[n]` (`label`), siblings are a
        // linked list in insertion order, so no per-node allocation is needed.
        let mut nodes: Vec<Node> = Vec::with_capacity(120_000);
        nodes.push(Node::default());
        let mut bytes: Vec<u8> = Vec::new();
        let mut previous_id = None;
        let mut parsed = 0usize;
        for (line_index, raw_line) in text.lines().enumerate() {
            let line_number = line_index + 1;
            let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
            if line.trim().is_empty() {
                return Err(format!("malformed vocabulary line {line_number}: blank line"));
            }
            let first = line.find(' ').ok_or_else(|| format!("malformed vocabulary line {line_number}: missing separator"))?;
            let last = line.rfind(' ').ok_or_else(|| format!("malformed vocabulary line {line_number}: missing length separator"))?;
            if first == 0 || last <= first + 1 || last + 1 >= line.len() {
                return Err(format!("malformed vocabulary line {line_number}: {line:?}"));
            }
            let id_text = &line[..first];
            // Plain ASCII digits only: `str::parse` would also accept a leading `+`.
            let id: u32 = digits(id_text).ok_or_else(|| format!("bad token ID at line {line_number}: {id_text:?}"))?;
            // VOCAB_SIZE == 1 << 16, so every in-range ID fits u16.
            let Ok(id16) = u16::try_from(id) else {
                return Err(format!("token ID {id} at line {line_number} exceeds {VOCAB_SIZE}"));
            };
            if id == RESERVED_TOKEN_ID {
                return Err(format!("token ID 0 is reserved at line {line_number}"));
            }
            if let Some(previous) = previous_id {
                if id <= previous {
                    if id == previous {
                        return Err(format!("duplicate token ID {id} at line {line_number}"));
                    }
                    return Err(format!("token IDs out of ascending order at line {line_number}: {previous} then {id}"));
                }
            }
            previous_id = Some(id);
            let literal = &line[first + 1..last];
            let length_text = &line[last + 1..];
            if length_text.bytes().any(|byte| byte.is_ascii_whitespace()) {
                return Err(format!("bad token length at line {line_number}: {length_text:?}"));
            }
            let wanted: usize = digits(length_text).ok_or_else(|| format!("bad token length at line {line_number}: {length_text:?}"))?;
            parse_python_literal_into(literal, &mut bytes).ok_or_else(|| format!("bad literal at line {line_number}: {literal:?}"))?;
            if bytes.is_empty() { return Err(format!("empty token at line {line_number} for ID {id}")); }
            if bytes.len() != wanted { return Err(format!("length mismatch for ID {id} at line {line_number}")); }
            if bytes.len() > u8::MAX as usize {
                return Err(format!("token ID {id} at line {line_number} length {} exceeds u8 storage", bytes.len()));
            }
            let duplicate = || format!("duplicate token bytes at line {line_number} for ID {id}");
            if bytes.len() == 1 {
                if single[bytes[0] as usize] >= 0 { return Err(duplicate()); }
                single[bytes[0] as usize] = i32::from(id16);
            } else {
                let root = bytes[0] as usize * 256 + bytes[1] as usize;
                if trie_roots[root] == 0 {
                    trie_roots[root] = u32::try_from(nodes.len()).map_err(|_| "trie exceeds u32")?;
                    nodes.push(Node::default());
                }
                let mut node = trie_roots[root] as usize;
                for &byte in &bytes[2..] {
                    let mut child = nodes[node].head;
                    while child != 0 && nodes[child as usize].label != byte { child = nodes[child as usize].next; }
                    if child == 0 {
                        child = u32::try_from(nodes.len()).map_err(|_| "trie exceeds u32")?;
                        nodes.push(Node { label: byte, ..Node::default() });
                        if nodes[node].head == 0 { nodes[node].head = child; } else { let tail = nodes[node].tail as usize; nodes[tail].next = child; }
                        nodes[node].tail = child;
                        nodes[node].count += 1;
                    }
                    node = child as usize;
                }
                if nodes[node].term != 0 { return Err(duplicate()); }
                nodes[node].term = id16;
            }
            // tok_off[i] = blob length before ID i; IDs without a token repeat the offset.
            let offset = u32::try_from(tok_blob.len()).map_err(|_| "decode storage exceeds u32".to_owned())?;
            tok_off.resize(usize::from(id16) + 1, offset);
            tok_blob.extend_from_slice(&bytes);
            parsed += 1;
        }
        if parsed == 0 { return Err("vocabulary is empty".to_owned()); }
        let end = u32::try_from(tok_blob.len()).map_err(|_| "decode storage exceeds u32".to_owned())?;
        tok_off.resize(VOCAB_SIZE + 1, end);

        // A state packs a unique XOR base, terminal ID, and incoming byte.
        // Unique bases make the incoming byte a sufficient ownership check:
        // base_a ^ byte == base_b ^ byte implies base_a == base_b.
        let mut bases = vec![0usize; nodes.len()];
        // Occupancy bitset, 4 words per 256-slot block. Block zero is the all-missing
        // leaf table. Every node's slots share one block (base ^ label stays inside it).
        let mut used: Vec<u64> = vec![!0u64; 4];
        let mut used_bases = vec![true; 256];
        // Bit i of the result is bit i ^ d of `word`.
        fn xor_perm(mut word: u64, d: usize) -> u64 {
            const MASKS: [u64; 5] = [0x5555_5555_5555_5555, 0x3333_3333_3333_3333, 0x0f0f_0f0f_0f0f_0f0f, 0x00ff_00ff_00ff_00ff, 0x0000_ffff_0000_ffff];
            for (bit, mask) in MASKS.iter().enumerate() {
                if d >> bit & 1 != 0 { word = (word & mask) << (1 << bit) | (word >> (1 << bit)) & mask; }
            }
            if d & 32 != 0 { word = word.rotate_left(32); }
            word
        }
        let mut order: Vec<_> = (1..nodes.len()).filter(|&node| nodes[node].count != 0).collect();
        // Node index breaks ties: sort_unstable leaves equal keys in an unspecified,
        // std-version-dependent order, which would make the pinned layout fragile.
        order.sort_unstable_by_key(|&node| (std::cmp::Reverse(nodes[node].count), node));
        let mut first_free = 256usize;
        let mut cursor_of = std::collections::HashMap::<Vec<u8>, usize>::new();
        let mut cursors: Vec<usize> = Vec::new();
        let mut labels: Vec<u8> = Vec::new();
        for node in order {
            labels.clear();
            let mut child = nodes[node].head;
            while child != 0 { labels.push(nodes[child as usize].label); child = nodes[child as usize].next; }
            let pivot = labels[0] as usize;
            // Identical label patterns have identical placement constraints. Earlier
            // rejected slots stay invalid because occupied slots and used bases only grow.
            let cursor_index = match cursor_of.get(labels.as_slice()) {
                Some(&index) => index,
                None => { cursors.push(256); cursor_of.insert(labels.clone(), cursors.len() - 1); cursors.len() - 1 }
            };
            let start = first_free.max(cursors[cursor_index]);
            // Smallest free slot >= start whose base (slot ^ pivot) is unused and whose
            // every other label slot is free. 64 candidate slots are filtered per word.
            let (slot, base) = 'search: {
                let mut block = start >> 8;
                let mut from = start & 255;
                loop {
                    if block * 4 >= used.len() {
                        used.extend([0u64; 4]);
                        used_bases.resize(used_bases.len() + 256, false);
                    }
                    // Whole 256-slot block at once: the label loop is branch-predictable
                    // (one `d` per label) and exits as soon as no candidate survives.
                    let words = block * 4;
                    let free = [!used[words], !used[words + 1], !used[words + 2], !used[words + 3]];
                    let mut cands = free;
                    for cand in cands.iter_mut().take(from >> 6) { *cand = 0; }
                    cands[from >> 6] &= !0u64 << (from & 63);
                    for &label in &labels[1..] {
                        let d = pivot ^ label as usize;
                        let mut any = 0;
                        for (w, cand) in cands.iter_mut().enumerate() {
                            *cand &= xor_perm(free[w ^ (d >> 6 & 3)], d & 63);
                            any |= *cand;
                        }
                        if any == 0 { break; }
                    }
                    for (w, &word) in cands.iter().enumerate() {
                        let mut cand = word;
                        while cand != 0 {
                            let slot = block << 8 | w << 6 | cand.trailing_zeros() as usize;
                            cand &= cand - 1;
                            let base = slot ^ pivot;
                            if base >= 256 && !used_bases[base] { break 'search (slot, base); }
                        }
                    }
                    block += 1;
                    from = 0;
                }
            };
            u32::try_from(base).map_err(|_| "transition table exceeds u32")?;
            bases[node] = base;
            cursors[cursor_index] = slot + 1;
            used_bases[base] = true;
            for &label in &labels {
                let slot = base ^ label as usize;
                used[slot >> 6] |= 1 << (slot & 63);
            }
            while first_free >> 6 < used.len() {
                let free = !used[first_free >> 6] & (!0u64 << (first_free & 63));
                if free != 0 { first_free = first_free & !63 | free.trailing_zeros() as usize; break; }
                first_free = ((first_free >> 6) + 1) << 6;
            }
        }
        let state = |node: usize, label: u8| (bases[node] as u64) << 32 | u64::from(nodes[node].term) << 8 | u64::from(label);
        let mut transitions = vec![0u64; used.len() * 64];
        for (node, entry) in nodes.iter().enumerate() {
            let mut child = entry.head;
            while child != 0 {
                let label = nodes[child as usize].label;
                transitions[bases[node] ^ label as usize] = state(child as usize, label);
                child = nodes[child as usize].next;
            }
        }
        let trie_roots: Vec<u64> = trie_roots.into_iter().map(|node| if node==0 {0} else {state(node as usize,0)}).collect();

        if !partial {
            for (byte, id) in single.iter().enumerate() {
                if *id < 0 { return Err(format!("missing singleton byte 0x{byte:02x}")); }
            }
        }
        tok_blob.extend_from_slice(&[0; 8]); // decode_run reads whole 8-byte blocks
        tok_blob.shrink_to_fit();
        permit.finish(text.len());
        Ok(Self { data: Arc::new(TokenizerData { tok_blob, tok_off, single, trie_roots, transitions, partial }) })
    }

    /// Always [`VOCAB_SIZE`].
    pub fn vocab_size(&self) -> usize { VOCAB_SIZE }
    /// True if built by [`Self::from_vocab_str_partial`].
    pub fn is_partial(&self) -> bool { self.data.partial }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "state packs base in bits 32.., id in bits 8..24, label in bits 0..8, so each narrowing cast keeps exactly its field; `single` entries are checked >= 0; the match length is at most 255 because tokens are stored with u8 lengths"
    )]
    #[inline(always)]
    fn match_at(&self, src: &[u8], i: usize) -> (u32,u8) {
        let first = src[i] as usize;
        let mut best = (0u32,0u8);
        if i+1 < src.len() {
            let mut state = self.data.trie_roots[first*256 + src[i+1] as usize];
            if state != 0 {
                let id = (state >> 8) as u16;
                if id != 0 {
                    best = (u32::from(id),2);
                    if state >> 32 == 0 { return best; }
                }
                for (offset,&byte) in src[i+2..].iter().enumerate() {
                    let next = self.data.transitions[(state >> 32) as usize ^ byte as usize];
                    if next == 0 || next as u8 != byte { break; }
                    state = next;
                    let id = (state >> 8) as u16;
                    // Every later terminal is a longer match, regardless of ID.
                    if id != 0 { best = (u32::from(id),(offset+3) as u8); }
                    if state >> 32 == 0 { break; }
                }
            }
        }
        if best.0 != 0 {best}
        else if self.data.single[first]>=0 {(self.data.single[first] as u32,1)}
        else {(0,0)}
    }

    /// Permit-free inner loop: with no Drop in scope, `i` and the output
    /// length stay in registers. Returns the next offset or the failing one.
    #[inline(never)]
    fn match_run(&self, src: &[u8], mut i: usize, end: usize, out: &mut Vec<u32>) -> Result<usize, usize> {
        while i < end {
            // Match against the entire input: tokens can cross a quantum boundary.
            let (id, len) = self.match_at(src, i);
            if len == 0 { return Err(i); }
            out.push(id);
            i += len as usize;
        }
        Ok(i)
    }

    fn encode_sequential_bytes(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> {
        let mut result = Vec::with_capacity(src.len() / 3 + 8);
        let mut i = 0;
        while i < src.len() {
            let permit = cooperate::acquire_for(src.len());
            let end = i.saturating_add(permit.byte_budget()).min(src.len());
            let start = i;
            i = self.match_run(src, i, end, &mut result)
                .map_err(|at| EncodeError { byte_offset: at, byte: src[at] })?;
            permit.finish(i - start);
        }
        Ok(result)
    }

    fn copy_input(src: &[u8]) -> Vec<u8> {
        let mut copy = Vec::with_capacity(src.len());
        let mut offset = 0;
        while offset < src.len() {
            let permit = cooperate::acquire();
            let end = offset.saturating_add(permit.byte_budget()).min(src.len());
            copy.extend_from_slice(&src[offset..end]);
            permit.finish(end - offset);
            offset = end;
        }
        copy
    }

    /// Encode with an explicit worker count and parallel threshold. Zero or one
    /// workers, or input shorter than `min_bytes`, stay sequential; larger
    /// requests share the global pool and CPU budget.
    ///
    /// Returns the tokens and the effective matching job count for this call:
    /// one means sequential (including empty input), larger counts mean the
    /// parallel path. It is not a measurement of simultaneous OS threads.
    pub fn encode_with_policy(&self, src: &[u8], workers: usize, min_bytes: usize) -> Result<(Vec<u32>, usize), EncodeError> {
        self.encode_with_worker_limit(src, workers, min_bytes, cooperate::worker_limit())
    }

    fn encode_with_worker_limit(&self, src: &[u8], workers: usize, min_bytes: usize, worker_limit: usize) -> Result<(Vec<u32>, usize), EncodeError> {
        let workers = workers.min(worker_limit).min(src.len());
        if src.is_empty() || src.len() < min_bytes || workers <= 1 {
            return self.encode_sequential_bytes(src).map(|tokens| (tokens, 1));
        }
        let input = Arc::new(Self::copy_input(src));
        let chunk = src.len().div_ceil(workers);
        let (sender, receiver) = mpsc::channel();
        for start in (0..src.len()).step_by(chunk) {
            let input = Arc::clone(&input);
            let tokenizer = self.clone();
            let sender = sender.clone();
            cooperate::submit(move || {
                let end = (start + chunk).min(input.len());
                let mut matches = Vec::with_capacity(end - start);
                let mut i = start;
                while i < end {
                    let permit = cooperate::acquire();
                    let stop = i.saturating_add(permit.byte_budget()).min(end);
                    for offset in i..stop { matches.push(tokenizer.match_at(&input, offset)); }
                    permit.finish(stop - i);
                    i = stop;
                }
                #[expect(clippy::expect_used, reason = "the receiver outlives every sender: the caller drains it until all senders drop")]
                sender.send((start / chunk, matches)).expect("matching receiver closed");
            });
        }
        drop(sender);
        let mut parts = vec![None; src.len().div_ceil(chunk)];
        for (index, matches) in receiver { parts[index] = Some(matches); }
        #[expect(clippy::expect_used, reason = "every part is filled unless its worker panicked; panicking here propagates that failure")]
        let parts: Vec<_> = parts.into_iter().map(|part| part.expect("matching worker panicked")).collect();
        let mut result = Vec::with_capacity(src.len() / 3 + 8);
        let mut i = 0;
        while i < src.len() {
            let permit = cooperate::acquire();
            let stop = i.saturating_add(permit.byte_budget()).min(src.len());
            let start = i;
            while i < stop {
                let (id, len) = parts[i / chunk][i % chunk];
                if len == 0 { return Err(EncodeError { byte_offset: i, byte: src[i] }); }
                result.push(id);
                i += len as usize;
            }
            permit.finish(i - start);
        }
        Ok((result, src.len().div_ceil(chunk)))
    }

    /// Parallel matching at the current CPU-budget worker cap (sequential below
    /// [`PARALLEL_MIN_BYTES`]).
    pub fn encode_parallel(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> {
        self.encode_with_policy(src, cooperate::worker_limit(), PARALLEL_MIN_BYTES).map(|(tokens, _)| tokens)
    }

    /// Encode documents through a shared queue and preserve input order.
    /// Errors identify the first failing document and its byte offset.
    pub fn encode_batch(&self, texts: &[&str], workers: usize) -> Result<Vec<Vec<u32>>, BatchEncodeError> {
        self.encode_batch_with_worker_limit(texts, workers, cooperate::worker_limit())
    }

    fn encode_batch_with_worker_limit(&self, texts: &[&str], workers: usize, worker_limit: usize) -> Result<Vec<Vec<u32>>, BatchEncodeError> {
        let workers = workers.max(1).min(worker_limit).min(texts.len());
        if workers <= 1 {
            return texts.iter().enumerate().map(|(document_index, text)| {
                self.encode_sequential_bytes(text.as_bytes()).map_err(|source| BatchEncodeError { document_index, source })
            }).collect();
        }
        let inputs = Arc::new(texts.iter().map(|text| Self::copy_input(text.as_bytes())).collect::<Vec<_>>());
        let next = Arc::new(AtomicUsize::new(0));
        // Lowest failing document so far: later documents cannot change the answer.
        let first_bad = Arc::new(AtomicUsize::new(usize::MAX));
        let (sender, receiver) = mpsc::channel();
        for _ in 0..workers {
            let inputs = Arc::clone(&inputs);
            let next = Arc::clone(&next);
            let first_bad = Arc::clone(&first_bad);
            let tokenizer = self.clone();
            let sender = sender.clone();
            cooperate::submit(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(input) = inputs.get(index) else { break };
                    if index > first_bad.load(Ordering::Relaxed) { continue; }
                    #[cfg(test)]
                    tests::BATCH_ENCODED.fetch_add(1, Ordering::Relaxed);
                    let result = tokenizer.encode_sequential_bytes(input);
                    if result.is_err() { first_bad.fetch_min(index, Ordering::Relaxed); }
                    #[expect(clippy::expect_used, reason = "the receiver outlives every sender: the caller drains it until all senders drop")]
                    sender.send((index, result)).expect("batch receiver closed");
                }
            });
        }
        drop(sender);
        let mut output = vec![None; texts.len()];
        for (index, result) in receiver { output[index] = Some(result); }
        output.into_iter().enumerate().map(|(document_index, result)| {
            // Slots after the first failing document may be skipped; `collect` stops at that failure first.
            #[expect(clippy::expect_used, reason = "every slot up to the first failure is filled unless its worker panicked; panicking here propagates that failure")]
            let result = result.expect("batch worker panicked");
            result.map_err(|source| BatchEncodeError { document_index, source })
        }).collect()
    }

    /// Small batches stay sequential; larger batches share a bounded worker pool.
    pub fn encode_batch_auto(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, BatchEncodeError> {
        let total = texts.iter().fold(0usize, |sum, text| sum.saturating_add(text.len()));
        let workers = if total < BATCH_MIN_BYTES { 1 } else { total / BATCH_BYTES_PER_WORKER };
        self.encode_batch(texts, workers)
    }

    /// Encode text sequentially.
    pub fn encode(&self, src: &str) -> Result<Vec<u32>, EncodeError> { self.encode_bytes(src.as_bytes()) }

    /// Encode arbitrary bytes sequentially; errors only on a partial vocabulary.
    pub fn encode_bytes(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> {
        self.encode_sequential_bytes(src)
    }

    /// Concatenate token bytes; errors on an unknown ID.
    pub fn decode_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, DecodeError> {
        self.decode_then(tokens, |bytes| bytes)
    }

    /// Permit-free decode loop. Tokens of at most 8 bytes are copied as a fixed
    /// 8-byte block (one mov, no memcpy call; `tok_blob` has 8 bytes of padding)
    /// and the length is trimmed afterwards. Returns the next index or the failing one.
    #[inline(never)]
    fn decode_run(&self, tokens: &[u32], mut index: usize, stop: usize, out: &mut Vec<u8>) -> Result<usize, usize> {
        let data = &*self.data;
        while index < stop {
            let Some(id) = usize::try_from(tokens[index]).ok().filter(|&id| id < VOCAB_SIZE) else { return Err(index) };
            let (start, end) = (data.tok_off[id] as usize, data.tok_off[id + 1] as usize);
            if start == end { return Err(index); }
            let produced = out.len();
            if end - start <= 8 {
                let block: [u8; 8] = data.tok_blob[start..start + 8].try_into().unwrap_or([0; 8]);
                out.extend_from_slice(&block);
                out.truncate(produced + end - start);
            } else {
                out.extend_from_slice(&data.tok_blob[start..end]);
            }
            index += 1;
        }
        Ok(index)
    }

    /// Decode under the shared budget; `tail` runs inside the final permit,
    /// so UTF-8 conversion does not pay for a second gate pass.
    fn decode_then<R>(&self, tokens: &[u32], tail: impl FnOnce(Vec<u8>) -> R) -> Result<R, DecodeError> {
        let mut result = Vec::with_capacity(tokens.len() * 4);
        let mut index = 0usize;
        loop {
            let permit = cooperate::acquire_for(tokens.len().saturating_mul(4));
            let stop = index.saturating_add(permit.byte_budget() / 4).min(tokens.len());
            let produced = result.len();
            index = self.decode_run(tokens, index, stop, &mut result)
                .map_err(|at| DecodeError::MissingToken { token_index: at, token_id: tokens[at] })?;
            let decoded = result.len() - produced;
            if index == tokens.len() {
                let output = tail(result);
                permit.finish(decoded);
                return Ok(output);
            }
            permit.finish(decoded);
        }
    }

    /// Decode to text; errors on an unknown ID or invalid UTF-8.
    pub fn decode_utf8(&self, tokens: &[u32]) -> Result<String, DecodeError> {
        self.decode_then(tokens, String::from_utf8)?.map_err(DecodeError::InvalidUtf8)
    }

    /// Decode to text, replacing invalid UTF-8 with U+FFFD; errors only on an unknown ID.
    pub fn decode_lossy(&self, tokens: &[u32]) -> Result<String, DecodeError> {
        // Valid UTF-8 (the common case) is moved, not copied.
        self.decode_then(tokens, |bytes| String::from_utf8(bytes)
            .unwrap_or_else(|error| String::from_utf8_lossy(error.as_bytes()).into_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_vocabulary_hash_and_roundtrip() {
        use sha2::{Digest, Sha256};
        assert_eq!(
            format!("{:x}", Sha256::digest(include_bytes!("../vocab/rwkv_vocab_v20230424.txt"))),
            "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"
        );
        let tokenizer = RwkvTokenizer::bundled().unwrap();
        let bytes: Vec<u8> = (0..=255).collect();
        assert_eq!(tokenizer.decode_bytes(&tokenizer.encode_bytes(&bytes).unwrap()).unwrap(), bytes);
        let text = "Hello, 世界! 🦀\n";
        assert_eq!(tokenizer.decode_utf8(&tokenizer.encode(text).unwrap()).unwrap(), text);
        assert!(tokenizer.decode_bytes(&[RESERVED_TOKEN_ID]).is_err());
    }

    #[test]
    fn reviewer_counterexample_prefers_the_longest_token() {
        // Reported counterexample: 'abc' has a LOWER ID than its prefix 'ab',
        // so the old higher-ID proxy for "longest match" returned [2, 5].
        let tokenizer = RwkvTokenizer::from_vocab_str_partial(
            "1 b'abc' 3
2 b'ab' 2
3 b'a' 1
4 b'b' 1
5 b'c' 1
",
        )
        .unwrap();
        assert_eq!(tokenizer.encode_bytes(b"abc").unwrap(), vec![1]);
        assert_eq!(tokenizer.encode_bytes(b"ab").unwrap(), vec![2]);
        assert_eq!(tokenizer.encode_bytes(b"abx"), Err(EncodeError { byte_offset: 2, byte: b'x' }));
    }

    // The old higher-ID proxy was valid on the SHIPPED vocabulary and only wrong
    // for custom ones. That is a property of this file, so check it by machine
    // rather than asserting it in prose: no token may have a shorter prefix
    // token with a higher ID.
    #[test]
    fn bundled_vocabulary_never_gives_a_shorter_prefix_a_higher_id() {
        let text = include_str!("../vocab/rwkv_vocab_v20230424.txt");
        let mut tokens = Vec::new();
        for line in text.lines() {
            let first = line.find(' ').unwrap();
            let last = line.rfind(' ').unwrap();
            tokens.push((
                line[..first].parse::<u32>().unwrap(),
                parse_python_literal(&line[first + 1..last]).unwrap(),
            ));
        }
        assert_eq!(tokens.len(), 65_529);
        let ids: std::collections::HashMap<&[u8], u32> =
            tokens.iter().map(|(id, bytes)| (bytes.as_slice(), *id)).collect();
        let mut violations = Vec::new();
        for (id, bytes) in &tokens {
            for len in 1..bytes.len() {
                if let Some(prefix_id) = ids.get(&bytes[..len]) {
                    if prefix_id > id { violations.push((*prefix_id, *id)); }
                }
            }
        }
        assert_eq!(violations, Vec::<(u32, u32)>::new());
    }

    fn literal(bytes: &[u8]) -> String {
        let mut result = String::from("b'");
        for byte in bytes { result.push_str(&format!("\\x{byte:02x}")); }
        result.push('\'');
        result
    }

    // IDs 1..65529 are a valid sparse shape; IDs 65530..65535 and reserved 0
    // are intentionally absent. The first 256 IDs provide singleton coverage.
    fn valid_fixture() -> String {
        let mut result = String::new();
        for id in 1..=65_529u32 {
            let bytes = if id <= 256 { vec![u8::try_from(id - 1).unwrap()] } else {
                let pair = id - 257;
                vec![u8::try_from(pair & 255).unwrap(), u8::try_from(pair >> 8).unwrap()]
            };
            result.push_str(&format!("{id} {} {}\n", literal(&bytes), bytes.len()));
        }
        result
    }

    struct Reference { single: [Option<u32>; 256], buckets: Vec<Vec<(u32, Vec<u8>)>> }

    fn reference(text: &str) -> Reference {
        let mut tokens = Vec::new();
        for line in text.lines() {
            let first = line.find(' ').unwrap();
            let last = line.rfind(' ').unwrap();
            tokens.push((line[..first].parse::<u32>().unwrap(), parse_python_literal(&line[first + 1..last]).unwrap()));
        }
        tokens.sort_by_key(|(_, bytes)| std::cmp::Reverse(bytes.len()));
        let mut single = [None; 256];
        let mut buckets = vec![Vec::new(); 256 * 256];
        for (id, bytes) in tokens {
            if bytes.len() == 1 { single[bytes[0] as usize] = Some(id); }
            else { buckets[bytes[0] as usize * 256 + bytes[1] as usize].push((id, bytes)); }
        }
        Reference { single, buckets }
    }

    fn reference_encode(reference: &Reference, src: &[u8]) -> Vec<u32> {
        let mut result = Vec::new();
        let mut i = 0;
        while i < src.len() {
            let mut hit = None;
            if i + 1 < src.len() {
                for (id, token) in &reference.buckets[src[i] as usize * 256 + src[i + 1] as usize] {
                    if src[i..].starts_with(token) { hit = Some((*id, token.len())); break; }
                }
            }
            if let Some((id, len)) = hit { result.push(id); i += len; }
            else if let Some(id) = reference.single[src[i] as usize] { result.push(id); i += 1; }
            else { i += 1; }
        }
        result
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        assert!(parse_python_literal(r"'\").is_none());
        assert!(parse_python_literal(r"'\q'").is_none());
        assert_eq!(parse_python_literal("'ä'"), Some("ä".as_bytes().to_vec()));
        assert_eq!(parse_python_literal(r"b'\xe4'"), Some(vec![0xe4]));
        assert!(parse_python_literal("b'ä'").is_none());
        for (text, expected) in [
            ("x b'a' 1", "bad token ID"), ("1 b'' 0", "empty token"),
            ("65536 b'a' 1", "exceeds"), ("1 b'a' 2", "length mismatch"),
            ("1 b'\\q' 1", "bad literal"), ("1 b'a' 1\n1 b'b' 1", "duplicate"),
            ("2 b'a' 1\n1 b'b' 1", "ascending order"), ("0 b'a' 1", "reserved"),
            ("1 b'a' 1\n2 b'a' 1", "duplicate token bytes"),
        ] {
            let error = RwkvTokenizer::from_vocab_str_partial(text).unwrap_err().to_string();
            assert!(error.contains(expected), "{text:?}: {error}");
        }
        let long = vec![b'a'; 256];
        let error = RwkvTokenizer::from_vocab_str_partial(&format!("1 {} {}", literal(&long), long.len())).unwrap_err().to_string();
        assert!(error.contains("u8 storage"), "{error}");
    }

    #[test]
    fn sparse_valid_fixture_has_singleton_coverage_and_legacy_parity() {
        let text = valid_fixture();
        let tokenizer = RwkvTokenizer::from_vocab_str(&text).unwrap();
        let reference = reference(&text);
        for input in [b"hello world".as_slice(), b"\xe4\xb8\xad\xe6\x96\x87\n\x00\xff", b"abcabc"] {
            assert_eq!(tokenizer.encode_bytes(input).unwrap(), reference_encode(&reference, input));
        }
        assert!(!tokenizer.is_partial());
        assert_eq!(tokenizer.decode_bytes(&tokenizer.encode_bytes(b"hello").unwrap()).unwrap(), b"hello");
    }

    #[test]
    fn partial_mode_reports_missing_encode_bytes_and_decode_ids() {
        let text = "1 b'a' 1\n2 b'b' 1\n";
        let tokenizer = RwkvTokenizer::from_vocab_str_partial(text).unwrap();
        assert!(tokenizer.is_partial());
        assert_eq!(tokenizer.encode_bytes(b"ab"), Ok(vec![1, 2]));
        for result in [
            tokenizer.encode("ac"),
            tokenizer.encode_bytes(b"ac"),
            tokenizer.encode_parallel(b"ac"),
            tokenizer.encode_with_policy(b"ac", 2, 0).map(|(tokens, _)| tokens),
        ] {
            assert_eq!(result, Err(EncodeError { byte_offset: 1, byte: b'c' }));
        }
        assert!(matches!(tokenizer.decode_bytes(&[3]), Err(DecodeError::MissingToken { token_index: 0, token_id: 3 })));
        assert!(matches!(tokenizer.decode_utf8(&[3]), Err(DecodeError::MissingToken { token_index: 0, token_id: 3 })));
        assert!(matches!(tokenizer.decode_lossy(&[3]), Err(DecodeError::MissingToken { token_index: 0, token_id: 3 })));
    }

    #[test]
    fn forced_parallel_path_reports_missing_byte_at_exact_offset() {
        let tokenizer = RwkvTokenizer::from_vocab_str_partial("1 b'a' 1\n").unwrap();
        let mut input = vec![b'a'; PARALLEL_MIN_BYTES];
        input.push(b'c');
        assert_eq!(
            tokenizer.encode_with_worker_limit(&input, 2, PARALLEL_MIN_BYTES, 2),
            Err(EncodeError {
                byte_offset: PARALLEL_MIN_BYTES,
                byte: b'c'
            })
        );
    }

    #[test]
    fn grouped_pair_and_triple_candidates_match_bucket_oracle() {
        let mut text = String::new();
        for id in 1..=256u32 {
            text.push_str(&format!("{id} {} 1\n", literal(&[u8::try_from(id - 1).unwrap()])));
        }
        // The shorter pair has the higher ID, but the longer triple wins.
        // The second triple exercises a group
        // with no pair candidate.
        text.push_str(&format!("300 {} 3\n", literal(b"xyz")));
        text.push_str(&format!("301 {} 2\n", literal(b"xy")));
        text.push_str(&format!("302 {} 3\n", literal(b"mno")));
        let tokenizer = RwkvTokenizer::from_vocab_str(&text).unwrap();
        let reference = reference(&text);
        let input = b"xyzmno";
        assert_eq!(
            tokenizer.encode_bytes(input).unwrap(),
            reference_encode(&reference, input)
        );
        assert_eq!(tokenizer.encode_bytes(input).unwrap(), vec![300, 302]);
    }

    #[test]
    fn longest_match_ignores_ids_across_encoding_paths() {
        // Deeper terminals have smaller IDs; abcd is a nonterminal prefix.
        let tokenizer = RwkvTokenizer::from_vocab_str_partial(
            "1 b'abcde' 5\n2 b'abc' 3\n3 b'ab' 2\n4 b'a' 1\n"
        ).unwrap();
        for (input, expected) in [(b"abcdeabcaba".as_slice(), vec![1, 2, 3, 4]), (b"abc", vec![2])] {
            assert_eq!(tokenizer.encode_bytes(input).unwrap(), expected);
            let (parallel, workers) = tokenizer.encode_with_worker_limit(input, 2, 0, 2).unwrap();
            assert_eq!(workers, 2);
            assert_eq!(parallel, expected);
            assert_eq!(tokenizer.decode_bytes(&expected).unwrap(), input);
        }
        for count in [1, 2, 6, 32] {
            assert_eq!(tokenizer.encode_batch(&vec!["abcdeabcaba"; count], 2).unwrap(), vec![vec![1, 2, 3, 4]; count]);
        }
        let missing = Err(EncodeError { byte_offset: 3, byte: b'd' });
        assert_eq!(tokenizer.encode_bytes(b"abcdx"), missing);
        assert_eq!(tokenizer.encode_with_worker_limit(b"abcdx", 2, 0, 2).map(|(tokens, _)| tokens), missing);
    }

    #[test]
    fn forced_sequential_and_parallel_paths_match() {
        let tokenizer = RwkvTokenizer::from_vocab_str(&valid_fixture()).unwrap();
        let seed = b"The quick brown fox \xe4\xb8\xad\xe6\x96\x87\n";
        let mut input = Vec::with_capacity(PARALLEL_MIN_BYTES + seed.len());
        while input.len() < PARALLEL_MIN_BYTES { input.extend_from_slice(seed); }
        let sequential = tokenizer.encode_bytes(&input).unwrap();
        // Exercise matching jobs on one-worker hosts without changing the CPU gate.
        let before = cooperation_stats();
        let (parallel, workers) = tokenizer.encode_with_worker_limit(&input, 2, PARALLEL_MIN_BYTES, 2).unwrap();
        let after = cooperation_stats();
        assert_eq!(workers, 2);
        assert!(after.submitted_jobs + after.inline_fallbacks >= before.submitted_jobs + before.inline_fallbacks + 2);
        assert_eq!(sequential, parallel);
    }
    #[test]
    fn effective_policy_reports_caps_thresholds_and_job_count() {
        let tokenizer = RwkvTokenizer::from_vocab_str_partial("1 b'a' 1\n").unwrap();
        for (len, requested, threshold, cap, expected) in [
            (0, 8, 0, 8, 1), (1, 8, 0, 8, 1),
            (10, 0, 0, 8, 1), (10, 1, 0, 8, 1),
            (10, 8, 11, 8, 1), (10, 8, 10, 1, 1),
            (10, 8, 10, 2, 2), (10, 8, 10, 8, 5),
        ] {
            let input = vec![b'a'; len];
            let (tokens, workers) = tokenizer.encode_with_worker_limit(&input, requested, threshold, cap).unwrap();
            assert_eq!(workers, expected);
            assert_eq!(tokens, vec![1; len]);
        }
    }
    #[test]
    fn direct_transitions_cover_vocabulary_and_reject_foreign_edges() {
        let tokenizer=RwkvTokenizer::bundled().unwrap();
        assert!(tokenizer.data.transitions[..256].iter().all(|&state|state==0));
        for state in tokenizer.data.trie_roots.iter().chain(&tokenizer.data.transitions) {
            assert!(((*state >> 32) as usize | 255) < tokenizer.data.transitions.len());
        }
        let mut count=0;
        for id in 1..VOCAB_SIZE {
            let start=tokenizer.data.tok_off[id] as usize;
            let end=tokenizer.data.tok_off[id+1] as usize;
            if start==end {continue;}
            let token=&tokenizer.data.tok_blob[start..end];
            assert_eq!(tokenizer.match_at(token,0),(u32::try_from(id).unwrap(),u8::try_from(token.len()).unwrap()),"token {id}");
            count+=1;
        }
        assert_eq!(count,65529);
        use sha2::{Digest, Sha256};
        let mut layout = Sha256::new();
        for state in tokenizer.data.trie_roots.iter().chain(&tokenizer.data.transitions) { layout.update(state.to_le_bytes()); }
        // Pinned: constructor changes must reproduce the exact packed layout.
        assert_eq!(format!("{:x}", layout.finalize()), "780121d33451dae07f03456b1cedd32463352383e23b9ad0293298a96fdd2f24");
        let mut vocabulary=String::new();
        for (id,token) in [b"abx".as_slice(),b"cdy",b"efz",b"abxy",b"cdxy",b"efxy"].iter().enumerate() {
            vocabulary.push_str(&format!("{} {} {}\n",id+1,literal(token),token.len()));
        }
        let sparse=RwkvTokenizer::from_vocab_str_partial(&vocabulary).unwrap();
        for prefix in [b"ab",b"cd",b"ef"] {
            for byte in 0..=255u8 {
                let input=[prefix[0],prefix[1],byte];
                let expected=if prefix==b"ab" && byte==b'x' {1} else if prefix==b"cd" && byte==b'y' {2} else if prefix==b"ef" && byte==b'z' {3} else {0};
                assert_eq!(sparse.match_at(&input,0).0,expected);
            }
        }
    }
    #[test]
    fn decode_run_matches_token_table_for_every_id() {
        // About 65k tokens: many decode quanta, and the last blob entry reads padding.
        let tokenizer = RwkvTokenizer::bundled().unwrap();
        let data = &tokenizer.data;
        let span = |id: usize| data.tok_off[id] as usize..data.tok_off[id + 1] as usize;
        let ids: Vec<u32> = (1..VOCAB_SIZE).filter(|&id| !span(id).is_empty()).map(|id| u32::try_from(id).unwrap()).collect();
        let expected: Vec<u8> = (1..VOCAB_SIZE).flat_map(|id| data.tok_blob[span(id)].iter().copied()).collect();
        assert_eq!(tokenizer.decode_bytes(&ids).unwrap(), expected);
    }
    pub(super) static BATCH_ENCODED: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn batch_stops_after_the_first_failing_document() {
        let tokenizer = RwkvTokenizer::from_vocab_str_partial("1 b'a' 1
").unwrap();
        let big = "a".repeat(1 << 18);
        let mut documents = vec!["a", "ab", "b"];
        documents.extend(std::iter::repeat(big.as_str()).take(200));
        let before = BATCH_ENCODED.load(Ordering::Relaxed);
        let error = tokenizer.encode_batch_with_worker_limit(&documents, 2, 2).unwrap_err();
        assert_eq!(error, BatchEncodeError { document_index: 1, source: EncodeError { byte_offset: 1, byte: b'b' } });
        // Positive control: without the abort all 203 documents are encoded. Other
        // batch tests share the counter, hence the generous bound.
        assert!(BATCH_ENCODED.load(Ordering::Relaxed) - before < 100);
        let ok = vec![big.as_str(); 8];
        assert_eq!(tokenizer.encode_batch_with_worker_limit(&ok, 2, 2).unwrap().len(), 8);
    }

    #[test]
    fn parser_rejects_signs_and_non_hex_escapes() {
        for (vocab, message) in [
            ("+1 b'a' 1\n", "bad token ID at line 1"),
            ("1 b'a' +1\n", "bad token length at line 1"),
            ("1 b'\\x+a' 1\n", "bad literal at line 1"),
            ("1 '\\u+041' 1\n", "bad literal at line 1"),
        ] {
            let error = RwkvTokenizer::from_vocab_str_partial(vocab).unwrap_err().to_string();
            assert!(error.starts_with(message), "{vocab:?}: {error}");
        }
        assert_eq!(parse_python_literal("b'\\x0a'"), Some(vec![b'\n']));
        assert_eq!(parse_python_literal("'\\u00e9'"), Some("\u{e9}".as_bytes().to_vec()));
    }
    #[test]
    fn explicit_policies_preserve_empty_input_and_greedy_error_offsets() {
        // There is no singleton for b: only actual token starts may raise errors.
        let tokenizer = RwkvTokenizer::from_vocab_str_partial("1 b'a' 1\n2 b'ab' 2\n").unwrap();
        let valid = b"ab".repeat(8192);
        let mut invalid = valid.clone(); invalid.push(b'c');
        for workers in [0,1,2,4,8,16,usize::MAX] {
            for threshold in [0,16384,131072,usize::MAX] {
                let tokens = |input: &[u8]| tokenizer.encode_with_policy(input,workers,threshold).map(|(tokens, _)| tokens);
                assert_eq!(tokens(b""),Ok(vec![]));
                assert_eq!(tokens(&valid),tokenizer.encode_bytes(&valid));
                assert_eq!(tokens(&invalid),
                    Err(EncodeError { byte_offset: valid.len(), byte: b'c' }));
            }
        }
    }
    #[test]
    fn batch_preserves_document_identity_and_first_error() {
        let tokenizer = RwkvTokenizer::from_vocab_str_partial("1 b'a' 1\n2 b'b' 1\n").unwrap();
        for size in [1,2,6,32] {
            let documents: Vec<&str> = ["", "a", "b", "abba"].into_iter().cycle().take(size).collect();
            let expected: Vec<Vec<u32>> = documents.iter().map(|text| tokenizer.encode(text).unwrap()).collect();
            for workers in [0,1,2,4,8,16,usize::MAX] {
                assert_eq!(tokenizer.encode_batch(&documents,workers).unwrap(),expected);
                assert_eq!(tokenizer.encode_batch(&vec!["abba";size],workers).unwrap(),vec![vec![1,2,2,1];size]);
            }
        }
        assert!(tokenizer.encode_batch(&[],16).unwrap().is_empty());
        for workers in [0,1,2,4,8,16] {
            let error = tokenizer.encode_batch(&["a", "abx", "y", "bb"],workers).unwrap_err();
            assert_eq!(error, BatchEncodeError { document_index: 1, source: EncodeError { byte_offset: 2, byte: b'x' } });
        }
        let expected = vec![vec![1],vec![2]];
        let mut permuted = tokenizer.encode_batch(&["a","b"],2).unwrap();
        permuted.swap(0,1);
        assert_ne!(permuted,expected,"positive control must detect swapped documents");
    }

    #[test]
    fn cooperative_boundaries_preserve_maximum_length_tokens() {
        let token = vec![b'a'; 255];
        let vocab = format!("7 {} 255\n", literal(&token));
        let tokenizer = RwkvTokenizer::from_vocab_str_partial(&vocab).unwrap();
        let mut input = token.repeat(4097);
        let before = cooperation_stats();
        let encoded = tokenizer.encode_bytes(&input).unwrap();
        assert_eq!(encoded, vec![7; 4097]);
        assert!(cooperation_stats().completed_quanta > before.completed_quanta + 1);
        assert_eq!(tokenizer.decode_bytes(&encoded).unwrap(), input);
        let offset = input.len();
        input.push(b'b');
        assert_eq!(tokenizer.encode_bytes(&input), Err(EncodeError { byte_offset: offset, byte: b'b' }));
    }

    #[test]
    fn shared_budget_is_used_by_public_paths() {
        let tokenizer = RwkvTokenizer::bundled().unwrap();
        let large = "hello world ".repeat(32768);
        let before = cooperation_stats();
        let expected = tokenizer.encode(&large).unwrap();
        assert_eq!(tokenizer.encode_with_policy(large.as_bytes(), 8, 0).unwrap().0, expected);
        for count in [1,2,6,32] {
            assert_eq!(tokenizer.encode_batch_auto(&vec![large.as_str(); count]).unwrap(), vec![expected.clone(); count]);
        }
        let after = cooperation_stats();
        assert!(after.completed_quanta > before.completed_quanta);
        assert!(after.yields > before.yields);
        // The CPU-budget cap never exceeds 8 workers.
        assert!(after.peak_active <= 8);
        assert!(after.pool_threads <= 8);
        if after.worker_limit > 1 { assert!(after.submitted_jobs > before.submitted_jobs); }
    }

    #[test]
    fn hybrid_degree_boundaries_preserve_longest_match_and_missing_offsets() {
        for count in [15u32,16,256] {
            for shorter_has_higher_id in [false,true] {
                let mut vocab=String::new();
                for byte in 0..256u32 { vocab.push_str(&format!("{} {} 1\n",byte+1,literal(&[u8::try_from(byte).unwrap()]))); }
                if !shorter_has_higher_id { vocab.push_str("299 b'ab' 2\n"); }
                let mut partial=String::new();
                if !shorter_has_higher_id { partial.push_str("299 b'ab' 2\n"); }
                for byte in 0..count {
                    let line=format!("{} {} 3\n",300+byte,literal(&[b'a',b'b',u8::try_from(byte).unwrap()]));
                    vocab.push_str(&line);partial.push_str(&line);
                }
                if shorter_has_higher_id { vocab.push_str("600 b'ab' 2\n");partial.push_str("600 b'ab' 2\n"); }
                let full=RwkvTokenizer::from_vocab_str(&vocab).unwrap();
                let partial=RwkvTokenizer::from_vocab_str_partial(&partial).unwrap();
                for byte in 0..count {
                    let input=[b'a',b'b',u8::try_from(byte).unwrap()];
                    assert_eq!(full.encode_bytes(&input).unwrap(),vec![300+byte]);
                    assert_eq!(partial.encode_bytes(&input),Ok(vec![300+byte]));
                }
                if count<256 {
                    let input=[b'a',b'b',255];
                    assert_eq!(partial.encode_bytes(&input),Err(EncodeError { byte_offset:2,byte:255 }));
                }
            }
        }
    }

    #[test]
    fn from_file_loads_a_vocabulary_and_reports_a_missing_file() {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("rwkv_tokenizer_test_{}_{nanos}.txt", std::process::id()));
        std::fs::write(&path, include_str!("../vocab/rwkv_vocab_v20230424.txt")).unwrap();
        let loaded = RwkvTokenizer::from_file(&path);
        std::fs::remove_file(&path).unwrap();
        let bundled = RwkvTokenizer::bundled().unwrap();
        assert_eq!(loaded.unwrap().encode("Hello 世界").unwrap(), bundled.encode("Hello 世界").unwrap());
        let error = RwkvTokenizer::from_file(&path).unwrap_err().to_string();
        assert!(error.contains("rwkv_tokenizer_test_"), "{error}");
    }

    #[test]
    fn decode_utf8_and_lossy_handle_invalid_utf8() {
        let tokenizer = RwkvTokenizer::bundled().unwrap();
        let lone = tokenizer.encode_bytes(&[0x80]).unwrap();
        assert!(matches!(tokenizer.decode_utf8(&lone), Err(DecodeError::InvalidUtf8(_))));
        assert_eq!(tokenizer.decode_lossy(&lone).unwrap(), "\u{fffd}");
        let mut mixed = tokenizer.encode("ok ").unwrap();
        mixed.extend(&lone);
        assert_eq!(tokenizer.decode_lossy(&mixed).unwrap(), "ok \u{fffd}");
        let valid = tokenizer.encode("Hello, 世界! 🦀").unwrap();
        assert_eq!(tokenizer.decode_lossy(&valid).unwrap(), "Hello, 世界! 🦀");
    }

    #[test]
    fn decode_reports_out_of_range_ids_with_their_index() {
        let tokenizer = RwkvTokenizer::bundled().unwrap();
        let good = tokenizer.encode("a").unwrap()[0];
        for bad in [65_536u32, u32::MAX] {
            assert!(matches!(tokenizer.decode_bytes(&[good, good, bad]), Err(DecodeError::MissingToken { token_index: 2, token_id }) if token_id == bad));
            assert!(matches!(tokenizer.decode_utf8(&[bad]), Err(DecodeError::MissingToken { token_index: 0, token_id }) if token_id == bad));
        }
    }

    #[test]
    fn error_display_and_source() {
        use std::error::Error;
        let encode = EncodeError { byte_offset: 7, byte: 0x63 };
        assert_eq!(encode.to_string(), "no token for byte 0x63 at byte offset 7");
        let batch = BatchEncodeError { document_index: 2, source: encode.clone() };
        assert_eq!(batch.to_string(), "document 2: no token for byte 0x63 at byte offset 7");
        assert_eq!(batch.source().unwrap().to_string(), encode.to_string());
        assert!(encode.source().is_none());
        let missing = DecodeError::MissingToken { token_index: 4, token_id: 99 };
        assert_eq!(missing.to_string(), "missing token ID 99 at token index 4");
        let utf8 = String::from_utf8(vec![0xff]).unwrap_err();
        let text = DecodeError::InvalidUtf8(utf8).to_string();
        assert!(text.starts_with("decoded bytes are not UTF-8: "), "{text}");
    }

    #[test]
    fn vocab_and_decode_error_traits() {
        use std::error::Error;
        let io = RwkvTokenizer::from_file(Path::new("definitely/missing/vocab.txt")).unwrap_err();
        assert!(matches!(io, VocabError::Io { .. }));
        assert!(io.source().is_some());
        let invalid = RwkvTokenizer::from_vocab_str("").unwrap_err();
        assert_eq!(invalid.to_string(), "vocabulary is empty");
        assert!(invalid.source().is_none());
        let tokenizer = RwkvTokenizer::bundled().unwrap();
        let utf8 = tokenizer.decode_utf8(&tokenizer.encode_bytes(&[0x80]).unwrap()).unwrap_err();
        assert!(utf8.source().is_some());
        assert_eq!(utf8.clone(), utf8);
        assert!(DecodeError::MissingToken { token_index: 0, token_id: 0 }.source().is_none());
    }

    #[test]
    fn batch_auto_small_batch_preserves_order_and_matches_encode() {
        let tokenizer = RwkvTokenizer::bundled().unwrap();
        let documents = ["", "Hello", "世界", "🦀 crab", "Hello"];
        let batch = tokenizer.encode_batch_auto(&documents).unwrap();
        let expected: Vec<Vec<u32>> = documents.iter().map(|text| tokenizer.encode(text).unwrap()).collect();
        assert_eq!(batch, expected);
    }

    #[test]
    fn vocabulary_errors_blank_separator_singletons_empty_and_crlf() {
        for (text, expected) in [
            ("1 b'a' 1\n\n2 b'b' 1\n", "malformed vocabulary line 2: blank line"),
            ("1b'a'1\n", "malformed vocabulary line 1: missing separator"),
            ("", "vocabulary is empty"),
        ] {
            let error = RwkvTokenizer::from_vocab_str_partial(text).unwrap_err().to_string();
            assert!(error.contains(expected), "{text:?}: {error}");
        }
        let error = RwkvTokenizer::from_vocab_str("1 b'a' 1\n").unwrap_err().to_string();
        assert!(error.contains("missing singleton byte 0x00"), "{error}");
        let lf = "1 b'a' 1\n2 b'b' 1\n3 b'ab' 2\n";
        let crlf = lf.replace('\n', "\r\n");
        let (lf, crlf) = (RwkvTokenizer::from_vocab_str_partial(lf).unwrap(), RwkvTokenizer::from_vocab_str_partial(&crlf).unwrap());
        assert_eq!(lf.encode("abba").unwrap(), crlf.encode("abba").unwrap());
        assert_eq!(crlf.encode("abba").unwrap(), vec![3, 2, 1]);
    }
}
