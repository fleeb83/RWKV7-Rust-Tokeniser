#![warn(clippy::undocumented_unsafe_blocks)]

//! Standalone RWKV World tokenizer.

use std::{fmt, path::Path, sync::{Arc, mpsc, atomic::{AtomicUsize, Ordering}}};

mod cooperate;
pub use cooperate::Stats as CooperationStats;
/// Process-wide activity for this crate. Peaks and counters are cumulative.
pub fn cooperation_stats() -> CooperationStats { cooperate::stats() }

pub const VOCAB_SIZE: usize = 65_536;
pub const RESERVED_TOKEN_ID: u32 = 0;
pub const PARALLEL_MIN_BYTES: usize = 1 << 17;
pub const PARALLEL_MAX_WORKERS: usize = 8;
pub const BATCH_MIN_BYTES: usize = 1 << 16;
pub const BATCH_BYTES_PER_WORKER: usize = 1 << 15;


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeError {
    pub byte_offset: usize,
    pub byte: u8,
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no token for byte 0x{:02x} at byte offset {}", self.byte, self.byte_offset)
    }
}
impl std::error::Error for EncodeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchEncodeError {
    pub document_index: usize,
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

#[derive(Debug)]
pub enum DecodeError {
    MissingToken { token_index: usize, token_id: u32 },
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
impl std::error::Error for DecodeError {}

/// Parse the Python-style literal subset used by `rwkv_vocab_v20230424.txt`.
pub fn parse_python_literal_pub(s: &str) -> Option<Vec<u8>> {
    parse_python_literal(s)
}

fn parse_python_literal(s: &str) -> Option<Vec<u8>> {
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
    let mut result = Vec::with_capacity(inner.len());
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
                let hex = std::str::from_utf8(&inner[i + 2..i + 4]).ok()?;
                let value = u8::from_str_radix(hex, 16).ok()?;
                if is_bytes {
                    result.push(value);
                } else {
                    result.extend_from_slice(char::from_u32(value as u32)?.encode_utf8(&mut utf8).as_bytes());
                }
                i += 4;
                continue;
            }
            b'u' => {
                if is_bytes || i + 6 > inner.len() { return None; }
                let hex = std::str::from_utf8(&inner[i + 2..i + 6]).ok()?;
                let value = u32::from_str_radix(hex, 16).ok()?;
                result.extend_from_slice(char::from_u32(value)?.encode_utf8(&mut utf8).as_bytes());
                i += 6;
                continue;
            }
            _ => return None,
        }
        i += 2;
    }
    Some(result)
}

#[derive(Debug, Clone)]
pub struct RwkvTokenizer { data: Arc<TokenizerData> }

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
    pub fn bundled() -> Result<Self, String> {
        Self::from_vocab_str(include_str!("../vocab/rwkv_vocab_v20230424.txt"))
    }

    pub fn from_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|error| format!("{path:?}: {error}"))?;
        Self::from_vocab_str(&text)
    }

    /// Load a normal vocabulary and require every byte to have a singleton.
    pub fn from_vocab_str(text: &str) -> Result<Self, String> {
        Self::from_vocab_str_mode(text, false)
    }

    /// Explicit partial-vocabulary mode. Missing IDs and singleton bytes are
    /// allowed, but checked encoding reports the first missing byte.
    pub fn from_vocab_str_partial(text: &str) -> Result<Self, String> {
        Self::from_vocab_str_mode(text, true)
    }

    fn from_vocab_str_mode(text: &str, partial: bool) -> Result<Self, String> {
        let permit = cooperate::acquire();
        let mut idx2tok: Vec<Option<Vec<u8>>> = vec![None; VOCAB_SIZE];
        let mut seen = std::collections::HashSet::new();
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
            let id: u32 = id_text.parse().map_err(|_| format!("bad token ID at line {line_number}: {id_text:?}"))?;
            if id as usize >= VOCAB_SIZE {
                return Err(format!("token ID {id} at line {line_number} exceeds {VOCAB_SIZE}"));
            }
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
            let wanted: usize = length_text.parse().map_err(|_| format!("bad token length at line {line_number}: {length_text:?}"))?;
            let bytes = parse_python_literal(literal).ok_or_else(|| format!("bad literal at line {line_number}: {literal:?}"))?;
            if bytes.is_empty() { return Err(format!("empty token at line {line_number} for ID {id}")); }
            if bytes.len() != wanted { return Err(format!("length mismatch for ID {id} at line {line_number}")); }
            if bytes.len() > u8::MAX as usize {
                return Err(format!("token ID {id} at line {line_number} length {} exceeds u8 storage", bytes.len()));
            }
            if !seen.insert(bytes.clone()) { return Err(format!("duplicate token bytes at line {line_number} for ID {id}")); }
            idx2tok[id as usize] = Some(bytes);
            parsed += 1;
        }
        if parsed == 0 { return Err("vocabulary is empty".to_owned()); }

        let mut single = [-1i32; 256];
        let mut tok_blob = Vec::new();
        let mut tok_off = Vec::with_capacity(VOCAB_SIZE + 1);
        for token in &idx2tok {
            tok_off.push(u32::try_from(tok_blob.len()).map_err(|_| "decode storage exceeds u32".to_owned())?);
            if let Some(token) = token {
                let new_len = tok_blob.len().checked_add(token.len()).ok_or_else(|| "decode storage overflow".to_owned())?;
                if new_len > u32::MAX as usize { return Err("decode storage exceeds u32".to_owned()); }
                tok_blob.extend_from_slice(token);
            }
        }
        tok_off.push(u32::try_from(tok_blob.len()).map_err(|_| "decode storage exceeds u32".to_owned())?);

        // Direct two-byte roots avoid searching the two highest-fanout levels.
        // Node zero is the missing-root sentinel; token zero is never terminal.
        let mut trie_roots = vec![0u32; 256 * 256];
        let mut build: Vec<(u16, Vec<(u8, u32)>)> = vec![(0, Vec::new())];
        for (id, token) in idx2tok.iter().enumerate() {
            let Some(token) = token.as_deref() else { continue };
            if token.len() == 1 {
                single[token[0] as usize] = id as i32;
                continue;
            }
            let root = token[0] as usize * 256 + token[1] as usize;
            if trie_roots[root] == 0 {
                trie_roots[root] = u32::try_from(build.len()).map_err(|_| "trie exceeds u32")?;
                build.push((0, Vec::new()));
            }
            let mut node = trie_roots[root] as usize;
            for &byte in &token[2..] {
                // ponytail: construction scans at most 256 siblings; add an index if load time dominates.
                let found = build[node].1.iter().find(|edge| edge.0 == byte).map(|edge| edge.1);
                let child = if let Some(child) = found { child } else {
                    let child = u32::try_from(build.len()).map_err(|_| "trie exceeds u32")?;
                    build.push((0, Vec::new()));
                    build[node].1.push((byte, child));
                    child
                };
                node = child as usize;
            }
            // Vocabulary loading already rejects IDs outside 1..65535.
            build[node].0 = id as u16;
        }
        // A state packs a unique XOR base, terminal ID, and incoming byte.
        // Unique bases make the incoming byte a sufficient ownership check:
        // base_a ^ byte == base_b ^ byte implies base_a == base_b.
        let mut bases = vec![0usize; build.len()];
        let mut used = vec![true; 256]; // base zero is the all-missing leaf table
        let mut used_bases = vec![true; 256];
        // Successor links skip occupied slots instead of repeatedly scanning them.
        fn next_free(links: &mut [usize], mut slot: usize) -> usize {
            let mut root = slot;
            while links[root] != root { root = links[root]; }
            while links[slot] != slot {
                let next = links[slot]; links[slot] = root; slot = next;
            }
            root
        }
        let mut free_links: Vec<usize> = (0..=256).collect();
        let mut order: Vec<_> = (1..build.len()).filter(|&node| !build[node].1.is_empty()).collect();
        order.sort_unstable_by_key(|&node| std::cmp::Reverse(build[node].1.len()));
        let mut first_free = 256usize;
        let mut search_cursors = std::collections::HashMap::<Vec<u8>, usize>::new();
        for node in order {
            let edges = &build[node].1;
            let pivot = edges[0].0 as usize;
            // Identical label patterns have identical placement constraints. Earlier
            // rejected slots stay invalid because occupied slots and used bases only grow.
            let labels: Vec<u8> = edges.iter().map(|edge| edge.0).collect();
            let cursor = search_cursors.entry(labels).or_insert(256);
            let mut slot = first_free.max(*cursor);
            let base = loop {
                let base = slot ^ pivot;
                let required = (base | 255) + 1;
                if required > used.len() {
                    free_links.extend(used.len()+1..=required);
                    used.resize(required, false); used_bases.resize(required, false);
                }
                if base >= 256 && !used_bases[base] && edges.iter().all(|&(label,_)| !used[base ^ label as usize]) {
                    break base;
                }
                slot = next_free(&mut free_links, slot+1);
            };
            u32::try_from(base).map_err(|_| "transition table exceeds u32")?;
            bases[node] = base;
            *cursor = slot + 1;
            used_bases[base] = true;
            for &(label, _) in edges {
                let slot = base ^ label as usize;
                used[slot] = true;
                free_links[slot] = next_free(&mut free_links,slot+1);
            }
            first_free = next_free(&mut free_links,first_free);
        }
        let state = |node: usize, label: u8| (bases[node] as u64) << 32 | (build[node].0 as u64) << 8 | label as u64;
        let mut transitions = vec![0u64; used.len()];
        for (node,(_,edges)) in build.iter().enumerate() {
            for &(label,child) in edges { transitions[bases[node] ^ label as usize] = state(child as usize,label); }
        }
        let trie_roots: Vec<u64> = trie_roots.into_iter().map(|node| if node==0 {0} else {state(node as usize,0)}).collect();

        if !partial {
            for (byte, id) in single.iter().enumerate() {
                if *id < 0 { return Err(format!("missing singleton byte 0x{byte:02x}")); }
            }
        }
        permit.finish(text.len());
        Ok(Self { data: Arc::new(TokenizerData { tok_blob, tok_off, single, trie_roots, transitions, partial }) })
    }

    pub fn vocab_size(&self) -> usize { VOCAB_SIZE }
    pub fn is_partial(&self) -> bool { self.data.partial }

    #[inline]
    fn match_at(&self, src: &[u8], i: usize) -> (u32,u8) {
        let first = src[i] as usize;
        let mut best = (0u32,0u8);
        if i+1 < src.len() {
            let mut state = self.data.trie_roots[first*256 + src[i+1] as usize];
            if state != 0 {
                let id = (state >> 8) as u16;
                if id != 0 {
                    best = (id as u32,2);
                    if state >> 32 == 0 { return best; }
                }
                for (offset,&byte) in src[i+2..].iter().enumerate() {
                    let next = self.data.transitions[(state >> 32) as usize ^ byte as usize];
                    if next == 0 || next as u8 != byte { break; }
                    state = next;
                    let id = (state >> 8) as u16;
                    // Every later terminal is a longer match, regardless of ID.
                    if id != 0 { best = (id as u32,(offset+3) as u8); }
                    if state >> 32 == 0 { break; }
                }
            }
        }
        if best.0 != 0 {best}
        else if self.data.single[first]>=0 {(self.data.single[first] as u32,1)}
        else {(0,0)}
    }

    fn encode_sequential_bytes(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> {
        let mut result = Vec::with_capacity(src.len() / 3 + 8);
        let mut i = 0;
        while i < src.len() {
            let permit = cooperate::acquire();
            let end = i.saturating_add(permit.byte_budget()).min(src.len());
            let start = i;
            while i < end {
                // Match against the entire input: tokens can cross a quantum boundary.
                let (id, len) = self.match_at(src, i);
                if len == 0 { return Err(EncodeError { byte_offset: i, byte: src[i] }); }
                result.push(id);
                i += len as usize;
            }
            permit.finish(i - start);
        }
        Ok(result)
    }

    /// Sequential greedy encoding under the shared CPU budget.
    pub fn encode_sequential(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> { self.encode_sequential_bytes(src) }

    /// Explicit parallel matching, capped by the process-wide CPU budget.
    pub fn encode_parallel_with_workers(&self, src: &[u8], workers: usize) -> Result<Vec<u32>, EncodeError> {
        self.encode_with_policy(src, workers, PARALLEL_MIN_BYTES)
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

    /// Zero/one workers stay sequential. Larger requests share the global pool.
    pub fn encode_with_policy(&self, src: &[u8], workers: usize, min_bytes: usize) -> Result<Vec<u32>, EncodeError> {
        self.encode_with_policy_report(src, workers, min_bytes).map(|(tokens, _)| tokens)
    }

    /// Encode and report the effective matching job count from this call.
    /// One means sequential (including empty input); larger counts mean the
    /// parallel matching path. Jobs share the pool and CPU budget, so this is
    /// not a measurement of simultaneous OS threads.
    pub fn encode_with_policy_report(&self, src: &[u8], workers: usize, min_bytes: usize) -> Result<(Vec<u32>, usize), EncodeError> {
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
                sender.send((start / chunk, matches)).expect("matching receiver closed");
            });
        }
        drop(sender);
        let mut parts = vec![None; src.len().div_ceil(chunk)];
        for (index, matches) in receiver { parts[index] = Some(matches); }
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

    pub fn encode_parallel(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> {
        self.encode_parallel_with_workers(src, cooperate::worker_limit())
    }

    /// Encode documents through a shared queue and preserve input order.
    /// Errors identify the first failing document and its byte offset.
    pub fn encode_batch(&self, texts: &[&str], workers: usize) -> Result<Vec<Vec<u32>>, BatchEncodeError> {
        let workers = workers.max(1).min(cooperate::worker_limit()).min(texts.len());
        if workers <= 1 {
            return texts.iter().enumerate().map(|(document_index, text)| {
                self.encode_sequential_bytes(text.as_bytes()).map_err(|source| BatchEncodeError { document_index, source })
            }).collect();
        }
        let inputs = Arc::new(texts.iter().map(|text| Self::copy_input(text.as_bytes())).collect::<Vec<_>>());
        let next = Arc::new(AtomicUsize::new(0));
        let (sender, receiver) = mpsc::channel();
        for _ in 0..workers {
            let inputs = Arc::clone(&inputs);
            let next = Arc::clone(&next);
            let tokenizer = self.clone();
            let sender = sender.clone();
            cooperate::submit(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(input) = inputs.get(index) else { break };
                    let result = tokenizer.encode_sequential_bytes(input);
                    sender.send((index, result)).expect("batch receiver closed");
                }
            });
        }
        drop(sender);
        let mut output = vec![None; texts.len()];
        for (index, result) in receiver { output[index] = Some(result); }
        output.into_iter().enumerate().map(|(document_index, result)| {
            result.expect("batch worker panicked").map_err(|source| BatchEncodeError { document_index, source })
        }).collect()
    }

    /// Small batches stay sequential; larger batches share a bounded worker pool.
    pub fn encode_batch_auto(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, BatchEncodeError> {
        let total = texts.iter().fold(0usize, |sum, text| sum.saturating_add(text.len()));
        let workers = if total < BATCH_MIN_BYTES { 1 } else { total / BATCH_BYTES_PER_WORKER };
        self.encode_batch(texts, workers)
    }

    pub fn encode(&self, src: &str) -> Result<Vec<u32>, EncodeError> { self.encode_bytes(src.as_bytes()) }

    pub fn encode_bytes(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> {
        self.encode_sequential_bytes(src)
    }

    pub fn encode_checked(&self, src: &str) -> Result<Vec<u32>, EncodeError> {
        self.encode(src)
    }

    pub fn encode_bytes_checked(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> {
        self.encode_bytes(src)
    }

    pub fn decode_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, DecodeError> {
        self.decode_bytes_checked(tokens)
    }

    pub fn decode_bytes_checked(&self, tokens: &[u32]) -> Result<Vec<u8>, DecodeError> {
        let mut result = Vec::with_capacity(tokens.len() * 4);
        let mut index = 0;
        while index < tokens.len() {
            let permit = cooperate::acquire();
            let stop = index.saturating_add(permit.byte_budget() / 4).min(tokens.len());
            let produced = result.len();
            while index < stop {
            let token_id = tokens[index];
            let Some(id) = usize::try_from(token_id).ok().filter(|&id| id < VOCAB_SIZE) else {
                return Err(DecodeError::MissingToken { token_index: index, token_id });
            };
            let start = self.data.tok_off[id] as usize;
            let end = self.data.tok_off[id + 1] as usize;
            if start == end {
                return Err(DecodeError::MissingToken { token_index: index, token_id });
            }
            result.extend_from_slice(&self.data.tok_blob[start..end]);
            index += 1;
            }
            permit.finish(result.len() - produced);
        }
        Ok(result)
    }

    pub fn decode_utf8(&self, tokens: &[u32]) -> Result<String, DecodeError> {
        let bytes = self.decode_bytes(tokens)?;
        let permit = cooperate::acquire();
        let size = bytes.len();
        let result = String::from_utf8(bytes).map_err(DecodeError::InvalidUtf8);
        permit.finish(size);
        result
    }

    pub fn decode_utf8_checked(&self, tokens: &[u32]) -> Result<String, DecodeError> {
        self.decode_utf8(tokens)
    }

    pub fn decode_lossy(&self, tokens: &[u32]) -> Result<String, DecodeError> {
        let bytes = self.decode_bytes(tokens)?;
        let permit = cooperate::acquire();
        let result = String::from_utf8_lossy(&bytes).into_owned();
        permit.finish(bytes.len());
        Ok(result)
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
            let bytes = if id <= 256 { vec![(id - 1) as u8] } else {
                let pair = id - 257;
                vec![(pair & 255) as u8, (pair >> 8) as u8]
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
            let error = RwkvTokenizer::from_vocab_str_partial(text).unwrap_err();
            assert!(error.contains(expected), "{text:?}: {error}");
        }
        let long = vec![b'a'; 256];
        let error = RwkvTokenizer::from_vocab_str_partial(&format!("1 {} {}", literal(&long), long.len())).unwrap_err();
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
        assert_eq!(tokenizer.encode_bytes_checked(b"ab"), Ok(vec![1, 2]));
        assert_eq!(tokenizer.encode_bytes_checked(b"ac"), Err(EncodeError { byte_offset: 1, byte: b'c' }));
        for result in [
            tokenizer.encode("ac"),
            tokenizer.encode_bytes(b"ac"),
            tokenizer.encode_sequential(b"ac"),
            tokenizer.encode_parallel(b"ac"),
            tokenizer.encode_parallel_with_workers(b"ac", 2),
            tokenizer.encode_checked("ac"),
            tokenizer.encode_bytes_checked(b"ac"),
        ] {
            assert_eq!(result, Err(EncodeError { byte_offset: 1, byte: b'c' }));
        }
        for result in [tokenizer.decode_bytes(&[3]), tokenizer.decode_bytes_checked(&[3])] {
            assert!(matches!(result, Err(DecodeError::MissingToken { token_index: 0, token_id: 3 })));
        }
        assert!(matches!(tokenizer.decode_utf8(&[3]), Err(DecodeError::MissingToken { token_index: 0, token_id: 3 })));
        assert!(matches!(tokenizer.decode_utf8_checked(&[3]), Err(DecodeError::MissingToken { token_index: 0, token_id: 3 })));
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
            text.push_str(&format!("{id} {} 1\n", literal(&[(id - 1) as u8])));
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
        let sequential = tokenizer.encode_sequential(&input).unwrap();
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
            assert_eq!(tokenizer.match_at(token,0),(id as u32,token.len() as u8),"token {id}");
            count+=1;
        }
        assert_eq!(count,65529);
        let owned=std::mem::size_of::<RwkvTokenizer>() + std::mem::size_of::<TokenizerData>() + 2*std::mem::size_of::<usize>() + tokenizer.data.tok_blob.capacity() + tokenizer.data.tok_off.capacity()*4
            + tokenizer.data.trie_roots.capacity()*8 + tokenizer.data.transitions.capacity()*8;
        println!("transition_slots={} owned_bytes={}",tokenizer.data.transitions.len(),owned);
        use sha2::{Digest, Sha256};
        let mut layout = Sha256::new();
        for state in tokenizer.data.trie_roots.iter().chain(&tokenizer.data.transitions) { layout.update(state.to_le_bytes()); }
        println!("layout_sha256={:x}", layout.finalize());
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
    fn explicit_policies_preserve_empty_input_and_greedy_error_offsets() {
        // There is no singleton for b: only actual token starts may raise errors.
        let tokenizer = RwkvTokenizer::from_vocab_str_partial("1 b'a' 1\n2 b'ab' 2\n").unwrap();
        let valid = b"ab".repeat(8192);
        let mut invalid = valid.clone(); invalid.push(b'c');
        for workers in [0,1,2,4,8,16,usize::MAX] {
            for threshold in [0,16384,131072,usize::MAX] {
                assert_eq!(tokenizer.encode_with_policy(b"",workers,threshold),Ok(vec![]));
                assert_eq!(tokenizer.encode_with_policy(&valid,workers,threshold),tokenizer.encode_sequential(&valid));
                assert_eq!(tokenizer.encode_with_policy(&invalid,workers,threshold),
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
        assert_eq!(tokenizer.encode_with_policy(large.as_bytes(), 8, 0).unwrap(), expected);
        for count in [1,2,6,32] {
            assert_eq!(tokenizer.encode_batch_auto(&vec![large.as_str(); count]).unwrap(), vec![expected.clone(); count]);
        }
        let after = cooperation_stats();
        assert!(after.completed_quanta > before.completed_quanta);
        assert!(after.yields > before.yields);
        assert!(after.peak_active <= PARALLEL_MAX_WORKERS);
        assert!(after.pool_threads <= PARALLEL_MAX_WORKERS);
        if after.worker_limit > 1 { assert!(after.submitted_jobs > before.submitted_jobs); }
    }

    #[test]
    fn hybrid_degree_boundaries_preserve_longest_match_and_missing_offsets() {
        for count in [15u32,16,256] {
            for shorter_has_higher_id in [false,true] {
                let mut vocab=String::new();
                for byte in 0..256u32 { vocab.push_str(&format!("{} {} 1\n",byte+1,literal(&[byte as u8]))); }
                if !shorter_has_higher_id { vocab.push_str("299 b'ab' 2\n"); }
                let mut partial=String::new();
                if !shorter_has_higher_id { partial.push_str("299 b'ab' 2\n"); }
                for byte in 0..count {
                    let line=format!("{} {} 3\n",300+byte,literal(&[b'a',b'b',byte as u8]));
                    vocab.push_str(&line);partial.push_str(&line);
                }
                if shorter_has_higher_id { vocab.push_str("600 b'ab' 2\n");partial.push_str("600 b'ab' 2\n"); }
                let full=RwkvTokenizer::from_vocab_str(&vocab).unwrap();
                let partial=RwkvTokenizer::from_vocab_str_partial(&partial).unwrap();
                for byte in 0..count {
                    let input=[b'a',b'b',byte as u8];
                    assert_eq!(full.encode_sequential(&input).unwrap(),vec![300+byte]);
                    assert_eq!(partial.encode_sequential(&input),Ok(vec![300+byte]));
                }
                if count<256 {
                    let input=[b'a',b'b',255];
                    assert_eq!(partial.encode_sequential(&input),Err(EncodeError { byte_offset:2,byte:255 }));
                }
            }
        }
    }

}
