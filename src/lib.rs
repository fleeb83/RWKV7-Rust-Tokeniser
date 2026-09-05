#![warn(clippy::undocumented_unsafe_blocks)]

//! Standalone RWKV World tokenizer.

use std::{fmt, path::Path};

pub const VOCAB_SIZE: usize = 65_536;
pub const RESERVED_TOKEN_ID: u32 = 0;
pub const PARALLEL_MIN_BYTES: usize = 1 << 17;

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

#[derive(Debug, Clone, Copy)]
struct Bucket { group_start: u32, group_count: u16 }

#[derive(Debug, Clone, Copy)]
struct Entry { blob_offset: u32, token_len: u8, token_id: u32 }

#[derive(Debug)]
pub struct RwkvTokenizer {
    tok_blob: Vec<u8>,
    tok_off: Vec<u32>,
    single: [i32; 256],
    blob: Vec<u8>,
    entries: Vec<Entry>,
    buckets: Vec<Bucket>,
    groups: Vec<(u16, u32, u32)>,
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

        let mut by_bucket: Vec<Vec<(u32, &[u8])>> = vec![Vec::new(); 256 * 256];
        for (id, token) in idx2tok.iter().enumerate().rev() {
            let Some(token) = token.as_deref() else { continue };
            if token.len() == 1 {
                single[token[0] as usize] = id as i32;
            } else {
                by_bucket[token[0] as usize * 256 + token[1] as usize].push((id as u32, token));
            }
        }

        let mut blob = Vec::new();
        let mut entries = Vec::new();
        let mut buckets = vec![Bucket { group_start: 0, group_count: 0 }; 256 * 256];
        let mut groups = Vec::new();
        let mut counts = [0u16; 257];
        for (bucket_id, candidates) in by_bucket.iter().enumerate() {
            if candidates.is_empty() { continue; }
            for (_, token) in candidates { counts[if token.len() == 2 { 256 } else { token[2] as usize }] += 1; }
            let group_start = groups.len() as u32;
            let mut offsets = [0u16; 257];
            let mut total = 0u32;
            let mut group_count = 0u16;
            for group in 0..257 {
                offsets[group] = total as u16;
                total += counts[group] as u32;
                if counts[group] != 0 { group_count += 1; }
            }
            let mut placed: Vec<Option<&(u32, &[u8])>> = vec![None; candidates.len()];
            for candidate in candidates {
                let group = if candidate.1.len() == 2 { 256 } else { candidate.1[2] as usize };
                placed[offsets[group] as usize] = Some(candidate);
                offsets[group] += 1;
            }
            let mut cursor = 0usize;
            for (group, count_ref) in counts.iter_mut().enumerate() {
                let count = *count_ref as usize;
                if count == 0 { continue; }
                let start = entries.len() as u32;
                for candidate in placed[cursor..cursor + count].iter().copied() {
                    let (id, token) = candidate.expect("counting-sort placement invariant");
                    let offset = u32::try_from(blob.len()).map_err(|_| "match storage exceeds u32".to_owned())?;
                    let new_len = blob.len().checked_add(token.len()).ok_or_else(|| "match blob overflow".to_owned())?;
                    if new_len > u32::MAX as usize { return Err("match storage exceeds u32".to_owned()); }
                    blob.extend_from_slice(token);
                    entries.push(Entry { blob_offset: offset, token_len: token.len() as u8, token_id: *id });
                }
                cursor += count;
                groups.push((group as u16, start, entries.len() as u32));
                *count_ref = 0;
            }
            buckets[bucket_id] = Bucket { group_start, group_count };
        }

        if !partial {
            for (byte, id) in single.iter().enumerate() {
                if *id < 0 { return Err(format!("missing singleton byte 0x{byte:02x}")); }
            }
        }
        Ok(Self { tok_blob, tok_off, single, blob, entries, buckets, groups, partial })
    }

    pub fn vocab_size(&self) -> usize { VOCAB_SIZE }
    pub fn is_partial(&self) -> bool { self.partial }

    #[inline]
    fn match_at(&self, src: &[u8], i: usize) -> (u32, u8) {
        let s0 = src[i] as usize;
        let mut best = None;
        if i + 1 < src.len() {
            let b = s0 * 256 + src[i + 1] as usize;
            let bucket = self.buckets[b];
            let groups = &self.groups[bucket.group_start as usize..bucket.group_start as usize + bucket.group_count as usize];
            let hit = |wanted: u16| -> Option<(u32, u8)> {
                for &(group, start, end) in groups {
                    if group != wanted { continue; }
                    for entry in &self.entries[start as usize..end as usize] {
                        let len = entry.token_len as usize;
                        let off = entry.blob_offset as usize;
                        if len <= src.len() - i && self.blob[off..off + len] == src[i..i + len] {
                            return Some((entry.token_id, entry.token_len));
                        }
                    }
                }
                None
            };
            if i + 2 < src.len() { best = hit(src[i + 2] as u16); }
            if let Some(pair) = hit(256) {
                if best.is_none_or(|candidate| pair.0 > candidate.0) { best = Some(pair); }
            }
        }
        best.or_else(|| (self.single[s0] >= 0).then_some((self.single[s0] as u32, 1))).unwrap_or((0, 0))
    }

    fn encode_sequential_bytes(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> {
        let mut result = Vec::with_capacity(src.len() / 3 + 8);
        let mut i = 0;
        while i < src.len() {
            let (id, len) = self.match_at(src, i);
            if len == 0 { return Err(EncodeError { byte_offset: i, byte: src[i] }); }
            result.push(id);
            i += len as usize;
        }
        Ok(result)
    }

    /// Explicit sequential path; unlike `encode`, this never auto-dispatches.
    pub fn encode_sequential(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> { self.encode_sequential_bytes(src) }

    /// Explicit worker path. `workers >= 2` and a threshold-sized input force
    /// the spawn branch even when host-reported parallelism is one.
    pub fn encode_parallel_with_workers(&self, src: &[u8], workers: usize) -> Result<Vec<u32>, EncodeError> {
        if src.len() < PARALLEL_MIN_BYTES || workers <= 1 { return self.encode_sequential_bytes(src); }
        let workers = workers.min(16);
        let mut matches = vec![(0u32, 0u8); src.len()];
        std::thread::scope(|scope| {
            let chunk = src.len().div_ceil(workers);
            for (worker, part) in matches.chunks_mut(chunk).enumerate() {
                let start = worker * chunk;
                scope.spawn(move || {
                    for (offset, slot) in part.iter_mut().enumerate() { *slot = self.match_at(src, start + offset); }
                });
            }
        });
        let mut result = Vec::with_capacity(src.len() / 3 + 8);
        let mut i = 0;
        while i < src.len() {
            let (id, len) = matches[i];
            if len == 0 { return Err(EncodeError { byte_offset: i, byte: src[i] }); }
            result.push(id);
            i += len as usize;
        }
        Ok(result)
    }

    pub fn encode_parallel(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> {
        let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        self.encode_parallel_with_workers(src, workers)
    }

    pub fn encode(&self, src: &str) -> Result<Vec<u32>, EncodeError> { self.encode_bytes(src.as_bytes()) }

    pub fn encode_bytes(&self, src: &[u8]) -> Result<Vec<u32>, EncodeError> {
        if src.len() >= PARALLEL_MIN_BYTES { self.encode_parallel(src) } else { self.encode_sequential_bytes(src) }
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
        for (index, &token_id) in tokens.iter().enumerate() {
            let Some(id) = usize::try_from(token_id).ok().filter(|&id| id < VOCAB_SIZE) else {
                return Err(DecodeError::MissingToken { token_index: index, token_id });
            };
            let start = self.tok_off[id] as usize;
            let end = self.tok_off[id + 1] as usize;
            if start == end {
                return Err(DecodeError::MissingToken { token_index: index, token_id });
            }
            result.extend_from_slice(&self.tok_blob[start..end]);
        }
        Ok(result)
    }

    pub fn decode_utf8(&self, tokens: &[u32]) -> Result<String, DecodeError> {
        String::from_utf8(self.decode_bytes(tokens)?).map_err(DecodeError::InvalidUtf8)
    }

    pub fn decode_utf8_checked(&self, tokens: &[u32]) -> Result<String, DecodeError> {
        self.decode_utf8(tokens)
    }

    pub fn decode_lossy(&self, tokens: &[u32]) -> Result<String, DecodeError> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(tokens)?).into_owned())
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
        tokens.sort_by_key(|(id, _)| std::cmp::Reverse(*id));
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
            tokenizer.encode_parallel_with_workers(&input, 2),
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
        // The shorter pair has the higher ID, so it wins even when the
        // longer triple also matches. The second triple exercises a group
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
        assert_eq!(tokenizer.encode_bytes(input).unwrap(), vec![301, 123, 302]);
    }

    #[test]
    fn forced_sequential_and_parallel_paths_match() {
        let tokenizer = RwkvTokenizer::from_vocab_str(&valid_fixture()).unwrap();
        let seed = b"The quick brown fox \xe4\xb8\xad\xe6\x96\x87\n";
        let mut input = Vec::with_capacity(PARALLEL_MIN_BYTES + seed.len());
        while input.len() < PARALLEL_MIN_BYTES { input.extend_from_slice(seed); }
        let sequential = tokenizer.encode_sequential(&input).unwrap();
        let parallel = tokenizer.encode_parallel_with_workers(&input, 2).unwrap();
        assert_eq!(sequential, parallel);
    }
}
