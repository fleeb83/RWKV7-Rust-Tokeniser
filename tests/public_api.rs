//! Integration checks that use only the public API.

use rwkv_tokenizer::RwkvTokenizer;

#[test]
fn bundled_text_round_trip_with_emoji_and_cjk() {
    let tokenizer = RwkvTokenizer::bundled().unwrap();
    for text in ["", "Hello, world!", "こんにちは世界", "🦀🎉 mixed 中文 text\n", "tab\tand\r\nnewline"] {
        let ids = tokenizer.encode(text).unwrap();
        assert_eq!(tokenizer.decode_utf8(&ids).unwrap(), text);
        assert_eq!(tokenizer.decode_lossy(&ids).unwrap(), text);
    }
}

#[test]
fn bytes_api_round_trips_nul_and_0xff() {
    let tokenizer = RwkvTokenizer::bundled().unwrap();
    let bytes = [0x00, 0xff, b'R', 0x80, 0xfe, 0x00];
    let ids = tokenizer.encode_bytes(&bytes).unwrap();
    assert_eq!(tokenizer.decode_bytes(&ids).unwrap(), bytes);
    assert!(tokenizer.decode_utf8(&ids).is_err());
}

#[test]
fn batch_keeps_document_order() {
    let tokenizer = RwkvTokenizer::bundled().unwrap();
    let texts = ["first", "second 世界", "", "fourth 🦀"];
    for batch in [tokenizer.encode_batch(&texts, 3).unwrap(), tokenizer.encode_batch_auto(&texts).unwrap()] {
        assert_eq!(batch.len(), texts.len());
        for (text, ids) in texts.iter().zip(&batch) {
            assert_eq!(&tokenizer.encode(text).unwrap(), ids);
            assert_eq!(tokenizer.decode_utf8(ids).unwrap(), *text);
        }
    }
}
