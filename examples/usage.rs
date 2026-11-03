//! Minimal encode/decode round trip with the bundled vocabulary.

use std::error::Error;

use rwkv_tokenizer::RwkvTokenizer;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize once alongside your model, then reuse for every request.
    let tokenizer = RwkvTokenizer::bundled()?;
    let token_ids = tokenizer.encode("User: hello")?;
    let decoded = tokenizer.decode_bytes(&token_ids)?;
    assert_eq!(decoded, b"User: hello");
    Ok(())
}
