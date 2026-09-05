#![warn(clippy::undocumented_unsafe_blocks)]

use std::{error::Error, io};

use rwkv_tokenizer::RwkvTokenizer;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize once alongside your model, then reuse for every request.
    let tokenizer = RwkvTokenizer::bundled()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let token_ids = tokenizer.encode("User: hello")?;
    let decoded = tokenizer.decode_bytes(&token_ids)?;
    assert_eq!(decoded, b"User: hello");
    Ok(())
}
