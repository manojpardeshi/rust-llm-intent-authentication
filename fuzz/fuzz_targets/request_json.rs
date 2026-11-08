#![no_main]

use intent_core::{IntentRequest, encode_digest, request_digest};
use libfuzzer_sys::fuzz_target;

// Arbitrary bytes as an IntentRequest, then its request digest. Must never panic.
fuzz_target!(|data: &[u8]| {
    if let Ok(request) = serde_json::from_slice::<IntentRequest>(data) {
        if let Ok(digest) = request_digest(&request) {
            let _ = encode_digest(&digest);
        }
    }
});
