#![no_main]

use intent_core::{ToolCall, tool_calls_digest};
use libfuzzer_sys::fuzz_target;

// Parse arbitrary bytes as tool calls, digest them, then re-serialize and
// re-parse: the digest must be stable across the round trip.
fuzz_target!(|data: &[u8]| {
    let Ok(calls) = serde_json::from_slice::<Vec<ToolCall>>(data) else {
        return;
    };
    let first = tool_calls_digest(&calls);

    let reserialized = serde_json::to_vec(&calls).expect("tool calls always serialize");
    let reparsed: Vec<ToolCall> =
        serde_json::from_slice(&reserialized).expect("serialized tool calls always parse");
    let second = tool_calls_digest(&reparsed);

    assert_eq!(first.is_ok(), second.is_ok(), "digest success changed on round trip");
    if let (Ok(a), Ok(b)) = (first, second) {
        assert_eq!(a, b, "digest changed on round trip");
    }
});
