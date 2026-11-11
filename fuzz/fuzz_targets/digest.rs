#![no_main]

use intent_core::{MAX_SAFE_INTEGER, ToolCall, tool_calls_digest};
use libfuzzer_sys::fuzz_target;

fn call(arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        name: "t".into(),
        arguments,
    }
}

fuzz_target!(|data: &[u8]| {
    // Integers across the full i64/u64 range: accepted exactly when a double
    // represents them, and accepted values never share a digest.
    if data.len() >= 16 {
        let a = i64::from_le_bytes(data[..8].try_into().unwrap());
        let b = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let da = tool_calls_digest(&[call(a.into())]);
        let db = tool_calls_digest(&[call(b.into())]);
        assert_eq!(da.is_ok(), a.unsigned_abs() <= MAX_SAFE_INTEGER);
        assert_eq!(db.is_ok(), b <= MAX_SAFE_INTEGER);
        if let (Ok(x), Ok(y)) = (da, db) {
            assert_eq!(x == y, a >= 0 && a as u64 == b, "integer digest collision");
        }
    }

    // Parse arbitrary bytes as tool calls, digest them, then re-serialize and
    // re-parse: the digest must be stable across the round trip.
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
