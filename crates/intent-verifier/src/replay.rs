//! Single-use token enforcement: remember each `jti` until its token expires.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Seconds past `exp` that intent-token's verifier still accepts a token.
/// Entries must outlive that window or an expiring token could be replayed.
const LEEWAY: i64 = 5;

/// How often (seconds) an insert also sweeps expired entries, so memory stays
/// bounded by live tokens even when the store is never near capacity.
const SWEEP_INTERVAL: i64 = 30;

/// Why a token id could not be consumed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReplayError {
    /// This `jti` was already used.
    #[error("token already used")]
    Replayed,
    /// The store holds only unexpired entries and is at capacity. Fails
    /// closed: nothing is evicted.
    #[error("replay store is full")]
    Full,
    /// An external store failed. The message must not contain secrets.
    #[error("replay store backend: {0}")]
    Backend(String),
}

/// Records token ids so each token can be used once.
///
/// `check_and_insert` must be atomic: of any number of concurrent calls with
/// the same `jti`, exactly one returns `Ok`. Entries need only be kept until
/// `exp` plus a few seconds of leeway.
///
/// The trait is synchronous, which suits in-process stores. A network-backed
/// implementation (for example Redis `SET jti 1 NX EXAT exp`) can be added
/// later, either by blocking in a `spawn_blocking`-friendly client or by
/// introducing an async variant of this trait; callers only depend on
/// `Arc<dyn ReplayStore>`.
pub trait ReplayStore: Send + Sync {
    /// Atomically records `jti` (a token expiring at unix time `exp`).
    fn check_and_insert(&self, jti: &str, exp: i64) -> Result<(), ReplayError>;
}

/// Bounded in-memory [`ReplayStore`]. Per-process only: with several gateway
/// instances, use a shared store instead.
///
/// Entries expire at `exp + 5s` and are purged lazily. When the store is full
/// of unexpired entries, new tokens are refused (`Full`) rather than evicting
/// a live entry, since eviction would re-enable replay of that token.
pub struct InMemoryReplayStore {
    capacity: usize,
    state: Mutex<State>,
}

struct State {
    /// jti -> unix time after which the entry may be forgotten.
    entries: HashMap<String, i64>,
    next_sweep: i64,
}

impl InMemoryReplayStore {
    /// Default capacity of [`InMemoryReplayStore::default`].
    pub const DEFAULT_CAPACITY: usize = 100_000;

    /// Store holding at most `capacity` unexpired token ids.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::new(State {
                entries: HashMap::new(),
                next_sweep: 0,
            }),
        }
    }

    fn check_and_insert_at(&self, jti: &str, exp: i64, now: i64) -> Result<(), ReplayError> {
        let expires = exp.saturating_add(LEEWAY);
        // A poisoned lock cannot leave the map inconsistent (single ops).
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if now >= st.next_sweep {
            st.entries.retain(|_, e| *e > now);
            st.next_sweep = now.saturating_add(SWEEP_INTERVAL);
        }
        match st.entries.get(jti) {
            Some(e) if *e > now => return Err(ReplayError::Replayed),
            Some(_) => {} // expired leftover; overwrite below
            None => {
                if st.entries.len() >= self.capacity {
                    st.entries.retain(|_, e| *e > now);
                    if st.entries.len() >= self.capacity {
                        return Err(ReplayError::Full);
                    }
                }
            }
        }
        st.entries.insert(jti.to_owned(), expires);
        Ok(())
    }
}

impl Default for InMemoryReplayStore {
    fn default() -> Self {
        Self::new(Self::DEFAULT_CAPACITY)
    }
}

impl ReplayStore for InMemoryReplayStore {
    fn check_and_insert(&self, jti: &str, exp: i64) -> Result<(), ReplayError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        self.check_and_insert_at(jti, exp, now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn first_ok_second_replayed() {
        let s = InMemoryReplayStore::new(10);
        assert_eq!(s.check_and_insert_at("a", 1000, 900), Ok(()));
        assert_eq!(
            s.check_and_insert_at("a", 1000, 901),
            Err(ReplayError::Replayed)
        );
        assert_eq!(s.check_and_insert_at("b", 1000, 901), Ok(()));
    }

    #[test]
    fn still_replayed_within_leeway() {
        let s = InMemoryReplayStore::new(10);
        s.check_and_insert_at("a", 1000, 900).unwrap();
        assert_eq!(
            s.check_and_insert_at("a", 1000, 1004),
            Err(ReplayError::Replayed)
        );
    }

    #[test]
    fn expired_entries_are_purged_and_slot_reusable() {
        let s = InMemoryReplayStore::new(1);
        s.check_and_insert_at("a", 1000, 900).unwrap();
        assert_eq!(
            s.check_and_insert_at("b", 2000, 901),
            Err(ReplayError::Full)
        );
        // Past exp + leeway the slot frees up.
        assert_eq!(s.check_and_insert_at("b", 2000, 1005), Ok(()));
        assert_eq!(s.state.lock().unwrap().entries.len(), 1);
    }

    #[test]
    fn full_store_fails_closed_without_eviction() {
        let s = InMemoryReplayStore::new(2);
        s.check_and_insert_at("a", 1000, 900).unwrap();
        s.check_and_insert_at("b", 1000, 900).unwrap();
        assert_eq!(
            s.check_and_insert_at("c", 1000, 900),
            Err(ReplayError::Full)
        );
        // Neither earlier entry was evicted.
        assert_eq!(
            s.check_and_insert_at("a", 1000, 901),
            Err(ReplayError::Replayed)
        );
        assert_eq!(
            s.check_and_insert_at("b", 1000, 901),
            Err(ReplayError::Replayed)
        );
    }

    #[test]
    fn zero_capacity_is_always_full() {
        let s = InMemoryReplayStore::new(0);
        assert_eq!(
            s.check_and_insert_at("a", 1000, 900),
            Err(ReplayError::Full)
        );
    }

    #[test]
    fn periodic_sweep_bounds_memory() {
        let s = InMemoryReplayStore::new(1000);
        for i in 0..10 {
            s.check_and_insert_at(&format!("j{i}"), 100, 50).unwrap();
        }
        s.check_and_insert_at("late", 500, 200).unwrap();
        assert_eq!(s.state.lock().unwrap().entries.len(), 1);
    }

    #[test]
    fn concurrent_same_jti_exactly_one_ok() {
        let s = Arc::new(InMemoryReplayStore::new(100));
        let exp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            + 60;
        let handles: Vec<_> = (0..32)
            .map(|_| {
                let s = s.clone();
                std::thread::spawn(move || s.check_and_insert("same", exp))
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        assert!(
            results
                .iter()
                .filter(|r| r.is_err())
                .all(|r| *r == Err(ReplayError::Replayed))
        );
    }
}
