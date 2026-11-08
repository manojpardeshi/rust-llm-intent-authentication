//! Keyed token-bucket rate limiting for the protected endpoints.

use std::net::IpAddr;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use governor::clock::{Clock as _, DefaultClock};
use governor::{DefaultKeyedRateLimiter, Quota, RateLimiter};

use crate::config::RateLimitConfig;

/// Prune idle keys once every this many checks, bounding memory.
const PRUNE_EVERY: u64 = 1024;

/// Identity a bucket is kept for.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum RateKey {
    /// SHA-256 digest of an authenticated API key.
    ApiKey([u8; 32]),
    Ip(IpAddr),
    /// No client information available (e.g. in-process tests).
    Shared,
}

pub(crate) struct KeyedLimiter {
    limiter: DefaultKeyedRateLimiter<RateKey>,
    checks: AtomicU64,
}

impl KeyedLimiter {
    pub(crate) fn new(config: RateLimitConfig) -> Self {
        // Validated by `RateLimitConfig::validate`; clamp defensively anyway.
        let per_second = NonZeroU32::new(config.per_second).unwrap_or(NonZeroU32::MIN);
        let burst = NonZeroU32::new(config.burst).unwrap_or(NonZeroU32::MIN);
        Self {
            limiter: RateLimiter::keyed(Quota::per_second(per_second).allow_burst(burst)),
            checks: AtomicU64::new(0),
        }
    }

    /// `Err(retry_after)` when the key's bucket is empty.
    pub(crate) fn check(&self, key: &RateKey) -> Result<(), Duration> {
        if self.checks.fetch_add(1, Ordering::Relaxed) % PRUNE_EVERY == PRUNE_EVERY - 1 {
            self.limiter.retain_recent();
            self.limiter.shrink_to_fit();
        }
        self.limiter
            .check_key(key)
            .map_err(|nu| nu.wait_time_from(DefaultClock::default().now()))
    }

    #[cfg(test)]
    fn tracked_keys(&self) -> usize {
        self.limiter.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burst_then_limited_per_key() {
        let l = KeyedLimiter::new(RateLimitConfig {
            per_second: 1,
            burst: 2,
        });
        let a = RateKey::Shared;
        let b = RateKey::Ip("10.0.0.1".parse().unwrap());
        assert!(l.check(&a).is_ok() && l.check(&a).is_ok());
        let wait = l.check(&a).unwrap_err();
        assert!(wait <= Duration::from_secs(1));
        assert!(l.check(&b).is_ok());
        assert_eq!(l.tracked_keys(), 2);
    }
}
