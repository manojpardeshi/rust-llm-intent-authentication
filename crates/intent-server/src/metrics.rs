//! Prometheus metrics backed by a private (non-global) recorder.

use std::sync::Arc;
use std::time::Duration;

use metrics::{counter, histogram};
use metrics_exporter_prometheus::{
    Matcher, PrometheusBuilder, PrometheusHandle, PrometheusRecorder,
};

const DURATION: &str = "intent_evaluate_duration_seconds";

/// Records metrics into its own recorder, so tests and embedders never touch
/// process-global state.
#[derive(Clone)]
pub struct Metrics {
    recorder: Arc<PrometheusRecorder>,
    handle: PrometheusHandle,
}

impl Metrics {
    pub fn new() -> Self {
        let recorder = PrometheusBuilder::new()
            .set_buckets_for_metric(
                Matcher::Full(DURATION.into()),
                &[
                    0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
                ],
            )
            .expect("static buckets are non-empty")
            .build_recorder();
        let handle = recorder.handle();
        Self {
            recorder: Arc::new(recorder),
            handle,
        }
    }

    pub fn record_evaluation(
        &self,
        decision: &'static str,
        category: &'static str,
        took: Duration,
    ) {
        metrics::with_local_recorder(&*self.recorder, || {
            counter!("intent_evaluations_total", "decision" => decision, "category" => category)
                .increment(1);
            histogram!(DURATION).record(took.as_secs_f64());
        });
    }

    pub fn record_verification(&self, result: &'static str) {
        metrics::with_local_recorder(&*self.recorder, || {
            counter!("intent_verifications_total", "result" => result).increment(1);
        });
    }

    pub fn record_reload(&self, result: &'static str) {
        metrics::with_local_recorder(&*self.recorder, || {
            counter!("intent_reloads_total", "result" => result).increment(1);
        });
    }

    pub fn record_rate_limited(&self) {
        metrics::with_local_recorder(&*self.recorder, || {
            counter!("intent_rate_limited_total").increment(1);
        });
    }

    /// Prometheus text exposition.
    pub fn render(&self) -> String {
        self.handle.render()
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}
