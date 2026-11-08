//! Reload triggers: SIGHUP and optional file watching.

use std::path::{Path, PathBuf};
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode, Watcher as _};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::{AppState, Error};

/// Quiet period after the last file event before reloading.
const DEBOUNCE: Duration = Duration::from_millis(500);

/// Keeps reload triggers alive; dropping it stops them.
pub struct ReloadGuard {
    tasks: Vec<JoinHandle<()>>,
    _watcher: Option<RecommendedWatcher>,
}

impl Drop for ReloadGuard {
    fn drop(&mut self) {
        for t in &self.tasks {
            t.abort();
        }
    }
}

/// Starts the reload triggers for `state`: SIGHUP on unix, and file watching
/// when the config sets `watch = true`. Must run inside a tokio runtime.
/// Returns an error if the state has no config or the watcher cannot start.
pub fn spawn_reload_triggers(state: &AppState) -> Result<ReloadGuard, Error> {
    let config = state
        .config()
        .ok_or_else(|| Error::Config("no config to reload from".into()))?;
    let mut guard = ReloadGuard {
        tasks: Vec::new(),
        _watcher: None,
    };
    #[cfg(unix)]
    guard.tasks.push(spawn_sighup(state.clone())?);
    if config.watch {
        let (watcher, task) = spawn_watcher(state.clone())?;
        guard._watcher = Some(watcher);
        guard.tasks.push(task);
    }
    Ok(guard)
}

#[cfg(unix)]
fn spawn_sighup(state: AppState) -> Result<JoinHandle<()>, Error> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut hup = signal(SignalKind::hangup())
        .map_err(|e| Error::Config(format!("installing SIGHUP handler: {e}")))?;
    Ok(tokio::spawn(async move {
        while hup.recv().await.is_some() {
            tracing::info!(target: "intent::server", "SIGHUP received, reloading");
            reload_blocking(&state).await;
        }
    }))
}

/// Reload does blocking file IO, so keep it off the async workers. Failures
/// are already logged by `reload`.
async fn reload_blocking(state: &AppState) {
    let state = state.clone();
    let _ = tokio::task::spawn_blocking(move || state.reload()).await;
}

/// Watches the parent directories of the policy and rules files (editors
/// replace files, which breaks per-file watches) and reloads after a quiet
/// period.
fn spawn_watcher(state: AppState) -> Result<(RecommendedWatcher, JoinHandle<()>), Error> {
    let config = state
        .config()
        .ok_or_else(|| Error::Config("no config to reload from".into()))?;
    let targets: Vec<PathBuf> = [&config.policy_path, &config.rules_path]
        .into_iter()
        .map(|p| canonical_target(p))
        .collect::<Result<_, _>>()?;

    let (tx, mut rx) = mpsc::unbounded_channel::<()>();
    let filter = targets.clone();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(ev) = res
            && ev.paths.iter().any(|p| filter.contains(p))
        {
            let _ = tx.send(());
        }
    })
    .map_err(|e| Error::Config(format!("starting file watcher: {e}")))?;

    let mut dirs: Vec<&Path> = targets.iter().filter_map(|t| t.parent()).collect();
    dirs.sort();
    dirs.dedup();
    for dir in dirs {
        watcher
            .watch(dir, RecursiveMode::NonRecursive)
            .map_err(|e| Error::Config(format!("watching {}: {e}", dir.display())))?;
    }

    let task = tokio::spawn(async move {
        while rx.recv().await.is_some() {
            // Debounce: wait until events stop arriving for DEBOUNCE.
            while let Ok(Some(())) = tokio::time::timeout(DEBOUNCE, rx.recv()).await {}
            tracing::info!(target: "intent::server", "file change detected, reloading");
            reload_blocking(&state).await;
        }
    });
    Ok((watcher, task))
}

/// Canonical parent directory joined with the file name, so it matches the
/// absolute paths watchers report even when the file is momentarily absent.
fn canonical_target(path: &Path) -> Result<PathBuf, Error> {
    let name = path
        .file_name()
        .ok_or_else(|| Error::Config(format!("{} has no file name", path.display())))?;
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let parent = parent
        .canonicalize()
        .map_err(|e| Error::Config(format!("resolving {}: {e}", parent.display())))?;
    Ok(parent.join(name))
}
