//! Background filesystem watcher triggering zero-search-stall weight hot-swaps.

use flume::Sender;
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::fs;
use std::path::{Path, PathBuf};

/// Starts monitoring `model_path` for atomic file updates, notifying `reload_tx`.
pub fn start_model_watcher(
    model_path: impl AsRef<Path>,
    reload_tx: Sender<PathBuf>,
) -> notify::Result<RecommendedWatcher> {
    let target_path = model_path.as_ref().to_path_buf();
    let watch_dir = target_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&watch_dir)?;

    let target_filename = target_path
        .file_name()
        .expect("Invalid model filename")
        .to_os_string();

    let mut watcher = RecommendedWatcher::new(
        move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                if matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                    for path in &event.paths {
                        if path.file_name() == Some(&target_filename) {
                            if let Ok(metadata) = fs::metadata(path) {
                                if metadata.len() > 0 {
                                    let _ = reload_tx.send(path.clone());
                                }
                            }
                        }
                    }
                }
            }
        },
        Config::default(),
    )?;

    watcher.watch(&watch_dir, RecursiveMode::NonRecursive)?;
    Ok(watcher)
}
