//! High-throughput, zero-inode-churn trajectory spooler with atomic rename synchronization.

use crate::wire::{ChunkHeader, StepRecord};
use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static CHUNK_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Trajectory spooler that buffers search steps and atomically flushes chunks to disk.
pub struct TrajectorySpooler {
    spool_dir: PathBuf,
    worker_id: u32,
    chunk_size: usize,
    game_id: u32,
    channels: u32,
    height: u32,
    width: u32,
    action_dim: u32,
    num_players: u32,
    records: Vec<StepRecord>,
}

impl TrajectorySpooler {
    /// Creates a new `TrajectorySpooler` configured to spool chunks into `spool_dir`.
    pub fn new(
        spool_dir: impl AsRef<Path>,
        worker_id: u32,
        chunk_size: usize,
        game_id: u32,
        channels: u32,
        height: u32,
        width: u32,
        action_dim: u32,
        num_players: u32,
    ) -> io::Result<Self> {
        let spool_path = spool_dir.as_ref().to_path_buf();
        fs::create_dir_all(&spool_path)?;

        Ok(Self {
            spool_dir: spool_path,
            worker_id,
            chunk_size,
            game_id,
            channels,
            height,
            width,
            action_dim,
            num_players,
            records: Vec::with_capacity(chunk_size),
        })
    }

    /// Appends a `record` to the in-memory buffer.
    ///
    /// If the buffer reaches `chunk_size`, triggers an automatic atomic flush to disk,
    /// returning `Ok(Some(path))` with the newly completed `.bin` chunk path.
    pub fn push(&mut self, record: StepRecord) -> io::Result<Option<PathBuf>> {
        self.records.push(record);
        if self.records.len() >= self.chunk_size {
            self.flush()
        } else {
            Ok(None)
        }
    }

    /// Number of records currently buffered in memory.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` if no records are currently buffered.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Flushes all currently buffered records to a binary chunk file via atomic rename.
    ///
    /// The file is written to `traj_<worker>_<timestamp>_<counter>.bin.tmp`, flushed,
    /// synced, and renamed to `.bin`.
    pub fn flush(&mut self) -> io::Result<Option<PathBuf>> {
        if self.records.is_empty() {
            return Ok(None);
        }

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let counter = CHUNK_COUNTER.fetch_add(1, Ordering::Relaxed);

        let file_stem = format!("traj_{}_{}_{}", self.worker_id, ts, counter);
        let tmp_path = self.spool_dir.join(format!("{file_stem}.bin.tmp"));
        let final_path = self.spool_dir.join(format!("{file_stem}.bin"));

        let header = ChunkHeader::new(
            self.game_id,
            self.records.len() as u32,
            self.channels,
            self.height,
            self.width,
            self.action_dim,
            self.num_players,
        );

        {
            let file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(file);

            header.write_to(&mut writer)?;
            for record in &self.records {
                record.write_to(&mut writer)?;
            }

            use std::io::Write;
            writer.flush()?;
            let inner_file = writer.into_inner().map_err(|e| e.into_error())?;
            inner_file.sync_all()?;
        }

        fs::rename(&tmp_path, &final_path)?;
        self.records.clear();

        Ok(Some(final_path))
    }
}
