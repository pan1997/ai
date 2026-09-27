//! Compact, fixed-stride binary wire protocol for trajectory chunk spooling.
//!
//! # Protocol Specification
//!
//! Each chunk file consists of a fixed 64-byte header followed by $K$ contiguous,
//! fixed-stride step records.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                    HEADER (64 Bytes)                         │
//! ├─────────────────┬──────────────┬──────────────┬──────────────┤
//! │ magic: [u8; 4]  │ version: u32 │ game_id: u32 │ num_steps:u32│
//! ├─────────────────┼──────────────┼──────────────┼──────────────┤
//! │ obs_dtype: u32  │ channels: u32│ height: u32  │ width: u32   │
//! ├─────────────────┼──────────────┼──────────────┼──────────────┤
//! │ action_dim: u32 │ num_players:u32│stride_bytes:u64│ RESERVED │
//! ├─────────────────┴──────────────┴──────────────┴──────────────┤
//! │                    CONTIGUOUS STEP RECORDS                   │
//! ├──────────────────────────────────────────────────────────────┤
//! │ Step 0:                                                      │
//! │   observation:      [f32; C * H * W]                         │
//! │   action_mask:      [u8; (action_dim + 7) / 8]               │
//! │   policy_target:    [f32; action_dim]   (visit distribution) │
//! │   value_target:     [f32; num_players]  (game return vector) │
//! │   action_taken:     u32                                      │
//! │   reward_target:    [f32; num_players]  (intermediate return)│
//! ├──────────────────────────────────────────────────────────────┤
//! │ Step 1: ...                                                  │
//! └──────────────────────────────────────────────────────────────┘
//! ```

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

/// Magic 4-byte header identifier: `b"MCTS"`.
pub const MAGIC: [u8; 4] = *b"MCTS";
/// Current protocol version.
pub const PROTOCOL_VERSION: u32 = 1;
/// Fixed header byte size.
pub const HEADER_SIZE: usize = 64;

/// Fixed 64-byte metadata header written at the beginning of every `.bin` chunk file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkHeader {
    /// Magic signature, must equal [`MAGIC`].
    pub magic: [u8; 4],
    /// Protocol version, must equal [`PROTOCOL_VERSION`].
    pub version: u32,
    /// Game identifier (e.g. 0 for TicTacToe, 1 for Connect4).
    pub game_id: u32,
    /// Number of step records stored in this chunk file.
    pub num_steps: u32,
    /// Observation data type: 0 for float32.
    pub obs_dtype: u32,
    /// Observation tensor channels ($C$).
    pub channels: u32,
    /// Observation tensor height ($H$).
    pub height: u32,
    /// Observation tensor width ($W$).
    pub width: u32,
    /// Total discrete action dimension ($|\mathcal{A}|$).
    pub action_dim: u32,
    /// Total participating players count ($N$).
    pub num_players: u32,
    /// Precomputed byte stride of every single step record.
    pub stride_bytes: u64,
    /// Reserved padding (16 bytes) zeroed out for future protocol expansion.
    pub reserved: [u8; 16],
}

impl ChunkHeader {
    /// Creates a new `ChunkHeader` with calculated `stride_bytes` and zeroed reserved fields.
    pub fn new(
        game_id: u32,
        num_steps: u32,
        channels: u32,
        height: u32,
        width: u32,
        action_dim: u32,
        num_players: u32,
    ) -> Self {
        let stride = Self::compute_stride(channels, height, width, action_dim, num_players);
        Self {
            magic: MAGIC,
            version: PROTOCOL_VERSION,
            game_id,
            num_steps,
            obs_dtype: 0,
            channels,
            height,
            width,
            action_dim,
            num_players,
            stride_bytes: stride as u64,
            reserved: [0u8; 16],
        }
    }

    /// Computes the exact byte stride of a single step record.
    #[inline]
    pub fn compute_stride(
        channels: u32,
        height: u32,
        width: u32,
        action_dim: u32,
        num_players: u32,
    ) -> usize {
        let obs_bytes = (channels * height * width * 4) as usize;
        let mask_bytes = (((action_dim + 31) / 32) * 4) as usize;
        let policy_bytes = (action_dim * 4) as usize;
        let value_bytes = (num_players * 4) as usize;
        let action_bytes = 4;
        let reward_bytes = (num_players * 4) as usize;

        obs_bytes + mask_bytes + policy_bytes + value_bytes + action_bytes + reward_bytes
    }

    /// Serializes the 64-byte header into `writer` in Little-Endian byte order.
    pub fn write_to<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(&self.magic)?;
        writer.write_u32::<LittleEndian>(self.version)?;
        writer.write_u32::<LittleEndian>(self.game_id)?;
        writer.write_u32::<LittleEndian>(self.num_steps)?;
        writer.write_u32::<LittleEndian>(self.obs_dtype)?;
        writer.write_u32::<LittleEndian>(self.channels)?;
        writer.write_u32::<LittleEndian>(self.height)?;
        writer.write_u32::<LittleEndian>(self.width)?;
        writer.write_u32::<LittleEndian>(self.action_dim)?;
        writer.write_u32::<LittleEndian>(self.num_players)?;
        writer.write_u64::<LittleEndian>(self.stride_bytes)?;
        writer.write_all(&self.reserved)?;
        Ok(())
    }

    /// Deserializes a `ChunkHeader` from `reader`.
    pub fn read_from<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid magic bytes: {:?}", magic),
            ));
        }

        let version = reader.read_u32::<LittleEndian>()?;
        if version != PROTOCOL_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported protocol version: {version}"),
            ));
        }

        let game_id = reader.read_u32::<LittleEndian>()?;
        let num_steps = reader.read_u32::<LittleEndian>()?;
        let obs_dtype = reader.read_u32::<LittleEndian>()?;
        let channels = reader.read_u32::<LittleEndian>()?;
        let height = reader.read_u32::<LittleEndian>()?;
        let width = reader.read_u32::<LittleEndian>()?;
        let action_dim = reader.read_u32::<LittleEndian>()?;
        let num_players = reader.read_u32::<LittleEndian>()?;
        let stride_bytes = reader.read_u64::<LittleEndian>()?;
        let mut reserved = [0u8; 16];
        reader.read_exact(&mut reserved)?;

        Ok(Self {
            magic,
            version,
            game_id,
            num_steps,
            obs_dtype,
            channels,
            height,
            width,
            action_dim,
            num_players,
            stride_bytes,
            reserved,
        })
    }
}

/// Single trajectory step transition record.
#[derive(Debug, Clone, PartialEq)]
pub struct StepRecord {
    /// Planar float observation features ($C \times H \times W$).
    pub observation: Vec<f32>,
    /// Bitpacked legal action mask ($1 = \text{legal}, 0 = \text{illegal}$).
    pub action_mask: Vec<u8>,
    /// MCTS visit distribution policy target ($\boldsymbol{\pi} \in \Delta^{|\mathcal{A}|}$).
    pub policy_target: Vec<f32>,
    /// Game outcome return target vector ($\mathbf{z} \in [-1, 1]^N$).
    pub value_target: Vec<f32>,
    /// The discrete action taken by the agent.
    pub action_taken: u32,
    /// Immediate step reward vector ($\mathbf{r} \in \mathbb{R}^N$).
    pub reward_target: Vec<f32>,
}

impl StepRecord {
    /// Serializes this step record into `writer` in Little-Endian byte order.
    pub fn write_to<W: Write>(&self, mut writer: W) -> io::Result<()> {
        for &f in &self.observation {
            writer.write_f32::<LittleEndian>(f)?;
        }
        writer.write_all(&self.action_mask)?;
        for &p in &self.policy_target {
            writer.write_f32::<LittleEndian>(p)?;
        }
        for &v in &self.value_target {
            writer.write_f32::<LittleEndian>(v)?;
        }
        writer.write_u32::<LittleEndian>(self.action_taken)?;
        for &r in &self.reward_target {
            writer.write_f32::<LittleEndian>(r)?;
        }
        Ok(())
    }

    /// Deserializes a single step record from `reader` according to `header` dimensions.
    pub fn read_from<R: Read>(mut reader: R, header: &ChunkHeader) -> io::Result<Self> {
        let obs_len = (header.channels * header.height * header.width) as usize;
        let mut observation = Vec::with_capacity(obs_len);
        for _ in 0..obs_len {
            observation.push(reader.read_f32::<LittleEndian>()?);
        }

        let mask_len = (((header.action_dim + 31) / 32) * 4) as usize;
        let mut action_mask = vec![0u8; mask_len];
        reader.read_exact(&mut action_mask)?;

        let policy_len = header.action_dim as usize;
        let mut policy_target = Vec::with_capacity(policy_len);
        for _ in 0..policy_len {
            policy_target.push(reader.read_f32::<LittleEndian>()?);
        }

        let val_len = header.num_players as usize;
        let mut value_target = Vec::with_capacity(val_len);
        for _ in 0..val_len {
            value_target.push(reader.read_f32::<LittleEndian>()?);
        }

        let action_taken = reader.read_u32::<LittleEndian>()?;

        let mut reward_target = Vec::with_capacity(val_len);
        for _ in 0..val_len {
            reward_target.push(reader.read_f32::<LittleEndian>()?);
        }

        Ok(Self {
            observation,
            action_mask,
            policy_target,
            value_target,
            action_taken,
            reward_target,
        })
    }
}
