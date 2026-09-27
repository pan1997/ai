//! Discrete tabular model for exact verification testing on `GraphEnv` and discrete MDPs.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use mcts_traits::{BatchedModel, Evaluation, Model};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic 4-byte header for serialized tabular models: `b"TABL"`.
pub const TABULAR_MAGIC: [u8; 4] = *b"TABL";

/// Discrete lookup-table model storing policy logits and value estimates per state index.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularModel {
    /// Action space cardinality.
    pub num_actions: usize,
    /// Participating player count.
    pub num_players: usize,
    /// Mapping from discrete state ID to policy logits vector of length `num_actions`.
    pub policy_logits: HashMap<u32, Vec<f32>>,
    /// Mapping from discrete state ID to value estimate vector of length `num_players`.
    pub values: HashMap<u32, Vec<f32>>,
}

impl TabularModel {
    /// Creates an empty `TabularModel` for `num_actions` and `num_players`.
    pub fn new(num_actions: usize, num_players: usize) -> Self {
        Self {
            num_actions,
            num_players,
            policy_logits: HashMap::new(),
            values: HashMap::new(),
        }
    }

    /// Sets or updates the policy logits and value vector for state `s`.
    pub fn set(&mut self, s: u32, logits: Vec<f32>, val: Vec<f32>) {
        assert_eq!(
            logits.len(),
            self.num_actions,
            "TabularModel::set: logits length mismatch"
        );
        assert_eq!(
            val.len(),
            self.num_players,
            "TabularModel::set: values length mismatch"
        );
        self.policy_logits.insert(s, logits);
        self.values.insert(s, val);
    }

    /// Computes normalized policy priors for state `s` using softmax.
    pub fn priors_for(&self, s: u32) -> Vec<f32> {
        if let Some(logits) = self.policy_logits.get(&s) {
            let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            if sum > 0.0 {
                return exps.into_iter().map(|e| e / sum).collect();
            }
        }
        vec![1.0 / (self.num_actions.max(1) as f32); self.num_actions]
    }

    /// Returns value estimates for state `s` (defaults to zero vector).
    pub fn values_for(&self, s: u32) -> Vec<f32> {
        self.values
            .get(&s)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.num_players])
    }

    /// Serializes this tabular model to `writer` in binary format.
    pub fn write_to<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(&TABULAR_MAGIC)?;
        writer.write_u32::<LittleEndian>(1)?; // version
        writer.write_u32::<LittleEndian>(self.policy_logits.len() as u32)?;
        writer.write_u32::<LittleEndian>(self.num_actions as u32)?;
        writer.write_u32::<LittleEndian>(self.num_players as u32)?;

        for (&state, logits) in &self.policy_logits {
            writer.write_u32::<LittleEndian>(state)?;
            for &l in logits {
                writer.write_f32::<LittleEndian>(l)?;
            }
            let vals = self.values.get(&state).cloned().unwrap_or_else(|| vec![0.0; self.num_players]);
            for v in vals {
                writer.write_f32::<LittleEndian>(v)?;
            }
        }
        Ok(())
    }

    /// Deserializes a tabular model from `reader`.
    pub fn read_from<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != TABULAR_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid tabular magic: {magic:?}"),
            ));
        }

        let version = reader.read_u32::<LittleEndian>()?;
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported tabular version: {version}"),
            ));
        }

        let num_entries = reader.read_u32::<LittleEndian>()? as usize;
        let num_actions = reader.read_u32::<LittleEndian>()? as usize;
        let num_players = reader.read_u32::<LittleEndian>()? as usize;

        let mut model = Self::new(num_actions, num_players);

        for _ in 0..num_entries {
            let state = reader.read_u32::<LittleEndian>()?;
            let mut logits = Vec::with_capacity(num_actions);
            for _ in 0..num_actions {
                logits.push(reader.read_f32::<LittleEndian>()?);
            }
            let mut values = Vec::with_capacity(num_players);
            for _ in 0..num_players {
                values.push(reader.read_f32::<LittleEndian>()?);
            }
            model.set(state, logits, values);
        }

        Ok(model)
    }

    /// Saves the tabular model to a file at `path`.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)?;
        writer.flush()
    }

    /// Loads the tabular model from a file at `path`.
    pub fn load_from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read_from(&mut reader)
    }
}

impl Model<u32> for TabularModel {
    fn evaluate(&self, s: &u32) -> Evaluation {
        Evaluation {
            priors: self.priors_for(*s),
            values: self.values_for(*s),
        }
    }
}

impl BatchedModel<u32> for TabularModel {
    fn evaluate_batch(&self, states: &[&u32]) -> Vec<Evaluation> {
        states.iter().map(|&s| self.evaluate(s)).collect()
    }
}
