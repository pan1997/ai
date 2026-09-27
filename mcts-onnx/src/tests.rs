use crate::spool::TrajectorySpooler;
use crate::wire::{ChunkHeader, HEADER_SIZE, StepRecord};
use std::fs::File;
use std::io::Cursor;

#[test]
fn test_header_serialization_roundtrip() {
    let header = ChunkHeader::new(0, 100, 2, 3, 3, 9, 2);
    assert_eq!(header.channels, 2);
    assert_eq!(header.height, 3);
    assert_eq!(header.width, 3);
    assert_eq!(header.action_dim, 9);
    assert_eq!(header.num_players, 2);

    let mut buf = Vec::new();
    header.write_to(&mut buf).unwrap();
    assert_eq!(buf.len(), HEADER_SIZE);

    let decoded = ChunkHeader::read_from(Cursor::new(&buf)).unwrap();
    assert_eq!(header, decoded);
}

#[test]
fn test_step_record_serialization_roundtrip() {
    let header = ChunkHeader::new(0, 1, 2, 3, 3, 9, 2);

    let record = StepRecord {
        observation: vec![1.0; 18],
        action_mask: vec![0b10101010, 0b00000001, 0, 0], // 4 bytes (32-bit aligned) for 9 actions
        policy_target: vec![1.0 / 9.0; 9],
        value_target: vec![1.0, -1.0],
        action_taken: 4,
        reward_target: vec![0.0, 0.0],
    };

    let mut buf = Vec::new();
    record.write_to(&mut buf).unwrap();
    assert_eq!(buf.len(), header.stride_bytes as usize);

    let decoded = StepRecord::read_from(Cursor::new(&buf), &header).unwrap();
    assert_eq!(record, decoded);
}

#[test]
fn test_spooler_chunking_and_atomic_rename() {
    let temp_dir = std::env::temp_dir().join(format!("spool_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let mut spooler = TrajectorySpooler::new(&temp_dir, 1, 5, 0, 2, 3, 3, 9, 2).unwrap();

    let record = StepRecord {
        observation: vec![0.5; 18],
        action_mask: vec![0xFF, 0x01, 0x00, 0x00],
        policy_target: vec![0.1; 9],
        value_target: vec![0.5, -0.5],
        action_taken: 0,
        reward_target: vec![0.0, 0.0],
    };

    for i in 0..4 {
        let chunk = spooler.push(record.clone()).unwrap();
        assert!(chunk.is_none(), "Step {i} should not have triggered flush");
    }
    assert_eq!(spooler.len(), 4);

    // 5th push triggers flush
    let chunk = spooler.push(record.clone()).unwrap();
    assert!(chunk.is_some(), "5th step should have triggered flush");
    assert_eq!(spooler.len(), 0);

    let chunk_path = chunk.unwrap();
    assert!(chunk_path.exists());
    assert!(chunk_path.to_string_lossy().ends_with(".bin"));

    // Verify reading the chunk back
    let file = File::open(&chunk_path).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let header = ChunkHeader::read_from(&mut reader).unwrap();
    assert_eq!(header.num_steps, 5);

    for _ in 0..5 {
        let step = StepRecord::read_from(&mut reader, &header).unwrap();
        assert_eq!(step.observation, vec![0.5; 18]);
        assert_eq!(step.action_taken, 0);
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_tabular_model_evaluate_and_serialization() {
    use crate::tabular::TabularModel;
    use mcts_traits::{BatchedModel, Model};

    let mut model = TabularModel::new(3, 1);
    model.set(0, vec![2.0, 0.0, -1.0], vec![0.75]);
    model.set(1, vec![0.0, 1.0, 0.0], vec![-0.5]);

    // Test Model::evaluate
    let eval0 = model.evaluate(&0);
    assert_eq!(eval0.values, vec![0.75]);
    assert_eq!(eval0.priors.len(), 3);
    assert!(eval0.priors[0] > eval0.priors[1]);
    assert!(eval0.priors[1] > eval0.priors[2]);

    // Test BatchedModel::evaluate_batch
    let batch = model.evaluate_batch(&[&0, &1]);
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].values, vec![0.75]);
    assert_eq!(batch[1].values, vec![-0.5]);

    // Test serialization roundtrip
    let temp_path = std::env::temp_dir().join(format!("tab_test_{}.bin", std::process::id()));
    model.save_to_file(&temp_path).unwrap();

    let loaded = TabularModel::load_from_file(&temp_path).unwrap();
    assert_eq!(model, loaded);

    let _ = std::fs::remove_file(&temp_path);
}

#[derive(Debug, Clone, PartialEq)]
struct MockEnv {
    step: usize,
}

impl mcts_traits::TensorRepresentable for MockEnv {
    const CHANNELS: usize = 1;
    const HEIGHT: usize = 1;
    const WIDTH: usize = 1;

    fn encode_tensor(&self, out: &mut [f32]) {
        out[0] = self.step as f32;
    }
}

struct MockDynamics;

impl mcts_traits::AgentDynamics for MockDynamics {
    type State = MockEnv;
    type Action = usize;
    type Reward = [f32; 2];
    type StepDelta = ();

    fn initial(&self) -> Self::State {
        MockEnv { step: 0 }
    }

    fn current_agent(&self, s: &Self::State) -> mcts_traits::AgentId {
        mcts_traits::AgentId((s.step % 2) as u32)
    }

    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        out.clear();
        if s.step < 2 {
            out.push(0);
            out.push(1);
        }
    }

    fn step(
        &self,
        s: &mut Self::State,
        _action: &Self::Action,
    ) -> mcts_traits::StepOutcome<Self::Reward, Self::StepDelta> {
        s.step += 1;
        let term = s.step >= 2;
        let rew = if term { [1.0, -1.0] } else { [0.0, 0.0] };
        mcts_traits::StepOutcome::new(rew, term)
    }
}

impl mcts_traits::BatchedAgentDynamics for MockDynamics {
    fn step_batch(
        &self,
        states: &mut [Self::State],
        actions: &[Self::Action],
        outcomes: &mut Vec<mcts_traits::StepOutcome<Self::Reward, Self::StepDelta>>,
    ) {
        mcts_traits::default_step_batch(self, states, actions, outcomes);
    }
}

impl crate::selfplay::SelfPlayEnv for MockEnv {
    type Dynamics = MockDynamics;

    fn dynamics(&self) -> Self::Dynamics {
        MockDynamics
    }

    fn initial() -> Self {
        MockEnv { step: 0 }
    }

    fn legal_actions(&self, out: &mut Vec<usize>) {
        out.clear();
        if self.step < 2 {
            out.push(0);
            out.push(1);
        }
    }

    fn action_mask(&self, out: &mut [u8]) {
        out.fill(0);
        if self.step < 2 {
            out[0] = 0b00000011;
        }
    }

    fn apply_action(&mut self, _action: usize) {
        self.step += 1;
    }

    fn is_terminal(&self) -> bool {
        self.step >= 2
    }

    fn terminal_returns(&self) -> [f32; 2] {
        [1.0, -1.0]
    }

    fn current_player_index(&self) -> usize {
        self.step % 2
    }
}

struct MockBatchedModel;

impl mcts_traits::Model<MockEnv> for MockBatchedModel {
    fn evaluate(&self, _s: &MockEnv) -> mcts_traits::Evaluation {
        mcts_traits::Evaluation::vector(vec![0.5, 0.5], vec![0.0, 0.0])
    }
}

impl mcts_traits::BatchedModel<MockEnv> for MockBatchedModel {
    fn evaluate_batch(&self, states: &[&MockEnv]) -> Vec<mcts_traits::Evaluation> {
        states.iter().map(|s| mcts_traits::Model::evaluate(self, s)).collect()
    }
}

#[test]
fn test_execute_episodes_with_multigamescheduler() {
    let temp_dir = std::env::temp_dir().join(format!("spool_mg_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let mut spooler = TrajectorySpooler::new(&temp_dir, 0, 100, 0, 1, 1, 1, 2, 2).unwrap();
    let config = crate::selfplay::SelfPlayConfig {
        spool_dir: temp_dir.clone(),
        model_path: None,
        num_games: 7,
        num_sims: 5,
        parallel_games: 3,
        c_puct: 1.414,
        worker_id: 0,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        game_id: 0,
        action_dim: 2,
        num_players: 2,
        chunk_size: 100,
    };

    let model = MockBatchedModel;
    let (completed_games, total_steps) =
        crate::selfplay::execute_episodes(&model, &mut spooler, &config);

    assert_eq!(completed_games, 7);
    assert_eq!(total_steps, 14); // 7 games * 2 steps each
    assert_eq!(spooler.len(), 14);

    let _ = std::fs::remove_dir_all(&temp_dir);
}
