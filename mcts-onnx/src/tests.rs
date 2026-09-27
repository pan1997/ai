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
