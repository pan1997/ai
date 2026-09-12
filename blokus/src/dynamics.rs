//! Planning dynamics utilities and score ranking for Blokus.
//!
//! Standard alternating planning dynamics are provided universally by
//! [`TurnBasedDynamics<BlokusWorld<B, P>>`](mcts_traits::TurnBasedDynamics).

/// Computes normalized fractional rank rewards for $P$ players given their final scores.
///
/// Returns values in $[-1.0, 1.0]$ summing to $0.0$, gracefully handling multi-way ties.
pub fn compute_rank_rewards<const P: usize>(scores: &[i32; P]) -> [f32; P] {
    let mut rewards = [0.0f32; P];
    if P <= 1 {
        return rewards;
    }

    let denom = (P - 1) as f32;
    for i in 0..P {
        let mut strictly_lower = 0;
        let mut strictly_higher = 0;
        for j in 0..P {
            if i != j {
                if scores[i] > scores[j] {
                    strictly_lower += 1;
                } else if scores[i] < scores[j] {
                    strictly_higher += 1;
                }
            }
        }
        rewards[i] = (strictly_lower as f32 - strictly_higher as f32) / denom;
    }
    rewards
}
