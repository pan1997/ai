//! Baseline and heuristic evaluation models for Blokus MCTS planning.

use crate::dynamics::BlokusDynamics;
use crate::game::{BlokusAction, BlokusState};
use crate::pieces::piece_size;
use mcts_traits::{AgentDynamics, Evaluation, Model};

/// Baseline evaluation model assigning uniform prior probability across legal actions.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformEvaluator;

impl<const B: usize, const P: usize> Model<BlokusState<B, P>> for UniformEvaluator {
    fn evaluate(&self, s: &BlokusState<B, P>) -> Evaluation {
        let mut legal = Vec::new();
        s.legal_actions(&mut legal);
        let n = legal.len();
        let priors = if n > 0 {
            vec![1.0 / (n as f32); n]
        } else {
            Vec::new()
        };
        Evaluation {
            priors,
            values: vec![0.0f32; P],
        }
    }
}

/// Domain heuristic evaluator prioritizing larger pieces and corner mobility.
///
/// Blokus opening and midgame strategy emphasizes placing pentominoes first to avoid getting
/// stuck with unplayable large pieces, while maximizing one's own corner count and constricting
/// opponent expansion.
#[derive(Debug, Clone, Copy, Default)]
pub struct AreaHeuristicEvaluator;

impl<const B: usize, const P: usize> Model<BlokusState<B, P>> for AreaHeuristicEvaluator {
    fn evaluate(&self, s: &BlokusState<B, P>) -> Evaluation {
        let mut legal = Vec::new();
        s.legal_actions(&mut legal);
        let n = legal.len();
        if n == 0 {
            return Evaluation {
                priors: Vec::new(),
                values: vec![0.0f32; P],
            };
        }

        // Action priors: Weight by piece size squared to favor larger polyominoes
        let mut weights: Vec<f32> = Vec::with_capacity(n);
        let mut total_weight = 0.0f32;

        for action in &legal {
            let w = match action {
                BlokusAction::Place { piece_id, .. } => {
                    let sz = piece_size(*piece_id) as f32;
                    sz * sz
                }
                BlokusAction::Pass => 0.1f32,
            };
            weights.push(w);
            total_weight += w;
        }

        let priors = if total_weight > 0.0 {
            weights.into_iter().map(|w| w / total_weight).collect()
        } else {
            vec![1.0 / (n as f32); n]
        };

        // State value: Relative square difference and corner advantage
        let mut scores = [0.0f32; P];
        let mut corner_buf = Vec::new();

        for p in 0..P {
            let placed = 89.0 - (s.unplaced_squares(p) as f32);
            s.find_valid_corners(p, &mut corner_buf);
            let corner_bonus = (corner_buf.len() as f32) * 1.5;
            scores[p] = placed + corner_bonus;
        }

        let mean_score = scores.iter().sum::<f32>() / (P as f32);
        let values = scores
            .iter()
            .map(|&sc| ((sc - mean_score) / 50.0).clamp(-1.0, 1.0))
            .collect();

        Evaluation { priors, values }
    }
}

/// Rollout evaluator estimating state value vectors via random simulation playouts.
pub struct RolloutEvaluator<const B: usize = 20, const P: usize = 4> {
    /// Number of independent simulated playouts to run from the evaluated state.
    pub num_rollouts: usize,
    /// Maximum search depth (turn steps) per simulation playout.
    pub max_depth: usize,
}

impl<const B: usize, const P: usize> RolloutEvaluator<B, P> {
    /// Creates a new `RolloutEvaluator`.
    pub fn new(num_rollouts: usize, max_depth: usize) -> Self {
        Self {
            num_rollouts,
            max_depth,
        }
    }
}

impl<const B: usize, const P: usize> Default for RolloutEvaluator<B, P> {
    fn default() -> Self {
        Self::new(3, 30)
    }
}

impl<const B: usize, const P: usize> Model<BlokusState<B, P>> for RolloutEvaluator<B, P> {
    fn evaluate(&self, s: &BlokusState<B, P>) -> Evaluation {
        let env = BlokusDynamics::<B, P>;
        let mut legal_actions = Vec::new();
        s.legal_actions(&mut legal_actions);
        let num_actions = legal_actions.len();
        if num_actions == 0 {
            return Evaluation {
                priors: Vec::new(),
                values: vec![0.0f32; P],
            };
        }

        let priors = vec![1.0 / (num_actions as f32); num_actions];
        let mut total_rewards = [0.0f32; P];
        let mut rng = rand::thread_rng();
        let mut actions_buf = Vec::new();

        for _ in 0..self.num_rollouts {
            let mut current = s.clone();
            let mut depth = 0;

            while depth < self.max_depth && !current.is_terminal() {
                current.legal_actions(&mut actions_buf);
                if actions_buf.is_empty() {
                    break;
                }
                let pick = rand::Rng::gen_range(&mut rng, 0..actions_buf.len());
                let outcome = env.step(&mut current, &actions_buf[pick]);

                if outcome.terminated {
                    for (tot, &rew) in total_rewards.iter_mut().zip(outcome.reward.iter()) {
                        *tot += rew;
                    }
                    break;
                }
                depth += 1;
            }

            if !current.is_terminal() {
                // If depth limit reached without game over, approximate with relative scores
                let mut scores = [0i32; P];
                for p in 0..P {
                    scores[p] = current.score(p);
                }
                let intermediate = crate::dynamics::compute_rank_rewards(&scores);
                for (tot, &rew) in total_rewards.iter_mut().zip(intermediate.iter()) {
                    *tot += rew;
                }
            }
        }

        let m = self.num_rollouts.max(1) as f32;
        let values = total_rewards.iter().map(|&r| r / m).collect();

        Evaluation { priors, values }
    }
}

/// Heuristic Utility evaluator assigning action priors proportionally to their 1-ply tactical score (size + corners).
///
/// Bridges the gap between the standalone greedy heuristic and tree search:
/// High-corner, large-piece moves receive exponentially or proportionally higher prior probabilities,
/// allowing MCTS to focus its search tree on high-potential tactical branches.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicUtilityEvaluator {
    /// Temperature scaling for priors. Default is 4.0.
    pub temperature: f32,
}

impl HeuristicUtilityEvaluator {
    /// Creates a new `HeuristicUtilityEvaluator`.
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl<const B: usize, const P: usize> Model<BlokusState<B, P>> for HeuristicUtilityEvaluator {
    fn evaluate(&self, s: &BlokusState<B, P>) -> Evaluation {
        let mut legal = Vec::new();
        s.legal_actions(&mut legal);
        let n = legal.len();
        if n == 0 {
            return Evaluation {
                priors: Vec::new(),
                values: vec![0.0f32; P],
            };
        }

        let player = s.current_player as usize;
        let mut corner_buf = Vec::new();
        let mut scores = Vec::with_capacity(n);

        for action in &legal {
            let score = match action {
                BlokusAction::Pass => 0.1f32,
                BlokusAction::Place { piece_id, .. } => {
                    let mut next = s.clone();
                    let _ = next.apply_action(action);
                    next.find_valid_corners(player, &mut corner_buf);
                    let corners = corner_buf.len() as f32;
                    let size = piece_size(*piece_id) as f32;
                    // Score = 10 * piece size + 2 * corners created
                    (10.0 * size + 2.0 * corners).max(0.1)
                }
            };
            scores.push(score);
        }

        // Softmax with temperature
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let temp = if self.temperature > 0.0 { self.temperature } else { 4.0 };
        let exps: Vec<f32> = scores.iter().map(|&sc| ((sc - max_score) / temp).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let priors = if sum_exp > 0.0 {
            exps.into_iter().map(|e| e / sum_exp).collect()
        } else {
            vec![1.0 / (n as f32); n]
        };

        // Static leaf values based on placed squares and corners
        let mut player_scores = [0.0f32; P];
        for p in 0..P {
            let placed = 89.0 - (s.unplaced_squares(p) as f32);
            s.find_valid_corners(p, &mut corner_buf);
            let corner_bonus = (corner_buf.len() as f32) * 1.5;
            player_scores[p] = placed + corner_bonus;
        }

        let mean_score = player_scores.iter().sum::<f32>() / (P as f32);
        let values = player_scores
            .iter()
            .map(|&sc| ((sc - mean_score) / 50.0).clamp(-1.0, 1.0))
            .collect();

        Evaluation { priors, values }
    }
}

/// Rollout evaluator running simulation playouts guided by the 1-ply corner heuristic.
pub struct HeuristicRolloutEvaluator<const B: usize = 20, const P: usize = 4> {
    /// Number of simulated playouts per leaf node.
    pub num_rollouts: usize,
    /// Maximum search depth per rollout.
    pub max_depth: usize,
    /// Probability of picking a random move instead of greedy heuristic (e.g. 0.15).
    pub epsilon: f32,
}

impl<const B: usize, const P: usize> HeuristicRolloutEvaluator<B, P> {
    /// Creates a new `HeuristicRolloutEvaluator`.
    pub fn new(num_rollouts: usize, max_depth: usize, epsilon: f32) -> Self {
        Self {
            num_rollouts,
            max_depth,
            epsilon,
        }
    }
}

impl<const B: usize, const P: usize> Default for HeuristicRolloutEvaluator<B, P> {
    fn default() -> Self {
        Self::new(2, 20, 0.15)
    }
}

impl<const B: usize, const P: usize> Model<BlokusState<B, P>> for HeuristicRolloutEvaluator<B, P> {
    fn evaluate(&self, s: &BlokusState<B, P>) -> Evaluation {
        let env = BlokusDynamics::<B, P>;
        let mut legal_actions = Vec::new();
        s.legal_actions(&mut legal_actions);
        let num_actions = legal_actions.len();
        if num_actions == 0 {
            return Evaluation {
                priors: Vec::new(),
                values: vec![0.0f32; P],
            };
        }

        // Action priors: Weight by piece size squared
        let mut weights = Vec::with_capacity(num_actions);
        let mut total_weight = 0.0f32;
        for action in &legal_actions {
            let w = match action {
                BlokusAction::Place { piece_id, .. } => {
                    let sz = piece_size(*piece_id) as f32;
                    sz * sz
                }
                BlokusAction::Pass => 0.1f32,
            };
            weights.push(w);
            total_weight += w;
        }
        let priors = if total_weight > 0.0 {
            weights.into_iter().map(|w| w / total_weight).collect()
        } else {
            vec![1.0 / (num_actions as f32); num_actions]
        };

        let mut total_rewards = [0.0f32; P];
        let mut rng = rand::thread_rng();
        let mut actions_buf = Vec::new();
        let mut corner_buf = Vec::new();

        for _ in 0..self.num_rollouts {
            let mut current = s.clone();
            let mut depth = 0;

            while depth < self.max_depth && !current.is_terminal() {
                current.legal_actions(&mut actions_buf);
                if actions_buf.is_empty() {
                    break;
                }

                let p = current.current_player as usize;
                let action = if rand::Rng::gen_range(&mut rng, 0.0f32..1.0f32) < self.epsilon
                    || actions_buf.len() <= 1
                {
                    let pick = rand::Rng::gen_range(&mut rng, 0..actions_buf.len());
                    actions_buf[pick]
                } else {
                    // Subsample candidates for fast tournament playout selection
                    let sample_size = actions_buf.len().min(6);
                    let mut best_score = i32::MIN;
                    let mut best_act = actions_buf[0];

                    for _ in 0..sample_size {
                        let pick = rand::Rng::gen_range(&mut rng, 0..actions_buf.len());
                        let cand = actions_buf[pick];
                        let score = match cand {
                            BlokusAction::Pass => -1000,
                            BlokusAction::Place { piece_id, .. } => {
                                let mut next = current.clone();
                                let _ = next.apply_action(&cand);
                                next.find_valid_corners(p, &mut corner_buf);
                                let corners = corner_buf.len() as i32;
                                let size = piece_size(piece_id) as i32;
                                10 * size + 2 * corners
                            }
                        };
                        if score > best_score {
                            best_score = score;
                            best_act = cand;
                        }
                    }
                    best_act
                };

                let outcome = env.step(&mut current, &action);
                if outcome.terminated {
                    for (tot, &rew) in total_rewards.iter_mut().zip(outcome.reward.iter()) {
                        *tot += rew;
                    }
                    break;
                }
                depth += 1;
            }

            if !current.is_terminal() {
                let mut scores = [0i32; P];
                for p in 0..P {
                    scores[p] = current.score(p);
                }
                let intermediate = crate::dynamics::compute_rank_rewards(&scores);
                for (tot, &rew) in total_rewards.iter_mut().zip(intermediate.iter()) {
                    *tot += rew;
                }
            }
        }

        let m = self.num_rollouts.max(1) as f32;
        let values = total_rewards.iter().map(|&r| r / m).collect();

        Evaluation { priors, values }
    }
}

