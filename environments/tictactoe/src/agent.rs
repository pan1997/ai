//! TicTacToe agent implementations: Human, Random, Tactical Heuristic, and MCTS.

use crate::dynamics::TicTacToeDynamics;
use crate::evaluator::{RolloutEvaluator, UniformEvaluator};
use crate::game::TicTacToeState;
use mcts_engine::backup::VectorBackup;
use mcts_engine::scheduler::SequentialScheduler;
use mcts_engine::selection::{MultiAgentPuctSelection, MultiAgentPuctStats};
use mcts_engine::tree_store::TreeStore;
use mcts_traits::{Agent, AgentId, Model, TensorRepresentable};
use rand::seq::SliceRandom;
use std::io::{self, BufRead, Write};

/// Dynamic boxed TicTacToe agent.
pub type BoxAgent = Box<dyn Agent<TicTacToeState, usize>>;

/// Interactive human player prompting for moves via standard input.
pub struct HumanAgent {
    name: String,
}

impl HumanAgent {
    /// Creates a new human player with the specified name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Agent<TicTacToeState, usize> for HumanAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &TicTacToeState) -> usize {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);

        loop {
            print!(
                "{} ({:?}), enter square {:?}: ",
                self.name, state.current_player, legal
            );
            let _ = stdout.flush();

            let mut input = String::new();
            if stdin.lock().read_line(&mut input).is_err() {
                println!("Error reading input, please try again.");
                continue;
            }

            let trimmed = input.trim();
            match trimmed.parse::<usize>() {
                Ok(cell) => {
                    if cell >= 9 {
                        println!("Cell {cell} is out of bounds (0..9).");
                    } else if !state.is_empty(cell) {
                        println!("Cell {cell} is already occupied! Choose another cell.");
                    } else {
                        return cell;
                    }
                }
                Err(_) => {
                    println!("Invalid input '{trimmed}'. Please enter an integer (0..9).");
                }
            }
        }
    }
}

/// Baseline agent selecting uniformly at random among legal actions.
pub struct RandomAgent {
    name: String,
}

impl RandomAgent {
    /// Creates a new random player with the specified name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        Self::new("Random Agent")
    }
}

impl Agent<TicTacToeState, usize> for RandomAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &TicTacToeState) -> usize {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        let mut rng = rand::thread_rng();
        *legal
            .choose(&mut rng)
            .expect("RandomAgent: no legal actions available in state")
    }
}

/// Tactical heuristic agent:
/// 1. Takes immediate winning move if available.
/// 2. Blocks immediate opponent winning move if available.
/// 3. Takes center (4) if empty.
/// 4. Takes corners (0, 2, 6, 8) if empty.
/// 5. Takes remaining squares randomly.
pub struct TacticalAgent {
    name: String,
}

impl TacticalAgent {
    /// Creates a new tactical agent.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for TacticalAgent {
    fn default() -> Self {
        Self::new("Tactical Agent")
    }
}

impl Agent<TicTacToeState, usize> for TacticalAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &TicTacToeState) -> usize {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.is_empty() {
            panic!("TacticalAgent: no legal actions in state");
        }

        let me = state.current_player;
        let opp = me.other();

        // 1. Can I win immediately?
        for &action in &legal {
            let mut next = state.clone();
            next.apply_action(action);
            if next.check_winner() == Some(me) {
                return action;
            }
        }

        // 2. Can opponent win immediately? Block it.
        for &action in &legal {
            let mut next = state.clone();
            next.current_player = opp;
            next.apply_action(action);
            if next.check_winner() == Some(opp) {
                return action;
            }
        }

        // 3. Center
        if legal.contains(&4) {
            return 4;
        }

        // 4. Corners
        let mut rng = rand::thread_rng();
        let corners: Vec<usize> = [0, 2, 6, 8]
            .iter()
            .copied()
            .filter(|c| legal.contains(c))
            .collect();
        if !corners.is_empty() {
            return *corners.choose(&mut rng).unwrap();
        }

        // 5. Random legal
        *legal.choose(&mut rng).unwrap()
    }
}

/// MCTS Agent playing TicTacToe using configurable model and simulations.
pub struct MctsAgent<M> {
    name: String,
    model: M,
    num_simulations: usize,
    c_puct: f32,
}

impl<M> MctsAgent<M> {
    /// Creates a new MCTS agent.
    pub fn new(name: impl Into<String>, model: M, num_simulations: usize, c_puct: f32) -> Self {
        Self {
            name: name.into(),
            model,
            num_simulations,
            c_puct,
        }
    }
}

impl<M: Model<TicTacToeState>> Agent<TicTacToeState, usize> for MctsAgent<M> {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &TicTacToeState) -> usize {
        let mut legal = Vec::new();
        state.legal_actions(&mut legal);
        if legal.len() <= 1 {
            return legal[0];
        }

        let mut tree = TreeStore::with_capacity(
            self.num_simulations + 10,
            (self.num_simulations + 10) * 9,
            MultiAgentPuctStats::<2>::new(),
        );

        let root_agent = AgentId(state.current_player.index() as u32);
        let root = tree.insert_root(root_agent);

        let dynamics = TicTacToeDynamics;
        let selection = MultiAgentPuctSelection::<2> {
            c_puct: self.c_puct,
        };
        let backup = VectorBackup::<2>::default();

        SequentialScheduler.search(
            &mut tree,
            &dynamics,
            &self.model,
            &selection,
            &backup,
            root,
            state,
            self.num_simulations,
        );

        // Select the child edge with the most visits
        let mut best_action = legal[0];
        let mut max_visits = -1;

        for edge in tree.child_edges(root) {
            let visits = tree.stats.visits[edge.as_usize()];
            if visits as i64 > max_visits {
                max_visits = visits as i64;
                best_action = *tree.edge_action(edge);
            }
        }

        best_action
    }
}

/// AlphaZero agent powered by an ONNX neural network and MCTS.
pub struct AlphaZeroAgent {
    name: String,
    _dispatcher: mcts_onnx::InferenceDispatcher,
    inner: MctsAgent<mcts_onnx::OnnxModelClient<TicTacToeState>>,
}

impl AlphaZeroAgent {
    /// Loads an AlphaZero agent from an ONNX model file.
    pub fn from_onnx_file(
        name: impl Into<String>,
        model_path: &str,
        num_simulations: usize,
        c_puct: f32,
    ) -> Result<Self, String> {
        let session = mcts_onnx::ort::session::Session::builder()
            .map_err(|e| format!("Failed to create ORT session builder: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load ONNX model from '{model_path}': {e}"))?;

        let config = mcts_onnx::BatcherConfig {
            channels: TicTacToeState::CHANNELS,
            height: TicTacToeState::HEIGHT,
            width: TicTacToeState::WIDTH,
            max_batch_size: 64,
            max_latency: std::time::Duration::from_millis(2),
        };

        let dispatcher = mcts_onnx::InferenceDispatcher::new(session, config);
        let client = mcts_onnx::OnnxModelClient::new(dispatcher.request_sender(), |s: &TicTacToeState| {
            let mut legal = Vec::new();
            s.legal_actions(&mut legal);
            legal
        });

        let name_str = name.into();
        let inner = MctsAgent::new(name_str.clone(), client, num_simulations, c_puct);

        Ok(Self {
            name: name_str,
            _dispatcher: dispatcher,
            inner,
        })
    }
}

impl Agent<TicTacToeState, usize> for AlphaZeroAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn select_action(&mut self, state: &TicTacToeState) -> usize {
        self.inner.select_action(state)
    }
}

/// Parses an agent specification string into a boxed agent instance.
///
/// Supported specs:
/// - `"human"` or `"human:Name"`: Interactive human player.
/// - `"random"` or `"random:Name"`: Uniform random player.
/// - `"tactical"`: Tactical 1-ply heuristic player.
/// - `"mcts:<sims>"`: MCTS player using rollout evaluator.
/// - `"mcts:uniform:<sims>"`: MCTS player using uniform evaluator.
/// - `"mcts:rollout:<sims>"`: MCTS player using rollout evaluator.
/// - `"alphazero:<model_path>[:<sims>]"`: AlphaZero player using ONNX model.
pub fn parse_agent_spec(spec: &str) -> Result<BoxAgent, String> {
    let parts: Vec<&str> = spec.split(':').collect();
    match parts[0] {
        "human" => {
            let name = parts.get(1).copied().unwrap_or("Human");
            Ok(Box::new(HumanAgent::new(name)))
        }
        "random" => {
            let name = parts.get(1).copied().unwrap_or("Random");
            Ok(Box::new(RandomAgent::new(name)))
        }
        "tactical" => {
            let name = parts.get(1).copied().unwrap_or("Tactical");
            Ok(Box::new(TacticalAgent::new(name)))
        }
        "mcts" => {
            if parts.len() == 2 {
                let sims_str = parts[1];
                let sims = sims_str
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid simulations count '{sims_str}': {e}"))?;
                Ok(Box::new(MctsAgent::new(
                    format!("MCTS(Rollout,{sims})"),
                    RolloutEvaluator::default(),
                    sims,
                    1.414,
                )))
            } else if parts.len() == 3 {
                let eval_type = parts[1];
                let sims_str = parts[2];
                let sims = sims_str
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid simulations count '{sims_str}': {e}"))?;
                match eval_type {
                    "uniform" => Ok(Box::new(MctsAgent::new(
                        format!("MCTS(Uniform,{sims})"),
                        UniformEvaluator,
                        sims,
                        1.414,
                    ))),
                    "rollout" => Ok(Box::new(MctsAgent::new(
                        format!("MCTS(Rollout,{sims})"),
                        RolloutEvaluator::default(),
                        sims,
                        1.414,
                    ))),
                    _ => Err(format!(
                        "Unknown evaluator type '{eval_type}'. Expected 'uniform' or 'rollout'"
                    )),
                }
            } else {
                Err(format!("Invalid MCTS spec '{spec}'. Expected 'mcts:<sims>' or 'mcts:<eval>:<sims>'"))
            }
        }
        "alphazero" => {
            if parts.len() < 2 {
                return Err(format!(
                    "Invalid AlphaZero spec '{spec}'. Expected 'alphazero:<model_path>[:sims]'"
                ));
            }
            let model_path = parts[1];
            let sims = if parts.len() >= 3 {
                parts[2]
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid simulations count '{s}': {e}", s = parts[2]))?
            } else {
                50
            };
            let agent = AlphaZeroAgent::from_onnx_file(format!("AlphaZero({sims})"), model_path, sims, 1.414)?;
            Ok(Box::new(agent))
        }
        _ => Err(format!("Unknown agent type '{spec}'")),
    }
}
