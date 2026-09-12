//! Opponent policies for afterstate nodes in Monte Carlo Tree Search.
//!
//! In multi-player games, afterstate nodes represent intermediate decision points
//! where an opponent acts between turns of the primary player. This module provides:
//! - [`TreeOpponentPolicy`]: Core abstraction for selecting opponent actions and initializing priors in the search tree.
//! - [`AdversarialOpponent`]: Adversarial MCTS opponent that uses a tree selection policy (e.g. PUCT) to search within the same tree.
//! - [`RandomOpponent`]: Stochastic opponent that uniformly samples a random legal child edge.
//! - [`HeuristicOpponent`]: Adapts any state-based [`OpponentPolicy`] to tree-based search.

use mcts_traits::OpponentPolicy;
use rand::Rng;

use crate::selection::SelectionPolicy;
use crate::tree_store::{EdgeId, EdgeStatsStore, NodeId, PriorStore, TreeStore};

/// Trait governing opponent behavior at Afterstate Nodes within the MCTS tree.
///
/// In coordinated round-based MCTS, afterstate nodes represent intermediate positions
/// where an opponent acts. Implementing this trait allows opponents to:
/// 1. Initialize prior distributions on outgoing edges when an afterstate node is first expanded.
/// 2. Select outgoing child edges to traverse during the simulation descent.
pub trait TreeOpponentPolicy<Action, Reward, Stats: EdgeStatsStore, State, StepDelta = ()> {
    /// Populates priors on outgoing edges when an afterstate node is first expanded.
    ///
    /// The default implementation is a no-op.
    ///
    /// # Parameters
    /// - `tree`: Mutable reference to the search tree.
    /// - `node`: The [`NodeId`] of the newly expanded afterstate node.
    /// - `state`: Impartial world state at `node`.
    fn init_afterstate_priors(
        &self,
        _tree: &mut TreeStore<Action, Reward, Stats, StepDelta>,
        _node: NodeId,
        _state: &State,
    ) {
    }

    /// Selects an edge out of an afterstate node during simulation traversal.
    ///
    /// # Parameters
    /// - `tree`: Search tree containing the afterstate node and its child edges.
    /// - `node`: The [`NodeId`] of the afterstate node.
    /// - `state`: Impartial world state at `node`.
    ///
    /// # Returns
    /// `Some(EdgeId)` if a child edge is selected, or `None` if no edges are available.
    fn select_afterstate_child(
        &self,
        tree: &TreeStore<Action, Reward, Stats, StepDelta>,
        node: NodeId,
        state: &State,
    ) -> Option<EdgeId>;
}

/// Adversarial opponent policy that searches within the same MCTS tree.
///
/// Delegates afterstate child selection to an underlying [`SelectionPolicy`].
/// Since the afterstate node's active agent is set to the opponent's ID, multi-agent
/// selection policies (e.g. [`crate::selection::MultiAgentPuctSelection`]) automatically
/// select moves maximizing the opponent's return $Q_{\text{opp}}$.
///
/// When an afterstate node is expanded, uniform priors $P(s, o) = 1 / K$ are assigned
/// across all $K$ legal opponent moves.
#[derive(Debug, Clone)]
pub struct AdversarialOpponent<S> {
    /// Underlying selection policy used to descend from afterstate nodes.
    pub selection: S,
}

impl<S> AdversarialOpponent<S> {
    /// Creates a new `AdversarialOpponent` wrapping the specified `selection` policy.
    pub const fn new(selection: S) -> Self {
        Self { selection }
    }
}

impl<Action, Reward, Stats, State, StepDelta, S>
    TreeOpponentPolicy<Action, Reward, Stats, State, StepDelta> for AdversarialOpponent<S>
where
    Stats: EdgeStatsStore + PriorStore,
    S: SelectionPolicy<Action, Reward, Stats, StepDelta>,
{
    fn init_afterstate_priors(
        &self,
        tree: &mut TreeStore<Action, Reward, Stats, StepDelta>,
        node: NodeId,
        _state: &State,
    ) {
        let first = tree.first_child_edge(node);
        let count = tree.num_children(node);
        if count > 0 {
            let prior = 1.0 / count as f32;
            for i in 0..count {
                tree.stats.set_prior(EdgeId(first.0 + i), prior);
            }
        }
    }

    fn select_afterstate_child(
        &self,
        tree: &TreeStore<Action, Reward, Stats, StepDelta>,
        node: NodeId,
        _state: &State,
    ) -> Option<EdgeId> {
        self.selection.select_child(tree, node)
    }
}

/// Stochastic opponent policy that selects uniformly random legal actions at afterstate nodes.
#[derive(Debug, Clone, Copy, Default)]
pub struct RandomOpponent;

impl RandomOpponent {
    /// Creates a new `RandomOpponent`.
    pub const fn new() -> Self {
        Self
    }
}

impl<Action, Reward, Stats: EdgeStatsStore, State, StepDelta>
    TreeOpponentPolicy<Action, Reward, Stats, State, StepDelta> for RandomOpponent
{
    fn select_afterstate_child(
        &self,
        tree: &TreeStore<Action, Reward, Stats, StepDelta>,
        node: NodeId,
        _state: &State,
    ) -> Option<EdgeId> {
        let count = tree.num_children(node);
        if count == 0 {
            return None;
        }
        let first = tree.first_child_edge(node);
        let idx = rand::thread_rng().gen_range(0..count);
        Some(EdgeId(first.0 + idx))
    }
}

/// Opponent policy that delegates afterstate action selection to an [`OpponentPolicy`].
///
/// Adapts a state-based policy into a tree-based opponent policy by matching the chosen action
/// against outgoing child edges of the afterstate node.
#[derive(Debug, Clone)]
pub struct HeuristicOpponent<P> {
    /// The wrapped state-based opponent policy.
    pub policy: P,
}

impl<P> HeuristicOpponent<P> {
    /// Wraps an [`OpponentPolicy`] into a tree-based opponent.
    pub const fn new(policy: P) -> Self {
        Self { policy }
    }
}

impl<Action, Reward, Stats, State, StepDelta, P>
    TreeOpponentPolicy<Action, Reward, Stats, State, StepDelta> for HeuristicOpponent<P>
where
    Action: PartialEq,
    Stats: EdgeStatsStore,
    P: OpponentPolicy<State, Action>,
{
    fn select_afterstate_child(
        &self,
        tree: &TreeStore<Action, Reward, Stats, StepDelta>,
        node: NodeId,
        state: &State,
    ) -> Option<EdgeId> {
        let action = self.policy.select_action(state);
        tree.child_edges(node)
            .find(|&edge| *tree.edge_action(edge) == action)
    }
}
