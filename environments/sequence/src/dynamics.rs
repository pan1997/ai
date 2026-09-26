//! Planning dynamics, root determinization, and opponent policies for Sequence.

use crate::board::{
    BOARD_CELLS, Card, FrenchBasicCard, Standard52, card_positions, coord_to_index,
    create_double_deck, index_to_coord, is_corner_index, is_jack,
};
use crate::game::{SequenceAction, SequenceState};
use crate::world::{SequenceObservation, SequenceWorld};
use mcts_traits::dynamics::{OpponentPolicy, RoundBasedDynamics, TurnBasedDynamics};
use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Determinizes a hidden-information [`SequenceObservation`] into a hypothetical ground-truth [`SequenceState`].
///
/// Pools all unseen cards ($104 - |\text{my\_hand}| - |\text{discards}|$), randomly shuffles them,
/// and deals hypothetical hands to other players and the draw deck, preserving exact card counts.
#[must_use]
pub fn determinize_state<R: Rng>(obs: &SequenceObservation, rng: &mut R) -> SequenceState {
    let mut unseen = create_double_deck();

    // Remove known observer cards from unseen pool
    for &card in &obs.my_hand {
        if let Some(pos) = unseen.iter().position(|&c| c == card) {
            unseen.swap_remove(pos);
        }
    }

    // Remove known discarded cards from unseen pool
    for &card in &obs.discards {
        if let Some(pos) = unseen.iter().position(|&c| c == card) {
            unseen.swap_remove(pos);
        }
    }

    // Shuffle remaining unseen cards
    unseen.shuffle(rng);

    // Deal hypothetical hands to all seated players
    let mut hands = Vec::with_capacity(obs.config.num_players);
    for p in 0..obs.config.num_players {
        if p == obs.player {
            hands.push(obs.my_hand.clone());
        } else {
            let count = obs.other_hand_counts.get(p).copied().unwrap_or(0);
            let mut hand = Vec::with_capacity(count);
            for _ in 0..count {
                if let Some(c) = unseen.pop() {
                    hand.push(c);
                }
            }
            hands.push(hand);
        }
    }

    // Remaining unseen cards form the hypothetical draw deck
    let deck = unseen;

    SequenceState {
        config: obs.config,
        board: obs.board,
        locked_chips: obs.locked_chips,
        sequences: obs.sequences.clone(),
        team_sequence_counts: obs.team_sequence_counts,
        current_player: obs.current_player,
        hands,
        deck,
        discards: obs.discards.clone(),
        total_moves: obs.total_moves,
        terminated: obs.terminated,
        winner_team: obs.winner_team,
    }
}

/// Standard alternating turn planning dynamics for a $P$-player Sequence match.
pub type SequenceTurnDynamics<const P: usize> = TurnBasedDynamics<SequenceWorld<P>>;

/// Uniform random opponent policy for Sequence simulations.
#[derive(Debug, Clone, Copy, Default)]
pub struct RandomOpponentPolicy;

impl OpponentPolicy<SequenceState, SequenceAction> for RandomOpponentPolicy {
    fn select_action(&self, state: &SequenceState) -> SequenceAction {
        let mut actions = Vec::with_capacity(32);
        state.legal_actions(&mut actions);
        if actions.is_empty() {
            panic!("RandomOpponentPolicy: no legal actions available in state");
        }
        let mut rng = rand::thread_rng();
        *actions.choose(&mut rng).expect("non-empty legal actions")
    }
}

/// Macro round-based planning dynamics for Sequence, stepping through opponent turns using an opponent policy.
pub type SequenceRoundDynamics<const P: usize, Pol> = RoundBasedDynamics<SequenceWorld<P>, Pol>;

/// Belief state determinization sampler for Sequence matches.
///
/// Samples ground-truth [`SequenceState`] determinizations conditioned on a [`SequenceObservation`].
#[derive(Debug, Clone, Default)]
pub struct SequenceBeliefSampler<R = rand::rngs::ThreadRng> {
    rng: R,
}

impl SequenceBeliefSampler<rand::rngs::ThreadRng> {
    /// Creates a new `SequenceBeliefSampler` with thread-local RNG.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }
}

impl<R: rand::Rng> SequenceBeliefSampler<R> {
    /// Creates a new `SequenceBeliefSampler` backed by a custom RNG `rng`.
    pub fn with_rng(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: rand::Rng> mcts_traits::belief::BeliefSampler for SequenceBeliefSampler<R> {
    type State = SequenceState;
    type Context = SequenceObservation;

    #[inline]
    fn sample(&mut self, context: &Self::Context) -> Self::State {
        determinize_state(context, &mut self.rng)
    }
}

/// Single-Tree ISMCTS planning dynamics for Sequence.
///
/// Overrides `expand_actions` to expand all plausible actions across unseen cards
/// at opponent/interior nodes, ensuring child compatibility across diverse state determinizations.
#[derive(Debug, Clone, Copy)]
pub struct SequenceIsmctsDynamics<const P: usize = 2> {
    /// Impartial world referee.
    pub world: SequenceWorld<P>,
    /// Seat index of the planning observer agent ($0..P$).
    pub observer: usize,
    /// Total moves played at the search root.
    pub root_moves: usize,
}

impl<const P: usize> SequenceIsmctsDynamics<P> {
    /// Creates a new `SequenceIsmctsDynamics` planning instance.
    #[must_use]
    pub const fn new(world: SequenceWorld<P>, observer: usize, root_moves: usize) -> Self {
        Self {
            world,
            observer,
            root_moves,
        }
    }
}

impl<const P: usize> mcts_traits::AgentDynamics for SequenceIsmctsDynamics<P> {
    type State = SequenceState;
    type Action = SequenceAction;
    type Reward = [f32; P];
    type StepDelta = ();

    #[inline]
    fn initial(&self) -> Self::State {
        mcts_traits::World::initial(&self.world)
    }

    #[inline]
    fn actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        let player = mcts_traits::TurnBasedWorld::current_player(&self.world, s);
        mcts_traits::World::actions(&self.world, s, player, out);
    }

    fn expand_actions(&self, s: &Self::State, out: &mut Vec<Self::Action>) {
        out.clear();
        if s.terminated {
            return;
        }

        // At the root search node, the observer's hand is known and deterministic.
        if s.current_player == self.observer && s.total_moves == self.root_moves {
            self.actions(s, out);
            return;
        }

        // At interior or opponent nodes, expand plausible candidate actions spanning unseen cards.
        let mut seen_counts: HashMap<Card, u8> = HashMap::with_capacity(32);
        if let Some(my_hand) = s.hands.get(self.observer) {
            for &c in my_hand {
                *seen_counts.entry(c).or_insert(0) += 1;
            }
        }
        for &c in &s.discards {
            *seen_counts.entry(c).or_insert(0) += 1;
        }

        let is_unseen = |c: Card| seen_counts.get(&c).copied().unwrap_or(0) < 2;

        // 1. Regular non-Jack cards
        for &c in &Standard52::DECK {
            if is_jack(c) || !is_unseen(c) {
                continue;
            }
            if s.is_dead_card(c) {
                out.push(SequenceAction::DiscardDeadCard { card: c });
            } else {
                for pos in card_positions(c) {
                    let idx = coord_to_index(pos.0, pos.1);
                    if s.board[idx].is_none() {
                        out.push(SequenceAction::PlayCard { card: c, pos });
                    }
                }
            }
        }

        // 2. One-Eyed Jacks (Anti-wild removal)
        let active_team = s.active_team();
        let one_eyed = [FrenchBasicCard::JACK_SPADES, FrenchBasicCard::JACK_HEARTS];
        for &jack in &one_eyed {
            if !is_unseen(jack) {
                continue;
            }
            for idx in 0..BOARD_CELLS {
                if let Some(t) = s.board[idx]
                    && t != active_team
                    && !s.locked_chips[idx]
                {
                    let pos = index_to_coord(idx);
                    out.push(SequenceAction::RemoveToken { card: jack, pos });
                }
            }
        }

        // 3. Two-Eyed Jacks (Wild placement on all unoccupied non-corner spaces)
        let two_eyed = [FrenchBasicCard::JACK_CLUBS, FrenchBasicCard::JACK_DIAMONDS];
        for &jack in &two_eyed {
            if !is_unseen(jack) {
                continue;
            }
            for idx in 0..BOARD_CELLS {
                if !is_corner_index(idx) && s.board[idx].is_none() {
                    let pos = index_to_coord(idx);
                    out.push(SequenceAction::PlayCard { card: jack, pos });
                }
            }
        }

        // Fallback safety if candidate set is empty
        if out.is_empty() {
            self.actions(s, out);
        }
    }

    #[inline]
    fn step(
        &self,
        s: &mut Self::State,
        action: &Self::Action,
    ) -> mcts_traits::StepOutcome<Self::Reward, ()> {
        let outcome = self.world.step_action(s, action);
        mcts_traits::StepOutcome::new(outcome.reward, outcome.terminated)
    }

    #[inline]
    fn current_agent(&self, s: &Self::State) -> mcts_traits::AgentId {
        mcts_traits::AgentId(mcts_traits::TurnBasedWorld::current_player(&self.world, s) as u32)
    }
}
