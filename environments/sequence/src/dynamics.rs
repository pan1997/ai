//! Planning dynamics, root determinization, and opponent policies for Sequence.

use crate::board::create_double_deck;
use crate::game::{SequenceAction, SequenceState};
use crate::world::{SequenceObservation, SequenceWorld};
use mcts_traits::dynamics::{OpponentPolicy, RoundBasedDynamics, TurnBasedDynamics};
use rand::seq::SliceRandom;
use rand::Rng;

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
pub type SequenceRoundDynamics<const P: usize, Pol> =
    RoundBasedDynamics<SequenceWorld<P>, Pol>;
