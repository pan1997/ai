//! Ground-truth World referee implementation for Sequence.

use crate::board::{Card, SequenceRecord, BOARD_CELLS};
use crate::game::{SequenceAction, SequenceConfig, SequenceState};
use mcts_traits::{StepOutcome, TurnBasedWorld, World};

/// Filtered player-specific observation of a Sequence game state.
///
/// Models imperfect information: hides opponents' private cards and the draw deck order,
/// while providing the public board, sequence status, discard pile, and own hand.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceObservation {
    /// Match configuration.
    pub config: SequenceConfig,
    /// The index of the observer player ($0..P$).
    pub player: usize,
    /// Public $10 \times 10$ board token grid (`None` = empty, `Some(team)` = occupied).
    pub board: [Option<u8>; BOARD_CELLS],
    /// Mask of tokens that are part of already completed sequences.
    pub locked_chips: [bool; BOARD_CELLS],
    /// Completed sequences recorded so far.
    pub sequences: Vec<SequenceRecord>,
    /// Number of completed sequences per team.
    pub team_sequence_counts: [u8; 3],
    /// Index of the player whose turn it is to move.
    pub current_player: usize,
    /// Observer's private hand of cards.
    pub my_hand: Vec<Card>,
    /// Number of cards held by each player seated at the table.
    pub other_hand_counts: Vec<usize>,
    /// Total cards remaining face-down in the draw deck.
    pub deck_remaining: usize,
    /// Public discard pile (face-up).
    pub discards: Vec<Card>,
    /// Total moves played in this match.
    pub total_moves: usize,
    /// Whether the game has concluded.
    pub terminated: bool,
    /// The winning team index, if the game concluded with a win.
    pub winner_team: Option<u8>,
}

impl SequenceObservation {
    /// Populates `out` with all legal actions for the observing player if it is their turn.
    pub fn legal_actions(&self, out: &mut Vec<SequenceAction>) {
        out.clear();
        if self.terminated || self.current_player != self.player {
            return;
        }

        let active_team = self.config.player_team(self.player);
        let mut seen_cards: Vec<Card> = Vec::with_capacity(self.my_hand.len());

        for &card in &self.my_hand {
            if seen_cards.contains(&card) {
                continue;
            }
            seen_cards.push(card);

            if crate::board::is_two_eyed_jack(card) {
                for idx in 0..BOARD_CELLS {
                    if !crate::board::is_corner_index(idx) && self.board[idx].is_none() {
                        let pos = crate::board::index_to_coord(idx);
                        out.push(SequenceAction::PlayCard { card, pos });
                    }
                }
            } else if crate::board::is_one_eyed_jack(card) {
                for idx in 0..BOARD_CELLS {
                    if let Some(t) = self.board[idx]
                        && t != active_team
                        && !self.locked_chips[idx]
                    {
                        let pos = crate::board::index_to_coord(idx);
                        out.push(SequenceAction::RemoveToken { card, pos });
                    }
                }
            } else {
                let positions = crate::board::card_positions(card);
                let idx0 = crate::board::coord_to_index(positions[0].0, positions[0].1);
                let idx1 = crate::board::coord_to_index(positions[1].0, positions[1].1);
                let is_dead = self.board[idx0].is_some() && self.board[idx1].is_some();

                if is_dead {
                    out.push(SequenceAction::DiscardDeadCard { card });
                } else {
                    for pos in positions {
                        let idx = crate::board::coord_to_index(pos.0, pos.1);
                        if self.board[idx].is_none() {
                            out.push(SequenceAction::PlayCard { card, pos });
                        }
                    }
                }
            }
        }
    }
}

/// Impartial referee and ground-truth environment for Sequence.
///
/// Models the external arbitration of a $P$-player Sequence match across 2 or 3 teams.
///
/// Standard configurations:
/// - 2 players, 2 teams: [`Sequence2PWorld`] (`SequenceWorld<2>`)
/// - 3 players, 3 teams: [`Sequence3PWorld`] (`SequenceWorld<3>`)
/// - 4 players, 2 teams: [`Sequence4PWorld`] (`SequenceWorld<4>`)
/// - 6 players, 2 or 3 teams: [`Sequence6PWorld`] (`SequenceWorld<6>`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceWorld<const P: usize = 2> {
    /// Match configuration.
    pub config: SequenceConfig,
}

impl<const P: usize> Default for SequenceWorld<P> {
    fn default() -> Self {
        let config = match P {
            2 => SequenceConfig::new_2p(),
            3 => SequenceConfig::new_3p(),
            4 => SequenceConfig::new_4p(),
            6 => SequenceConfig::new_6p(2),
            _ => SequenceConfig {
                num_players: P,
                num_teams: 2,
                cards_per_player: 6,
                target_sequences: 2,
            },
        };
        Self { config }
    }
}

impl<const P: usize> SequenceWorld<P> {
    /// Creates a new `SequenceWorld` referee with a specific configuration.
    ///
    /// # Panics
    /// Panics if `config.num_players != P`.
    pub const fn new(config: SequenceConfig) -> Self {
        assert!(
            config.num_players == P,
            "SequenceWorld: config.num_players does not match generic P"
        );
        Self { config }
    }

    /// Computes zero-sum team rewards for all seated players given final game state.
    #[inline]
    pub fn compute_rewards(&self, ws: &SequenceState) -> [f32; P] {
        let mut rewards = [0.0f32; P];
        if let Some(win_team) = ws.winner_team {
            let win_reward = 1.0f32;
            let loss_reward = if self.config.num_teams == 3 {
                -0.5f32
            } else {
                -1.0f32
            };

            for (p, r) in rewards.iter_mut().enumerate().take(P) {
                let team = self.config.player_team(p);
                *r = if team == win_team {
                    win_reward
                } else {
                    loss_reward
                };
            }
        }
        rewards
    }

    /// Transitions `ws` forward by `action` for the active player without heap allocation.
    ///
    /// Returns the immediate reward vector $\mathbf{r} \in [-1.0, 1.0]^P$ and episode termination flag.
    #[inline]
    pub fn step_action(
        &self,
        ws: &mut SequenceState,
        action: &SequenceAction,
    ) -> StepOutcome<[f32; P]> {
        let mut rng = rand::thread_rng();
        ws.step(action, &mut rng);

        if ws.terminated {
            let rewards = self.compute_rewards(ws);
            StepOutcome::new(rewards, true)
        } else {
            StepOutcome::new([0.0f32; P], false)
        }
    }

    /// Populates `out` with all legal actions for the current player in `ws`.
    #[inline]
    pub fn legal_actions(&self, ws: &SequenceState, out: &mut Vec<SequenceAction>) {
        ws.legal_actions(out);
    }

    /// Returns `true` if the match is in a terminal state.
    #[inline]
    pub fn is_terminal(&self, ws: &SequenceState) -> bool {
        ws.terminated
    }
}

impl<const P: usize> World for SequenceWorld<P> {
    type WorldState = SequenceState;
    type Action = SequenceAction;
    type Observation = SequenceObservation;

    #[inline]
    fn n_players(&self) -> usize {
        P
    }

    #[inline]
    fn initial(&self) -> Self::WorldState {
        let mut rng = rand::thread_rng();
        SequenceState::new(self.config, &mut rng)
    }

    #[inline]
    fn observe(&self, ws: &Self::WorldState, player: usize) -> Self::Observation {
        SequenceObservation {
            config: ws.config,
            player,
            board: ws.board,
            locked_chips: ws.locked_chips,
            sequences: ws.sequences.clone(),
            team_sequence_counts: ws.team_sequence_counts,
            current_player: ws.current_player,
            my_hand: ws.hands.get(player).cloned().unwrap_or_default(),
            other_hand_counts: ws.hands.iter().map(|h| h.len()).collect(),
            deck_remaining: ws.deck.len(),
            discards: ws.discards.clone(),
            total_moves: ws.total_moves,
            terminated: ws.terminated,
            winner_team: ws.winner_team,
        }
    }

    #[inline]
    fn actions(&self, ws: &Self::WorldState, player: usize, out: &mut Vec<Self::Action>) {
        out.clear();
        if player == ws.current_player && !ws.terminated {
            ws.legal_actions(out);
        }
    }

    #[inline]
    fn step(&self, ws: &mut Self::WorldState, joint: &[Self::Action]) -> (Vec<f32>, bool) {
        let active_player = ws.current_player;
        let action = &joint[active_player];
        let outcome = self.step_action(ws, action);
        (outcome.reward.to_vec(), outcome.terminated)
    }

    #[inline]
    fn terminal(&self, ws: &Self::WorldState) -> bool {
        self.is_terminal(ws)
    }
}

impl<const P: usize> TurnBasedWorld for SequenceWorld<P> {
    type StepReward = [f32; P];

    #[inline]
    fn current_player(&self, ws: &Self::WorldState) -> usize {
        ws.current_player
    }

    #[inline]
    fn step_action(
        &self,
        ws: &mut Self::WorldState,
        action: &Self::Action,
    ) -> StepOutcome<Self::StepReward> {
        self.step_action(ws, action)
    }
}

/// Standard 2-player Sequence referee.
pub type Sequence2PWorld = SequenceWorld<2>;

/// Standard 3-player Sequence referee.
pub type Sequence3PWorld = SequenceWorld<3>;

/// Standard 4-player Sequence referee (2 teams of 2).
pub type Sequence4PWorld = SequenceWorld<4>;

/// Standard 6-player Sequence referee (2 teams of 3 or 3 teams of 2).
pub type Sequence6PWorld = SequenceWorld<6>;
