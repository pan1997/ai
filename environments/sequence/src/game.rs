//! Core game state, rules, actions, and configuration for Sequence.

use crate::board::{
    BOARD_CELLS, Card, SequenceRecord, card_positions, coord_to_index, create_double_deck,
    detect_sequences, is_corner, is_corner_index, is_one_eyed_jack, is_two_eyed_jack,
};
use rand::Rng;
use rand::seq::SliceRandom;

/// Configuration parameters for a Sequence match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceConfig {
    /// Number of individual players seated around the table (2..=6).
    pub num_players: usize,
    /// Number of competing teams (2 or 3).
    pub num_teams: usize,
    /// Number of cards dealt to each player at the start of the game.
    pub cards_per_player: usize,
    /// Target number of completed 5-chip sequences needed for a team to win (2 for 2 teams; 1 for 3 teams).
    pub target_sequences: u8,
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self::new_2p()
    }
}

impl SequenceConfig {
    /// Creates a standard 2-player configuration (2 players, 2 teams, 7 cards each, 2 sequences to win).
    #[must_use]
    pub const fn new_2p() -> Self {
        Self {
            num_players: 2,
            num_teams: 2,
            cards_per_player: 7,
            target_sequences: 2,
        }
    }

    /// Creates a 3-player configuration (3 players, 3 teams, 6 cards each, 1 sequence to win).
    #[must_use]
    pub const fn new_3p() -> Self {
        Self {
            num_players: 3,
            num_teams: 3,
            cards_per_player: 6,
            target_sequences: 1,
        }
    }

    /// Creates a 4-player configuration (4 players, 2 teams of 2, 6 cards each, 2 sequences to win).
    #[must_use]
    pub const fn new_4p() -> Self {
        Self {
            num_players: 4,
            num_teams: 2,
            cards_per_player: 6,
            target_sequences: 2,
        }
    }

    /// Creates a 6-player configuration with 2 or 3 teams.
    ///
    /// * `num_teams`: 2 (two teams of 3, 2 sequences to win) or 3 (three teams of 2, 1 sequence to win).
    #[must_use]
    pub const fn new_6p(num_teams: usize) -> Self {
        let target_sequences = if num_teams == 3 { 1 } else { 2 };
        Self {
            num_players: 6,
            num_teams,
            cards_per_player: 5,
            target_sequences,
        }
    }

    /// Returns the team index $(0, 1, \dots)$ for player $p$.
    #[inline]
    #[must_use]
    pub const fn player_team(&self, player: usize) -> u8 {
        (player % self.num_teams) as u8
    }
}

/// Action available to a player on their turn in Sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SequenceAction {
    /// Plays a card from hand and places a team token on the board.
    ///
    /// Used for regular cards (playing on one of its 2 board spaces) or Two-Eyed Jacks
    /// (playing on any unoccupied non-corner board space).
    PlayCard {
        /// The card played from hand.
        card: Card,
        /// The target board coordinate $(r, c)$ where the token is placed.
        pos: (u8, u8),
    },
    /// Plays a One-Eyed Jack from hand and removes an opponent's token from the board.
    ///
    /// The target token must belong to an opponent and must not be part of an already
    /// completed (locked) sequence.
    RemoveToken {
        /// The One-Eyed Jack played from hand.
        card: Card,
        /// The coordinate $(r, c)$ of the opponent's token to remove.
        pos: (u8, u8),
    },
    /// Discards a Dead Card (a card whose two board positions are both occupied) and draws a replacement.
    DiscardDeadCard {
        /// The dead card discarded from hand.
        card: Card,
    },
}

impl SequenceAction {
    /// Returns the card played or discarded in this action.
    #[inline]
    #[must_use]
    pub const fn card(&self) -> Card {
        match *self {
            Self::PlayCard { card, .. }
            | Self::RemoveToken { card, .. }
            | Self::DiscardDeadCard { card } => card,
        }
    }

    /// Returns the target coordinate on the board, if applicable.
    #[inline]
    #[must_use]
    pub const fn target_pos(&self) -> Option<(u8, u8)> {
        match *self {
            Self::PlayCard { pos, .. } | Self::RemoveToken { pos, .. } => Some(pos),
            Self::DiscardDeadCard { .. } => None,
        }
    }
}

/// Complete game state for Sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceState {
    /// Configuration of players and teams.
    pub config: SequenceConfig,
    /// The $10 \times 10$ board token grid (`None` = empty, `Some(team)` = occupied by team).
    pub board: [Option<u8>; BOARD_CELLS],
    /// Mask of tokens that are part of already completed sequences (immune to removal).
    pub locked_chips: [bool; BOARD_CELLS],
    /// Completed sequences recorded so far.
    pub sequences: Vec<SequenceRecord>,
    /// Number of completed sequences per team ($0..3$).
    pub team_sequence_counts: [u8; 3],
    /// Index of the player whose turn it is to move ($0..\text{num\_players}$).
    pub current_player: usize,
    /// Private cards currently held by each seated player.
    pub hands: Vec<Vec<Card>>,
    /// Remaining cards in the face-down draw deck.
    pub deck: Vec<Card>,
    /// Discard pile.
    pub discards: Vec<Card>,
    /// Total moves played in this match.
    pub total_moves: usize,
    /// Whether the game has concluded (a team reached the target sequences or draw).
    pub terminated: bool,
    /// The winning team index, if the game concluded with a win.
    pub winner_team: Option<u8>,
}

impl SequenceState {
    /// Creates a new game state with a freshly shuffled double deck.
    #[must_use]
    pub fn new<R: Rng>(config: SequenceConfig, rng: &mut R) -> Self {
        let mut deck = create_double_deck();
        deck.shuffle(rng);

        let mut hands = Vec::with_capacity(config.num_players);
        for _ in 0..config.num_players {
            let mut hand = Vec::with_capacity(config.cards_per_player);
            for _ in 0..config.cards_per_player {
                if let Some(card) = deck.pop() {
                    hand.push(card);
                }
            }
            hands.push(hand);
        }

        Self {
            config,
            board: [None; BOARD_CELLS],
            locked_chips: [false; BOARD_CELLS],
            sequences: Vec::with_capacity(8),
            team_sequence_counts: [0; 3],
            current_player: 0,
            hands,
            deck,
            discards: Vec::with_capacity(104),
            total_moves: 0,
            terminated: false,
            winner_team: None,
        }
    }

    /// Creates a deterministic game state with a predetermined deck order (useful for testing).
    #[must_use]
    pub fn new_with_deck(config: SequenceConfig, mut deck: Vec<Card>) -> Self {
        let mut hands = Vec::with_capacity(config.num_players);
        for _ in 0..config.num_players {
            let mut hand = Vec::with_capacity(config.cards_per_player);
            for _ in 0..config.cards_per_player {
                if let Some(card) = deck.pop() {
                    hand.push(card);
                }
            }
            hands.push(hand);
        }

        Self {
            config,
            board: [None; BOARD_CELLS],
            locked_chips: [false; BOARD_CELLS],
            sequences: Vec::with_capacity(8),
            team_sequence_counts: [0; 3],
            current_player: 0,
            hands,
            deck,
            discards: Vec::with_capacity(104),
            total_moves: 0,
            terminated: false,
            winner_team: None,
        }
    }

    /// Returns the team index for the currently active player.
    #[inline]
    #[must_use]
    pub const fn active_team(&self) -> u8 {
        self.config.player_team(self.current_player)
    }

    /// Returns true if `card` is a Dead Card (both board positions are occupied).
    #[must_use]
    pub fn is_dead_card(&self, card: Card) -> bool {
        if is_two_eyed_jack(card) || is_one_eyed_jack(card) {
            return false;
        }
        let positions = card_positions(card);
        let idx0 = coord_to_index(positions[0].0, positions[0].1);
        let idx1 = coord_to_index(positions[1].0, positions[1].1);
        self.board[idx0].is_some() && self.board[idx1].is_some()
    }

    /// Populates `out` with all legal actions for the active player in this state.
    pub fn legal_actions(&self, out: &mut Vec<SequenceAction>) {
        out.clear();
        if self.terminated {
            return;
        }

        let hand = &self.hands[self.current_player];
        let active_team = self.active_team();

        // Track cards seen in hand to avoid generating duplicate actions for duplicate cards
        let mut seen_cards: Vec<Card> = Vec::with_capacity(hand.len());

        for &card in hand {
            if seen_cards.contains(&card) {
                continue;
            }
            seen_cards.push(card);

            if is_two_eyed_jack(card) {
                // Two-Eyed Jack is wild: can place on any unoccupied non-corner space
                for idx in 0..BOARD_CELLS {
                    if !is_corner_index(idx) && self.board[idx].is_none() {
                        let pos = crate::board::index_to_coord(idx);
                        out.push(SequenceAction::PlayCard { card, pos });
                    }
                }
            } else if is_one_eyed_jack(card) {
                // One-Eyed Jack is removal: can remove any opponent token that is NOT locked
                for idx in 0..BOARD_CELLS {
                    if let Some(t) = self.board[idx]
                        && t != active_team
                        && !self.locked_chips[idx]
                    {
                        let pos = crate::board::index_to_coord(idx);
                        out.push(SequenceAction::RemoveToken { card, pos });
                    }
                }
            } else if self.is_dead_card(card) {
                // Dead card: can be exchanged for a new card
                out.push(SequenceAction::DiscardDeadCard { card });
            } else {
                // Regular card: place token on either matching board space that is unoccupied
                let positions = card_positions(card);
                for pos in positions {
                    let idx = coord_to_index(pos.0, pos.1);
                    if self.board[idx].is_none() {
                        out.push(SequenceAction::PlayCard { card, pos });
                    }
                }
            }
        }
    }

    /// Applies `action` to this game state, updating the board, checking sequences,
    /// drawing a replacement card, and advancing the turn.
    pub fn step<R: Rng>(&mut self, action: &SequenceAction, rng: &mut R) {
        if self.terminated {
            return;
        }

        let active_player = self.current_player;
        let active_team = self.active_team();
        let card = action.card();

        // 1. Remove played card from hand
        if let Some(pos) = self.hands[active_player].iter().position(|&c| c == card) {
            self.hands[active_player].swap_remove(pos);
        }
        self.discards.push(card);

        // 2. Execute board action
        match *action {
            SequenceAction::PlayCard { pos, .. } => {
                let idx = coord_to_index(pos.0, pos.1);
                debug_assert!(
                    self.board[idx].is_none() && !is_corner(pos.0, pos.1),
                    "Cannot place token on occupied space or corner"
                );
                self.board[idx] = Some(active_team);

                // Localized O(1) sequence detection around pos
                let new_seqs = detect_sequences(
                    &self.board,
                    active_team,
                    Some(pos),
                    &mut self.locked_chips,
                    &mut self.sequences,
                );

                self.team_sequence_counts[active_team as usize] += new_seqs as u8;

                // Check victory condition
                if self.team_sequence_counts[active_team as usize] >= self.config.target_sequences {
                    self.terminated = true;
                    self.winner_team = Some(active_team);
                }
            }
            SequenceAction::RemoveToken { pos, .. } => {
                let idx = coord_to_index(pos.0, pos.1);
                debug_assert!(
                    self.board[idx].is_some()
                        && self.board[idx] != Some(active_team)
                        && !self.locked_chips[idx],
                    "Cannot remove own token or locked sequence token"
                );
                self.board[idx] = None;
                // No sequence detection check needed: token removal cannot create a sequence
            }
            SequenceAction::DiscardDeadCard { .. } => {
                // Hand card was discarded; player immediately draws replacement below
            }
        }

        // 3. Draw replacement card from draw deck
        if !self.terminated {
            if self.deck.is_empty() && !self.discards.is_empty() {
                // Reshuffle discards into draw pile
                self.deck.append(&mut self.discards);
                self.deck.shuffle(rng);
            }

            if let Some(new_card) = self.deck.pop() {
                self.hands[active_player].push(new_card);
            }
        }

        // 4. Advance player turn
        self.current_player = (self.current_player + 1) % self.config.num_players;
        self.total_moves += 1;

        // 5. Board full draw detection (if 96 playable cells are full and no one reached target)
        if !self.terminated {
            let occupied_non_corners = self
                .board
                .iter()
                .enumerate()
                .filter(|&(idx, cell)| !is_corner_index(idx) && cell.is_some())
                .count();

            if occupied_non_corners == 96 {
                self.terminated = true;
                self.winner_team = None; // Draw
            }
        }
    }
}
