//! Board layout, coordinate mapping, and sequence detection for the game of Sequence.

use cardpack::prelude::{FrenchBasicCard, Standard52};

pub use cardpack::prelude::BasicCard as Card;

/// The width and height of the Sequence board ($10 \times 10$).
pub const BOARD_DIM: usize = 10;
/// Total number of cells on the Sequence board (100).
pub const BOARD_CELLS: usize = 100;

/// Returns true if `card` is a Two-Eyed Jack (Wild: Jack of Clubs or Jack of Diamonds).
#[inline]
pub fn is_two_eyed_jack(card: Card) -> bool {
    card == FrenchBasicCard::JACK_CLUBS || card == FrenchBasicCard::JACK_DIAMONDS
}

/// Returns true if `card` is a One-Eyed Jack (Anti-Wild / Removal: Jack of Spades or Jack of Hearts).
#[inline]
pub fn is_one_eyed_jack(card: Card) -> bool {
    card == FrenchBasicCard::JACK_SPADES || card == FrenchBasicCard::JACK_HEARTS
}

/// Returns true if `card` is any Jack.
#[inline]
pub fn is_jack(card: Card) -> bool {
    is_two_eyed_jack(card) || is_one_eyed_jack(card)
}

/// Generates the standard 104-card double deck used in Sequence (two standard 52-card decks).
#[must_use]
pub fn create_double_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(104);
    deck.extend_from_slice(&Standard52::DECK);
    deck.extend_from_slice(&Standard52::DECK);
    deck
}

/// Cell content printed on the physical Sequence game board.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoardCell {
    /// Corner space (Wild/Free for all players and teams).
    Corner,
    /// Regular card space corresponding to one of the 48 non-Jack playing cards.
    Card(Card),
}

/// Converts a 2D grid coordinate $(r, c)$ to a flat 1D cell index in $0..100$.
#[inline]
pub const fn coord_to_index(r: u8, c: u8) -> usize {
    r as usize * BOARD_DIM + c as usize
}

/// Converts a flat 1D cell index in $0..100$ to a 2D grid coordinate $(r, c)$.
#[inline]
pub const fn index_to_coord(idx: usize) -> (u8, u8) {
    ((idx / BOARD_DIM) as u8, (idx % BOARD_DIM) as u8)
}

/// Returns true if the flat index `idx` is one of the four corner wild spaces.
#[inline]
pub const fn is_corner_index(idx: usize) -> bool {
    idx == 0 || idx == 9 || idx == 90 || idx == 99
}

/// Returns true if coordinate $(r, c)$ is one of the four corner wild spaces.
#[inline]
pub const fn is_corner(r: u8, c: u8) -> bool {
    (r == 0 || r == 9) && (c == 0 || c == 9)
}

// Aliases for board card matrix definition readability
const C_2S: BoardCell = BoardCell::Card(FrenchBasicCard::DEUCE_SPADES);
const C_3S: BoardCell = BoardCell::Card(FrenchBasicCard::TREY_SPADES);
const C_4S: BoardCell = BoardCell::Card(FrenchBasicCard::FOUR_SPADES);
const C_5S: BoardCell = BoardCell::Card(FrenchBasicCard::FIVE_SPADES);
const C_6S: BoardCell = BoardCell::Card(FrenchBasicCard::SIX_SPADES);
const C_7S: BoardCell = BoardCell::Card(FrenchBasicCard::SEVEN_SPADES);
const C_8S: BoardCell = BoardCell::Card(FrenchBasicCard::EIGHT_SPADES);
const C_9S: BoardCell = BoardCell::Card(FrenchBasicCard::NINE_SPADES);
const C_10S: BoardCell = BoardCell::Card(FrenchBasicCard::TEN_SPADES);
const C_QS: BoardCell = BoardCell::Card(FrenchBasicCard::QUEEN_SPADES);
const C_KS: BoardCell = BoardCell::Card(FrenchBasicCard::KING_SPADES);
const C_AS: BoardCell = BoardCell::Card(FrenchBasicCard::ACE_SPADES);

const C_2H: BoardCell = BoardCell::Card(FrenchBasicCard::DEUCE_HEARTS);
const C_3H: BoardCell = BoardCell::Card(FrenchBasicCard::TREY_HEARTS);
const C_4H: BoardCell = BoardCell::Card(FrenchBasicCard::FOUR_HEARTS);
const C_5H: BoardCell = BoardCell::Card(FrenchBasicCard::FIVE_HEARTS);
const C_6H: BoardCell = BoardCell::Card(FrenchBasicCard::SIX_HEARTS);
const C_7H: BoardCell = BoardCell::Card(FrenchBasicCard::SEVEN_HEARTS);
const C_8H: BoardCell = BoardCell::Card(FrenchBasicCard::EIGHT_HEARTS);
const C_9H: BoardCell = BoardCell::Card(FrenchBasicCard::NINE_HEARTS);
const C_10H: BoardCell = BoardCell::Card(FrenchBasicCard::TEN_HEARTS);
const C_QH: BoardCell = BoardCell::Card(FrenchBasicCard::QUEEN_HEARTS);
const C_KH: BoardCell = BoardCell::Card(FrenchBasicCard::KING_HEARTS);
const C_AH: BoardCell = BoardCell::Card(FrenchBasicCard::ACE_HEARTS);

const C_2D: BoardCell = BoardCell::Card(FrenchBasicCard::DEUCE_DIAMONDS);
const C_3D: BoardCell = BoardCell::Card(FrenchBasicCard::TREY_DIAMONDS);
const C_4D: BoardCell = BoardCell::Card(FrenchBasicCard::FOUR_DIAMONDS);
const C_5D: BoardCell = BoardCell::Card(FrenchBasicCard::FIVE_DIAMONDS);
const C_6D: BoardCell = BoardCell::Card(FrenchBasicCard::SIX_DIAMONDS);
const C_7D: BoardCell = BoardCell::Card(FrenchBasicCard::SEVEN_DIAMONDS);
const C_8D: BoardCell = BoardCell::Card(FrenchBasicCard::EIGHT_DIAMONDS);
const C_9D: BoardCell = BoardCell::Card(FrenchBasicCard::NINE_DIAMONDS);
const C_10D: BoardCell = BoardCell::Card(FrenchBasicCard::TEN_DIAMONDS);
const C_QD: BoardCell = BoardCell::Card(FrenchBasicCard::QUEEN_DIAMONDS);
const C_KD: BoardCell = BoardCell::Card(FrenchBasicCard::KING_DIAMONDS);
const C_AD: BoardCell = BoardCell::Card(FrenchBasicCard::ACE_DIAMONDS);

const C_2C: BoardCell = BoardCell::Card(FrenchBasicCard::DEUCE_CLUBS);
const C_3C: BoardCell = BoardCell::Card(FrenchBasicCard::TREY_CLUBS);
const C_4C: BoardCell = BoardCell::Card(FrenchBasicCard::FOUR_CLUBS);
const C_5C: BoardCell = BoardCell::Card(FrenchBasicCard::FIVE_CLUBS);
const C_6C: BoardCell = BoardCell::Card(FrenchBasicCard::SIX_CLUBS);
const C_7C: BoardCell = BoardCell::Card(FrenchBasicCard::SEVEN_CLUBS);
const C_8C: BoardCell = BoardCell::Card(FrenchBasicCard::EIGHT_CLUBS);
const C_9C: BoardCell = BoardCell::Card(FrenchBasicCard::NINE_CLUBS);
const C_10C: BoardCell = BoardCell::Card(FrenchBasicCard::TEN_CLUBS);
const C_QC: BoardCell = BoardCell::Card(FrenchBasicCard::QUEEN_CLUBS);
const C_KC: BoardCell = BoardCell::Card(FrenchBasicCard::KING_CLUBS);
const C_AC: BoardCell = BoardCell::Card(FrenchBasicCard::ACE_CLUBS);

const CORNER: BoardCell = BoardCell::Corner;

/// The official $10 \times 10$ Sequence game board layout.
///
/// Contains 4 free corner spaces and 96 regular card positions (each of the 48 non-Jack cards
/// appears exactly twice on the board).
pub const BOARD_LAYOUT: [BoardCell; BOARD_CELLS] = [
    // Row 0
    CORNER, C_2S, C_3S, C_4S, C_5S, C_6S, C_7S, C_8S, C_9S, CORNER, // Row 1
    C_6C, C_5C, C_4C, C_3C, C_2C, C_AH, C_KH, C_QH, C_10H, C_10S, // Row 2
    C_7C, C_AS, C_2D, C_3D, C_4D, C_5D, C_6D, C_7D, C_9H, C_QS, // Row 3
    C_8C, C_KS, C_6C, C_5C, C_4C, C_3C, C_2C, C_8D, C_8H, C_KS, // Row 4
    C_9C, C_QS, C_7C, C_6H, C_5H, C_4H, C_AH, C_9D, C_7H, C_AS, // Row 5
    C_10C, C_10S, C_8C, C_7H, C_2H, C_3H, C_KH, C_10D, C_6H, C_2D, // Row 6
    C_QC, C_9S, C_9C, C_8H, C_9H, C_10H, C_QH, C_QD, C_5H, C_3D, // Row 7
    C_KC, C_8S, C_10C, C_QC, C_KC, C_AC, C_AD, C_KD, C_4H, C_4D, // Row 8
    C_AC, C_7S, C_6S, C_5S, C_4S, C_3S, C_2S, C_2H, C_3H, C_5D, // Row 9
    CORNER, C_AD, C_KD, C_QD, C_10D, C_9D, C_8D, C_7D, C_6D, CORNER,
];

/// Returns the two grid positions on the board where `card` appears.
///
/// Panics if `card` is a Jack (Jacks do not appear on the board).
#[must_use]
pub fn card_positions(card: Card) -> [(u8, u8); 2] {
    let mut found = [(0, 0); 2];
    let mut count = 0;
    for (idx, cell) in BOARD_LAYOUT.iter().enumerate() {
        if let BoardCell::Card(c) = cell
            && *c == card
        {
            found[count] = index_to_coord(idx);
            count += 1;
            if count == 2 {
                return found;
            }
        }
    }
    panic!("Card {:?} not found twice on board", card);
}

/// A completed sequence of 5 connected positions for a team.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceRecord {
    /// The team that completed this sequence.
    pub team: u8,
    /// The 5 flat cell indices forming this sequence, sorted ascending.
    pub cells: [u8; 5],
}

impl SequenceRecord {
    /// Creates a new `SequenceRecord` with cells sorted for canonical comparison.
    #[must_use]
    pub fn new(team: u8, mut cells: [u8; 5]) -> Self {
        cells.sort_unstable();
        Self { team, cells }
    }

    /// Returns the number of common non-corner cells shared with another sequence.
    #[must_use]
    pub fn shared_non_corner_count(&self, other: &Self) -> usize {
        let mut count = 0;
        for &c in &self.cells {
            if !is_corner_index(c as usize) && other.cells.contains(&c) {
                count += 1;
            }
        }
        count
    }
}

/// Detects new sequences formed by `team` on the board.
///
/// # Arguments
/// * `board`: Flat array of 100 cells with tokens placed (`None` or `Some(team)`).
/// * `team`: The active team index (0, 1, or 2) to check.
/// * `last_placed`: Optional coordinate `(r, c)` of the chip placed on this turn.
///   - When `Some((r, c))`, performs a localized $O(1)$ scan inspecting only the 4 lines
///     intersecting `(r, c)`.
///   - When `None`, performs a comprehensive full-board scan across all 100 cells.
/// * `locked_chips`: In-out array marking which cells belong to completed sequences.
/// * `existing_sequences`: In-out list of sequences already completed by all teams.
///
/// # Returns
/// The number of new sequences completed by `team` on this move.
pub fn detect_sequences(
    board: &[Option<u8>; BOARD_CELLS],
    team: u8,
    last_placed: Option<(u8, u8)>,
    locked_chips: &mut [bool; BOARD_CELLS],
    existing_sequences: &mut Vec<SequenceRecord>,
) -> usize {
    let mut new_sequences_count = 0;

    // Helper closure to evaluate a candidate 5-cell line
    let check_line = |line: [(i8, i8); 5],
                      locked_chips: &mut [bool; BOARD_CELLS],
                      existing_sequences: &mut Vec<SequenceRecord>|
     -> bool {
        let mut cells = [0u8; 5];
        for (i, &(r, c)) in line.iter().enumerate() {
            if !(0..10).contains(&r) || !(0..10).contains(&c) {
                return false;
            }
            let idx = coord_to_index(r as u8, c as u8);
            cells[i] = idx as u8;

            // Must be either a corner (wild for everyone) or have the team's token
            if !is_corner_index(idx) && board[idx] != Some(team) {
                return false;
            }
        }

        let cand = SequenceRecord::new(team, cells);

        // Sequence duplicate check
        if existing_sequences.iter().any(|s| s == &cand) {
            return false;
        }

        // 1-chip intersection rule: Can share AT MOST 1 non-corner chip with any existing sequence of the same team
        for s in existing_sequences.iter().filter(|s| s.team == team) {
            if cand.shared_non_corner_count(s) > 1 {
                return false;
            }
        }

        // Valid new sequence! Lock its non-corner chips and record
        for &idx in &cand.cells {
            if !is_corner_index(idx as usize) {
                locked_chips[idx as usize] = true;
            }
        }
        existing_sequences.push(cand);
        true
    };

    if let Some((pr, pc)) = last_placed {
        let pr = pr as i8;
        let pc = pc as i8;

        // 4 lines intersecting (pr, pc): Horizontal, Vertical, Diagonal (\), Anti-diagonal (/)
        let directions: [(i8, i8); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

        for &(dr, dc) in &directions {
            // A 5-in-a-row containing (pr, pc) can start at offset k in -4..=0
            for k in -4..=0 {
                let start_r = pr + k * dr;
                let start_c = pc + k * dc;

                let line = [
                    (start_r, start_c),
                    (start_r + dr, start_c + dc),
                    (start_r + 2 * dr, start_c + 2 * dc),
                    (start_r + 3 * dr, start_c + 3 * dc),
                    (start_r + 4 * dr, start_c + 4 * dc),
                ];

                if check_line(line, locked_chips, existing_sequences) {
                    new_sequences_count += 1;
                }
            }
        }
    } else {
        // Full board scan across all rows, columns, and diagonals
        // 1. Horizontal
        for r in 0..10i8 {
            for c in 0..=5i8 {
                let line = [(r, c), (r, c + 1), (r, c + 2), (r, c + 3), (r, c + 4)];
                if check_line(line, locked_chips, existing_sequences) {
                    new_sequences_count += 1;
                }
            }
        }
        // 2. Vertical
        for r in 0..=5i8 {
            for c in 0..10i8 {
                let line = [(r, c), (r + 1, c), (r + 2, c), (r + 3, c), (r + 4, c)];
                if check_line(line, locked_chips, existing_sequences) {
                    new_sequences_count += 1;
                }
            }
        }
        // 3. Diagonal (\)
        for r in 0..=5i8 {
            for c in 0..=5i8 {
                let line = [
                    (r, c),
                    (r + 1, c + 1),
                    (r + 2, c + 2),
                    (r + 3, c + 3),
                    (r + 4, c + 4),
                ];
                if check_line(line, locked_chips, existing_sequences) {
                    new_sequences_count += 1;
                }
            }
        }
        // 4. Anti-Diagonal (/)
        for r in 0..=5i8 {
            for c in 4..10i8 {
                let line = [
                    (r, c),
                    (r + 1, c - 1),
                    (r + 2, c - 2),
                    (r + 3, c - 3),
                    (r + 4, c - 4),
                ];
                if check_line(line, locked_chips, existing_sequences) {
                    new_sequences_count += 1;
                }
            }
        }
    }

    new_sequences_count
}
