//! Terminal rendering utilities for Sequence board grids, card representations, and scoreboards.

use crate::board::{
    coord_to_index, is_corner, is_one_eyed_jack, is_two_eyed_jack, BoardCell, Card, BOARD_CELLS,
    BOARD_DIM,
};
use crate::game::{SequenceAction, SequenceState};

/// Team color names and symbols.
pub const TEAM_NAMES: [&str; 3] = ["Blue", "Green", "Red"];
pub const TEAM_SYMBOLS: [&str; 3] = ["B", "G", "R"];

/// Formats a card into a compact 3-character representation with suit symbols.
pub fn format_card(card: Card) -> String {
    format!("{card}")
}

/// Formats a board cell at coordinate $(r, c)$ given the token and locked status.
pub fn format_cell(
    r: u8,
    c: u8,
    token: Option<u8>,
    locked: bool,
    use_color: bool,
) -> String {
    if is_corner(r, c) {
        if use_color {
            return "\x1b[1;33m ★  \x1b[0m".to_string();
        } else {
            return " ★  ".to_string();
        }
    }

    let idx = coord_to_index(r, c);
    let cell_card = match crate::board::BOARD_LAYOUT[idx] {
        BoardCell::Card(card) => format!("{card:3}"),
        BoardCell::Corner => " ★ ".to_string(),
    };

    if let Some(team) = token {
        let sym = match team {
            0 => {
                if use_color {
                    if locked {
                        "\x1b[1;37;44m*B*\x1b[0m "
                    } else {
                        "\x1b[1;34m ●B\x1b[0m "
                    }
                } else if locked {
                    "[B] "
                } else {
                    " B  "
                }
            }
            1 => {
                if use_color {
                    if locked {
                        "\x1b[1;37;42m*G*\x1b[0m "
                    } else {
                        "\x1b[1;32m ●G\x1b[0m "
                    }
                } else if locked {
                    "[G] "
                } else {
                    " G  "
                }
            }
            2 => {
                if use_color {
                    if locked {
                        "\x1b[1;37;41m*R*\x1b[0m "
                    } else {
                        "\x1b[1;31m ●R\x1b[0m "
                    }
                } else if locked {
                    "[R] "
                } else {
                    " R  "
                }
            }
            _ => " ?  ",
        };
        sym.to_string()
    } else if use_color {
        format!("\x1b[90m{cell_card}\x1b[0m ")
    } else {
        format!("{cell_card} ")
    }
}

/// Renders the full $10 \times 10$ board to a string.
pub fn render_board(
    board: &[Option<u8>; BOARD_CELLS],
    locked_chips: &[bool; BOARD_CELLS],
    use_color: bool,
) -> String {
    let mut out = String::new();

    // Column indices header
    out.push_str("    ");
    for c in 0..BOARD_DIM {
        out.push_str(&format!("  {c} "));
    }
    out.push('\n');

    // Top border
    out.push_str("   ┌");
    for c in 0..BOARD_DIM {
        out.push_str("────");
        if c + 1 < BOARD_DIM {
            out.push('┬');
        }
    }
    out.push_str("┐\n");

    // Board rows
    for r in 0..BOARD_DIM {
        out.push_str(&format!("{r:2} │", r = r));
        for c in 0..BOARD_DIM {
            let idx = coord_to_index(r as u8, c as u8);
            let cell_str = format_cell(
                r as u8,
                c as u8,
                board[idx],
                locked_chips[idx],
                use_color,
            );
            out.push_str(&cell_str);
            if c + 1 < BOARD_DIM {
                out.push('│');
            }
        }
        out.push_str("│\n");

        if r + 1 < BOARD_DIM {
            out.push_str("   ├");
            for c in 0..BOARD_DIM {
                out.push_str("────");
                if c + 1 < BOARD_DIM {
                    out.push('┼');
                }
            }
            out.push_str("┤\n");
        }
    }

    // Bottom border
    out.push_str("   └");
    for c in 0..BOARD_DIM {
        out.push_str("────");
        if c + 1 < BOARD_DIM {
            out.push('┴');
        }
    }
    out.push_str("┘\n");

    out
}

/// Renders full game status scoreboard and board state.
pub fn render_state(state: &SequenceState, use_color: bool) -> String {
    let mut out = String::new();

    // Title & status banner
    out.push_str("\n╔══════════════════════════════════════════════════════════════════════╗\n");
    out.push_str("║                          SEQUENCE MATCH                              ║\n");
    out.push_str("╠══════════════════════════════════════════════════════════════════════╣\n");

    let team_str = format!(
        "Turn {}: Player {} (Team {} - {})",
        state.total_moves + 1,
        state.current_player,
        state.active_team(),
        TEAM_NAMES[state.active_team() as usize]
    );
    out.push_str(&format!("║ {team_str:<68} ║\n"));

    let mut scores_str = String::from("Sequences: ");
    for t in 0..state.config.num_teams {
        scores_str.push_str(&format!(
            "{} {}: {}/{} | ",
            TEAM_SYMBOLS[t],
            TEAM_NAMES[t],
            state.team_sequence_counts[t],
            state.config.target_sequences
        ));
    }
    scores_str.push_str(&format!("Deck: {} left", state.deck.len()));
    out.push_str(&format!("║ {scores_str:<68} ║\n"));

    if state.terminated {
        let win_msg = match state.winner_team {
            Some(w) => format!("GAME OVER: Team {} ({}) WINS!", w, TEAM_NAMES[w as usize]),
            None => "GAME OVER: Match ended in a DRAW!".to_string(),
        };
        out.push_str(&format!("║ \x1b[1;32m{win_msg:<68}\x1b[0m ║\n"));
    }
    out.push_str("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    out.push_str(&render_board(&state.board, &state.locked_chips, use_color));

    // Active player's hand
    let hand = &state.hands[state.current_player];
    out.push_str(&format!(
        "\nPlayer {}'s Hand ({} cards): ",
        state.current_player,
        hand.len()
    ));
    for (i, &card) in hand.iter().enumerate() {
        let tag = if is_two_eyed_jack(card) {
            " (Wild)"
        } else if is_one_eyed_jack(card) {
            " (Remove)"
        } else if state.is_dead_card(card) {
            " (DEAD)"
        } else {
            ""
        };
        out.push_str(&format!("[{i}]: {card}{tag}  "));
    }
    out.push('\n');

    out
}

/// Formats a single [`SequenceAction`] as a readable string.
pub fn format_action(action: &SequenceAction) -> String {
    match *action {
        SequenceAction::PlayCard { card, pos } => {
            if is_two_eyed_jack(card) {
                format!("Play Wild Jack {card} -> ({}, {})", pos.0, pos.1)
            } else {
                format!("Play {card} -> ({}, {})", pos.0, pos.1)
            }
        }
        SequenceAction::RemoveToken { card, pos } => {
            format!("Remove token at ({}, {}) with {card}", pos.0, pos.1)
        }
        SequenceAction::DiscardDeadCard { card } => {
            format!("Discard Dead Card {card} (draw replacement)")
        }
    }
}
