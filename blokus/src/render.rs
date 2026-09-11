//! Terminal rendering utilities for Blokus board grids, player scoreboards, and MCTS statistics.

use crate::game::{BlokusAction, BlokusState, Player, EMPTY};
use crate::pieces::{piece_size, PIECE_NAMES};

/// Formats a Blokus board into a string with optional ANSI colors.
pub fn render_board<const B: usize, const P: usize>(
    state: &BlokusState<B, P>,
    use_color: bool,
) -> String {
    let mut out = String::new();

    // Column indices header
    out.push_str("    ");
    for c in 0..B {
        out.push_str(&format!("{:2} ", c));
    }
    out.push('\n');

    // Top border
    out.push_str("   ┌");
    for c in 0..B {
        out.push_str("──");
        if c + 1 < B {
            out.push('┬');
        }
    }
    out.push_str("┐\n");

    // Board rows
    for r in 0..B {
        out.push_str(&format!("{:2} │", r));
        for c in 0..B {
            let cell = state.board[r][c];
            if cell == EMPTY {
                // Check if this empty cell is a start square
                let is_start = (0..P).any(|p| state.start_square(p) == (r, c));
                if is_start {
                    if use_color {
                        out.push_str("\x1b[35m★ \x1b[0m");
                    } else {
                        out.push_str("★ ");
                    }
                } else {
                    out.push_str("· ");
                }
            } else {
                let token = match cell {
                    0 => {
                        if use_color {
                            "\x1b[1;34m██\x1b[0m"
                        } else {
                            "0 "
                        }
                    }
                    1 => {
                        if use_color {
                            "\x1b[1;33m██\x1b[0m"
                        } else {
                            "1 "
                        }
                    }
                    2 => {
                        if use_color {
                            "\x1b[1;31m██\x1b[0m"
                        } else {
                            "2 "
                        }
                    }
                    3 => {
                        if use_color {
                            "\x1b[1;32m██\x1b[0m"
                        } else {
                            "3 "
                        }
                    }
                    _ => "??",
                };
                out.push_str(token);
            }
            if c + 1 < B {
                out.push('│');
            }
        }
        out.push_str("│\n");

        // Intermediate row dividers
        if r + 1 < B {
            out.push_str("   ├");
            for c in 0..B {
                out.push_str("──");
                if c + 1 < B {
                    out.push('┼');
                }
            }
            out.push_str("┤\n");
        }
    }

    // Bottom border
    out.push_str("   └");
    for c in 0..B {
        out.push_str("──");
        if c + 1 < B {
            out.push('┴');
        }
    }
    out.push_str("┘\n");

    out
}

/// Formats the current game scoreboard across all players.
pub fn render_scoreboard<const B: usize, const P: usize>(
    state: &BlokusState<B, P>,
    use_color: bool,
) -> String {
    let mut out = String::new();
    out.push_str("┌─────────────────────────────────────────────────────────────┐\n");
    out.push_str("│                     BLOKUS SCOREBOARD                       │\n");
    out.push_str("├────────┬────────────┬──────────┬──────────┬─────────────────┤\n");
    out.push_str("│ Player │ Name       │ Score    │ Unplaced │ Status          │\n");
    out.push_str("├────────┼────────────┼──────────┼──────────┼─────────────────┤\n");

    for p in 0..P {
        let player = Player::from_index(p);
        let name = player.name();
        let score = state.score(p);
        let unplaced = state.unplaced_squares(p);
        let is_active = !state.is_terminal() && state.current_player as usize == p;
        let has_passed = state.passed[p];

        let status = if is_active {
            "► ACTIVE ◄"
        } else if has_passed {
            "Passed"
        } else {
            "Waiting"
        };

        let color_code = match p {
            0 => "\x1b[1;34m",
            1 => "\x1b[1;33m",
            2 => "\x1b[1;31m",
            3 => "\x1b[1;32m",
            _ => "",
        };

        if use_color {
            out.push_str(&format!(
                "│ {color_code}Player {p}\x1b[0m │ {color_code}{:<10}\x1b[0m │ {:<8} │ {:<8} │ {:<15} │\n",
                name, score, unplaced, status
            ));
        } else {
            out.push_str(&format!(
                "│ Player {p} │ {:<10} │ {:<8} │ {:<8} │ {:<15} │\n",
                name, score, unplaced, status
            ));
        }
    }
    out.push_str("└────────┴────────────┴──────────┴──────────┴─────────────────┘\n");

    out
}

/// Formats an inventory list of unplaced pieces for `player`.
pub fn render_inventory<const B: usize, const P: usize>(
    state: &BlokusState<B, P>,
    player: usize,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("Inventory for Player {player} ({}):\n", Player::from_index(player).name()));

    let mask = state.remaining_pieces[player];
    let mut count = 0;
    for i in 0..21 {
        if (mask & (1 << i)) != 0 {
            out.push_str(&format!(
                "  [{:2}] {:<14} ({} sq) ",
                i,
                PIECE_NAMES[i],
                piece_size(i as u8)
            ));
            count += 1;
            if count % 3 == 0 {
                out.push('\n');
            }
        }
    }
    if count % 3 != 0 {
        out.push('\n');
    }
    out
}

/// Candidate move statistics produced during MCTS search.
#[derive(Debug, Clone)]
pub struct MoveCandidate {
    /// Action considered.
    pub action: BlokusAction,
    /// Edge traversal count.
    pub visits: u32,
    /// Prior probability from model.
    pub prior: f32,
    /// Mean return vector across all agents.
    pub mean_values: Vec<f32>,
}

/// Formats a table of top MCTS candidate moves.
pub fn format_move_candidates(candidates: &[MoveCandidate], limit: usize) -> String {
    let mut out = String::new();
    out.push_str("┌─────────────────────────────────────────────────────────────┐\n");
    out.push_str("│                    TOP MCTS CANDIDATES                      │\n");
    out.push_str("├──────┬───────────────────────────────┬────────┬─────────────┤\n");
    out.push_str("│ Rank │ Action                        │ Visits │ Prior (%)   │\n");
    out.push_str("├──────┼───────────────────────────────┼────────┼─────────────┤\n");

    for (i, cand) in candidates.iter().take(limit).enumerate() {
        let action_str = format!("{}", cand.action);
        out.push_str(&format!(
            "│ {:4} │ {:<29} │ {:6} │ {:>10.2}% │\n",
            i + 1,
            action_str,
            cand.visits,
            cand.prior * 100.0
        ));
    }
    out.push_str("└──────┴───────────────────────────────┴────────┴─────────────┘\n");
    out
}
