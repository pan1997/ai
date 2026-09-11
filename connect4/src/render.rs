//! Terminal rendering utilities for Connect 4 boards and MCTS search statistics.

use crate::game::{Connect4State, Player};

/// Formats a Connect 4 board into a human-readable string.
///
/// Uses standard box-drawing characters and clean tokens (`X` for Red, `O` for Yellow).
pub fn render_board<const R: usize, const C: usize>(state: &Connect4State<R, C>) -> String {
    render_board_styled(state, false)
}

/// Formats a Connect 4 board with optional ANSI color styling.
///
/// When `use_color` is true:
/// - Grid frame is rendered in blue (`\x1b[34m`), matching classic Connect 4 plastic boards.
/// - Red checkers are rendered in bold red (`\x1b[1;31mX\x1b[0m`).
/// - Yellow checkers are rendered in bold yellow (`\x1b[1;33mO\x1b[0m`).
pub fn render_board_styled<const R: usize, const C: usize>(
    state: &Connect4State<R, C>,
    use_color: bool,
) -> String {
    let mut out = String::new();

    // Column numbers header
    out.push_str("  ");
    for col in 0..C {
        out.push_str(&format!("  {col} "));
    }
    out.push('\n');

    // Top border
    let (h_bar, v_bar, cross, top_t, bot_t, top_l, top_r, bot_l, bot_r, left_t, right_t) =
        if use_color {
            (
                "\x1b[34m───\x1b[0m",
                "\x1b[34m│\x1b[0m",
                "\x1b[34m┼\x1b[0m",
                "\x1b[34m┬\x1b[0m",
                "\x1b[34m┴\x1b[0m",
                "\x1b[34m┌\x1b[0m",
                "\x1b[34m┐\x1b[0m",
                "\x1b[34m└\x1b[0m",
                "\x1b[34m┘\x1b[0m",
                "\x1b[34m├\x1b[0m",
                "\x1b[34m┤\x1b[0m",
            )
        } else {
            ("───", "│", "┼", "┬", "┴", "┌", "┐", "└", "┘", "├", "┤")
        };

    out.push_str("  ");
    out.push_str(top_l);
    for col in 0..C {
        out.push_str(h_bar);
        if col + 1 < C {
            out.push_str(top_t);
        }
    }
    out.push_str(top_r);
    out.push('\n');

    // Grid rows
    for row in 0..R {
        out.push_str("  ");
        for col in 0..C {
            out.push_str(v_bar);
            let cell = match state.board[row][col] {
                Some(Player::Red) => {
                    if use_color {
                        "\x1b[1;31m X \x1b[0m"
                    } else {
                        " X "
                    }
                }
                Some(Player::Yellow) => {
                    if use_color {
                        "\x1b[1;33m O \x1b[0m"
                    } else {
                        " O "
                    }
                }
                None => "   ",
            };
            out.push_str(cell);
        }
        out.push_str(v_bar);
        out.push('\n');

        // Middle dividers between rows
        if row + 1 < R {
            out.push_str("  ");
            out.push_str(left_t);
            for col in 0..C {
                out.push_str(h_bar);
                if col + 1 < C {
                    out.push_str(cross);
                }
            }
            out.push_str(right_t);
            out.push('\n');
        }
    }

    // Bottom border
    out.push_str("  ");
    out.push_str(bot_l);
    for col in 0..C {
        out.push_str(h_bar);
        if col + 1 < C {
            out.push_str(bot_t);
        }
    }
    out.push_str(bot_r);
    out.push('\n');

    // Footer column numbers
    out.push_str("  ");
    for col in 0..C {
        out.push_str(&format!("  {col} "));
    }
    out.push('\n');

    out
}

/// Statistics for a candidate action evaluated by MCTS.
#[derive(Debug, Clone, PartialEq)]
pub struct MoveCandidate {
    /// Action (column index 0..C).
    pub action: usize,
    /// Number of visits in the search tree.
    pub visits: u32,
    /// Fraction of total root visits.
    pub visit_fraction: f32,
    /// Model prior policy probability $P(s, a)$.
    pub prior: f32,
    /// Mean value return estimates $[Q_{\text{Red}}, Q_{\text{Yellow}}]$.
    pub mean_value: [f32; 2],
}

/// Formats a list of candidate actions and their search statistics as an ASCII table.
pub fn format_move_candidates(candidates: &[MoveCandidate]) -> String {
    if candidates.is_empty() {
        return "No candidate moves evaluated.\n".to_string();
    }

    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| b.visits.cmp(&a.visits));

    let mut out = String::new();
    out.push_str("┌──────┬────────┬────────┬───────┬─────────────────────────┐\n");
    out.push_str("│ Col  │ Visits │ Ratio  │ Prior │ Q-Values [Red, Yellow]  │\n");
    out.push_str("├──────┼────────┼────────┼───────┼─────────────────────────┤\n");

    for c in &sorted {
        out.push_str(&format!(
            "│  {:>2}  │ {:>6} │ {:>5.1}% │ {:>5.3} │ [{:>+6.3}, {:>+6.3}]     │\n",
            c.action,
            c.visits,
            c.visit_fraction * 100.0,
            c.prior,
            c.mean_value[0],
            c.mean_value[1],
        ));
    }

    out.push_str("└──────┴────────┴────────┴───────┴─────────────────────────┘\n");
    out
}

