//! ANSI color terminal rendering for 2048 boards and MCTS search inspection.

use crate::game::{Direction, Tzf8State};

/// Formats a single tile value with terminal ANSI styling.
pub fn format_tile(val: u32) -> String {
    match val {
        0 => "\x1b[90m     .\x1b[0m".to_string(),
        2 => "\x1b[36m     2\x1b[0m".to_string(),
        4 => "\x1b[34m     4\x1b[0m".to_string(),
        8 => "\x1b[33m     8\x1b[0m".to_string(),
        16 => "\x1b[38;5;208m    16\x1b[0m".to_string(),
        32 => "\x1b[31m    32\x1b[0m".to_string(),
        64 => "\x1b[35m    64\x1b[0m".to_string(),
        128 => "\x1b[93m   128\x1b[0m".to_string(),
        256 => "\x1b[92m   256\x1b[0m".to_string(),
        512 => "\x1b[96m   512\x1b[0m".to_string(),
        1024 => "\x1b[95m  1024\x1b[0m".to_string(),
        2048 => "\x1b[1;93m  2048\x1b[0m".to_string(),
        other => format!("\x1b[1;97;41m{:6}\x1b[0m", other),
    }
}

/// Renders a $4 \times 4$ board as an ASCII box with ANSI-colored tiles and game summary.
pub fn render_board(state: &Tzf8State) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Score: {:<8} | Max Tile: {:<6} | Ongoing: {}\n",
        state.score,
        state.max_tile(),
        state.ongoing
    ));
    out.push_str("+-------+-------+-------+-------+\n");
    for r in 0..4 {
        out.push('|');
        for c in 0..4 {
            out.push_str(&format!("{} |", format_tile(state.board[r][c])));
        }
        out.push('\n');
        out.push_str("+-------+-------+-------+-------+\n");
    }
    out
}

/// Candidate move statistics for MCTS search introspection.
#[derive(Debug, Clone)]
pub struct MoveCandidate {
    /// Move direction.
    pub direction: Direction,
    /// Edge visit count.
    pub visits: u32,
    /// Running mean Q-value estimate.
    pub mean_q: f32,
    /// Policy prior probability assigned to this direction.
    pub prior: f32,
}

/// Formats candidate move statistics into an ASCII table.
pub fn format_move_candidates(candidates: &[MoveCandidate]) -> String {
    let mut out = String::new();
    out.push_str("+-----------+---------+-----------+---------+\n");
    out.push_str("| Direction | Visits  | Q-Value   | Prior   |\n");
    out.push_str("+-----------+---------+-----------+---------+\n");
    for c in candidates {
        out.push_str(&format!(
            "| {:<9} | {:<7} | {:<9.1} | {:<6.1}% |\n",
            c.direction.name(),
            c.visits,
            c.mean_q,
            c.prior * 100.0
        ));
    }
    out.push_str("+-----------+---------+-----------+---------+\n");
    out
}
