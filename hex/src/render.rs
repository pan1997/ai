//! Terminal rendering utilities for Hex boards and MCTS search statistics.

use crate::game::{HexPlayer, HexState};

/// Formats a Hex board into an ANSI or plain text rhombus string.
pub fn render_board<const N: usize>(state: &HexState<N>) -> String {
    render_board_styled(state, true)
}

/// Formats a Hex board with optional ANSI color styling.
///
/// When `use_color` is true:
/// - Black stones and top/bottom boundaries are highlighted in bold blue.
/// - White stones and left/right boundaries are highlighted in bold red.
/// - Empty cells are rendered as neutral dots (`·`).
pub fn render_board_styled<const N: usize>(state: &HexState<N>, use_color: bool) -> String {
    let mut out = String::new();

    let black_color = if use_color { "\x1b[1;34m" } else { "" };
    let white_color = if use_color { "\x1b[1;31m" } else { "" };
    let dim_color = if use_color { "\x1b[90m" } else { "" };
    let reset = if use_color { "\x1b[0m" } else { "" };

    // Header indicating player connection objectives
    out.push_str(&format!(
        "   {black_color}▼ Top: Black ({black_color}X{reset}) connects Top ↔ Bottom ▼{reset}\n"
    ));

    // Column headers (letters A, B, C, ...)
    out.push_str("     ");
    for c in 0..N {
        let col_char = (b'A' + c as u8) as char;
        out.push_str(&format!(" {col_char}"));
    }
    out.push('\n');

    // Top boundary line
    out.push_str(&format!("    {black_color}╔"));
    for _ in 0..N {
        out.push_str("══");
    }
    out.push_str(&format!("╗{reset}\n"));

    // Grid rows with rhombus diagonal indentation
    for r in 0..N {
        let row_num = r + 1;
        // Indentation for rhombus shape
        for _ in 0..r {
            out.push(' ');
        }

        // Left boundary: row number and White left border
        out.push_str(&format!("{row_num:>2} {white_color}║{reset}"));

        // Cells in row r
        for c in 0..N {
            let idx = HexState::<N>::idx(r, c);
            match state.board[idx] {
                Some(HexPlayer::Black) => out.push_str(&format!(" {black_color}X{reset}")),
                Some(HexPlayer::White) => out.push_str(&format!(" {white_color}O{reset}")),
                None => out.push_str(&format!(" {dim_color}·{reset}")),
            }
        }

        // Right boundary: White right border and row number
        out.push_str(&format!(" {white_color}║{reset} {row_num}\n"));
    }

    // Bottom boundary line
    for _ in 0..N {
        out.push(' ');
    }
    out.push_str(&format!("    {black_color}╚"));
    for _ in 0..N {
        out.push_str("══");
    }
    out.push_str(&format!("╝{reset}\n"));

    // Bottom column headers
    for _ in 0..N {
        out.push(' ');
    }
    out.push_str("     ");
    for c in 0..N {
        let col_char = (b'A' + c as u8) as char;
        out.push_str(&format!(" {col_char}"));
    }
    out.push_str(&format!(
        "\n   {white_color}◄ Left: White ({white_color}O{reset}) connects Left ↔ Right ►{reset}\n"
    ));

    out
}

/// Statistics for a candidate action evaluated by MCTS search.
#[derive(Debug, Clone)]
pub struct MoveCandidate {
    /// 1D cell index.
    pub action: usize,
    /// Algebraic coordinate string (e.g. `"F6"`).
    pub coord: String,
    /// MCTS visit count $N(s, a)$.
    pub visits: u32,
    /// Mean value estimate $Q(s, a)$ from acting player's perspective.
    pub q_value: f32,
    /// Policy prior $P(s, a)$.
    pub prior: f32,
}

/// Formats a ranked list of candidate moves into an ANSI terminal table.
pub fn format_move_candidates(candidates: &[MoveCandidate], max_display: usize) -> String {
    let mut out = String::new();
    out.push_str(
        "--------------------------------------------------------------------------------\n",
    );
    out.push_str(
        " Rank | Move  | Cell  | Visits     | Win % / Q   | Prior       | Policy Bar     \n",
    );
    out.push_str(
        "--------------------------------------------------------------------------------\n",
    );

    let total_visits: u32 = candidates.iter().map(|c| c.visits).sum();
    let display_count = candidates.len().min(max_display);

    for (i, c) in candidates.iter().take(display_count).enumerate() {
        let visit_ratio = if total_visits > 0 {
            c.visits as f32 / total_visits as f32
        } else {
            0.0
        };
        let bar_len = (visit_ratio * 14.0).round() as usize;
        let bar = "█".repeat(bar_len);

        out.push_str(&format!(
            " {:>4} | {:<5} | {:>5} | {:>10} | {:>+10.4} | {:>10.4} | {:<14}\n",
            i + 1,
            c.coord,
            c.action,
            c.visits,
            c.q_value,
            c.prior,
            bar,
        ));
    }
    out.push_str(
        "--------------------------------------------------------------------------------\n",
    );
    out
}
