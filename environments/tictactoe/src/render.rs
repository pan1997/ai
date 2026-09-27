//! Terminal rendering utilities for TicTacToe boards and search statistics.

use crate::game::{Player, TicTacToeState};

/// Formats a TicTacToe board into a human-readable string.
pub fn render_board(state: &TicTacToeState) -> String {
    render_board_styled(state, false)
}

/// Formats a TicTacToe board with optional ANSI color styling.
///
/// When `use_color` is true:
/// - X is rendered in bold red (`\x1b[1;31mX\x1b[0m`).
/// - O is rendered in bold blue (`\x1b[1;34mO\x1b[0m`).
/// - Empty cells show their index (0..9) in dim gray (`\x1b[2m{i}\x1b[0m`).
pub fn render_board_styled(state: &TicTacToeState, use_color: bool) -> String {
    let mut out = String::new();

    let (h_bar, v_bar, cross) = if use_color {
        ("\x1b[37m───\x1b[0m", "\x1b[37m│\x1b[0m", "\x1b[37m┼\x1b[0m")
    } else {
        ("───", "│", "┼")
    };

    for row in 0..3 {
        out.push_str(" ");
        for col in 0..3 {
            let idx = row * 3 + col;
            let token = match state.board[idx] {
                Some(Player::X) => {
                    if use_color {
                        "\x1b[1;31m X \x1b[0m".to_string()
                    } else {
                        " X ".to_string()
                    }
                }
                Some(Player::O) => {
                    if use_color {
                        "\x1b[1;34m O \x1b[0m".to_string()
                    } else {
                        " O ".to_string()
                    }
                }
                None => {
                    if use_color {
                        format!("\x1b[2m {idx} \x1b[0m")
                    } else {
                        format!(" {idx} ")
                    }
                }
            };
            out.push_str(&token);
            if col < 2 {
                out.push_str(v_bar);
            }
        }
        out.push('\n');
        if row < 2 {
            out.push_str(&format!(" {h_bar}{cross}{h_bar}{cross}{h_bar}\n"));
        }
    }

    out
}
