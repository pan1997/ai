use crate::agent::{MctsAgent, RandomAgent, TacticalAgent};
use crate::dynamics::TicTacToeDynamics;
use crate::evaluator::RolloutEvaluator;
use crate::game::{Player, TicTacToeState, WIN_LINES};
use crate::world::TicTacToeWorld;
use mcts_engine::arena::{GameOutcome, MatchDriver};
use mcts_traits::{Agent, AgentDynamics, AgentId};

#[test]
fn test_initial_state() {
    let state = TicTacToeState::new();
    assert_eq!(state.current_player, Player::X);
    assert!(!state.is_full());
    assert!(!state.is_terminal());
    assert_eq!(state.check_winner(), None);
    assert_eq!(state.outcome(), None);

    let mut legal = Vec::new();
    state.legal_actions(&mut legal);
    assert_eq!(legal.len(), 9);
    assert_eq!(legal, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn test_all_eight_win_lines() {
    for &[a, b, c] in &WIN_LINES {
        let mut state = TicTacToeState::new();
        state.board[a] = Some(Player::X);
        state.board[b] = Some(Player::X);
        state.board[c] = Some(Player::X);
        assert_eq!(state.check_winner(), Some(Player::X));
        assert!(state.is_terminal());
        assert_eq!(state.outcome(), Some([1.0, -1.0]));

        let mut state_o = TicTacToeState::new();
        state_o.board[a] = Some(Player::O);
        state_o.board[b] = Some(Player::O);
        state_o.board[c] = Some(Player::O);
        assert_eq!(state_o.check_winner(), Some(Player::O));
        assert!(state_o.is_terminal());
        assert_eq!(state_o.outcome(), Some([-1.0, 1.0]));
    }
}

#[test]
fn test_full_board_draw() {
    // Classic draw board:
    // X O X
    // X X O
    // O X O
    let mut state = TicTacToeState::new();
    let moves = [
        (0, Player::X),
        (1, Player::O),
        (2, Player::X),
        (4, Player::X),
        (6, Player::O),
        (3, Player::X),
        (5, Player::O),
        (7, Player::X),
        (8, Player::O),
    ];
    for (idx, p) in moves {
        state.board[idx] = Some(p);
    }
    assert!(state.is_full());
    assert_eq!(state.check_winner(), None);
    assert!(state.is_terminal());
    assert_eq!(state.outcome(), Some([0.0, 0.0]));
}

#[test]
fn test_dynamics_step_and_agent_rotation() {
    let dynamics = TicTacToeDynamics;
    let mut state = dynamics.initial();
    assert_eq!(dynamics.current_agent(&state), AgentId(0));

    let outcome1 = dynamics.step(&mut state, &0);
    assert!(!outcome1.terminated);
    assert_eq!(outcome1.reward, [0.0, 0.0]);
    assert_eq!(state.board[0], Some(Player::X));
    assert_eq!(dynamics.current_agent(&state), AgentId(1));

    let outcome2 = dynamics.step(&mut state, &4);
    assert!(!outcome2.terminated);
    assert_eq!(state.board[4], Some(Player::O));
    assert_eq!(dynamics.current_agent(&state), AgentId(0));
}

#[test]
fn test_world_turn_based_match() {
    let world = TicTacToeWorld::new();
    let mut p0 = TacticalAgent::new("Tactical0");
    let mut p1 = TacticalAgent::new("Tactical1");

    let driver = MatchDriver::new();
    let result = driver.play_2p(&world, &mut p0, &mut p1, None);
    assert!(result.total_moves >= 5 && result.total_moves <= 9);
    // Two tactical agents on TicTacToe always draw!
    assert_eq!(result.outcome_2p, Some(GameOutcome::Draw));
}

#[test]
fn test_tactical_agent_immediate_win() {
    let mut state = TicTacToeState::new();
    // X at 0, 1 -> legal is 2..8, 2 is an immediate win
    state.board[0] = Some(Player::X);
    state.board[1] = Some(Player::X);
    state.current_player = Player::X;

    let mut agent = TacticalAgent::new("Tactical");
    let action = agent.select_action(&state);
    assert_eq!(action, 2);
}

#[test]
fn test_tactical_agent_immediate_block() {
    let mut state = TicTacToeState::new();
    // O at 0, 1 -> X must block at 2
    state.board[0] = Some(Player::O);
    state.board[1] = Some(Player::O);
    state.current_player = Player::X;

    let mut agent = TacticalAgent::new("Tactical");
    let action = agent.select_action(&state);
    assert_eq!(action, 2);
}

#[test]
fn test_mcts_agent_rollout_defeats_random() {
    let world = TicTacToeWorld::new();
    let mut mcts = MctsAgent::new("MCTS", RolloutEvaluator::default(), 100, 1.414);
    let mut random = RandomAgent::default();

    let mut mcts_wins = 0;
    let games = 10;
    let driver = MatchDriver::new();
    for _ in 0..games {
        let result = driver.play_2p(&world, &mut mcts, &mut random, None);
        if result.outcome_2p == Some(GameOutcome::Seat0Wins) {
            mcts_wins += 1;
        }
    }
    // MCTS playing X should win the vast majority against purely random play
    assert!(mcts_wins >= 8, "MCTS won only {mcts_wins}/{games} games");
}
