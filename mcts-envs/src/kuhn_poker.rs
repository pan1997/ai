use mcts_traits::{AgentDynamics, Transition, World};

/// Actions available in Kuhn Poker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KuhnAction {
    /// Pass action without adding chips to the pot.
    Check,
    /// Bet 1 additional chip into the pot.
    Bet,
    /// Match the opponent's bet with 1 chip.
    Call,
    /// Concede the pot to the opponent.
    Fold,
}

/// Impartial ground-truth world state for Kuhn Poker.
///
/// Contains true hole cards for both players, pot contributions, action history,
/// and acting player index.
#[derive(Debug, Clone, PartialEq)]
pub struct KuhnWorldState {
    /// Private hole cards: 0 = Jack, 1 = Queen, 2 = King.
    pub cards: [u8; 2],
    /// Chips contributed to the pot by player 0 and player 1.
    pub pot: [f32; 2],
    /// Complete sequence of betting actions played so far.
    pub history: Vec<KuhnAction>,
    /// Index of the player whose turn it is to act (0 or 1).
    pub current_player: usize,
    /// Whether the hand has reached showdown or folded termination.
    pub terminated: bool,
}

/// Player-specific observation in Kuhn Poker (imperfect information).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KuhnObservation {
    /// The player's private card (0 = Jack, 1 = Queen, 2 = King).
    pub my_card: u8,
    /// Visible public betting sequence.
    pub history: Vec<KuhnAction>,
    /// Whether the hand is complete.
    pub terminal: bool,
}

/// External referee and match manager for Kuhn Poker implementing [`World`].
#[derive(Default)]
pub struct KuhnWorld {
    /// Optional predetermined card deal `[p0_card, p1_card]` for deterministic test scenarios.
    pub fixed_deal: Option<[u8; 2]>,
}

impl World for KuhnWorld {
    type WorldState = KuhnWorldState;
    type Action = KuhnAction;
    type Observation = KuhnObservation;

    fn n_players(&self) -> usize {
        2
    }

    fn initial(&self) -> Self::WorldState {
        let cards = self.fixed_deal.unwrap_or([1, 2]); // Default Q to P0, K to P1
        KuhnWorldState {
            cards,
            pot: [1.0, 1.0], // 1 ante each
            history: Vec::new(),
            current_player: 0,
            terminated: false,
        }
    }

    fn observe(&self, ws: &Self::WorldState, player: usize) -> Self::Observation {
        KuhnObservation {
            my_card: ws.cards[player],
            history: ws.history.clone(),
            terminal: ws.terminated,
        }
    }

    fn actions(&self, ws: &Self::WorldState, player: usize) -> Vec<Self::Action> {
        if ws.terminated || player != ws.current_player {
            return Vec::new();
        }
        match ws.history.as_slice() {
            [] => vec![KuhnAction::Check, KuhnAction::Bet],
            [KuhnAction::Check] => vec![KuhnAction::Check, KuhnAction::Bet],
            [KuhnAction::Bet] => vec![KuhnAction::Fold, KuhnAction::Call],
            [KuhnAction::Check, KuhnAction::Bet] => vec![KuhnAction::Fold, KuhnAction::Call],
            _ => Vec::new(),
        }
    }

    fn step(
        &self,
        mut ws: Self::WorldState,
        joint: &[Self::Action],
    ) -> (Self::WorldState, Vec<f32>, bool) {
        let action = joint[ws.current_player];
        ws.history.push(action);

        let mut rewards = vec![0.0, 0.0];

        match ws.history.as_slice() {
            [KuhnAction::Check, KuhnAction::Check] => {
                ws.terminated = true;
                if ws.cards[0] > ws.cards[1] {
                    rewards = vec![1.0, -1.0];
                } else {
                    rewards = vec![-1.0, 1.0];
                }
            }
            [KuhnAction::Check, KuhnAction::Bet] => {
                ws.pot[1] += 1.0;
                ws.current_player = 0;
            }
            [KuhnAction::Check, KuhnAction::Bet, KuhnAction::Fold] => {
                ws.terminated = true;
                rewards = vec![-1.0, 1.0]; // P1 folded
            }
            [KuhnAction::Check, KuhnAction::Bet, KuhnAction::Call] => {
                ws.pot[0] += 1.0;
                ws.terminated = true;
                if ws.cards[0] > ws.cards[1] {
                    rewards = vec![2.0, -2.0];
                } else {
                    rewards = vec![-2.0, 2.0];
                }
            }
            [KuhnAction::Bet] => {
                ws.pot[0] += 1.0;
                ws.current_player = 1;
            }
            [KuhnAction::Bet, KuhnAction::Fold] => {
                ws.terminated = true;
                rewards = vec![1.0, -1.0]; // P2 folded
            }
            [KuhnAction::Bet, KuhnAction::Call] => {
                ws.pot[1] += 1.0;
                ws.terminated = true;
                if ws.cards[0] > ws.cards[1] {
                    rewards = vec![2.0, -2.0];
                } else {
                    rewards = vec![-2.0, 2.0];
                }
            }
            _ => {}
        }

        let done = ws.terminated;
        (ws, rewards, done)
    }

    fn terminal(&self, ws: &Self::WorldState) -> bool {
        ws.terminated
    }
}

/// Agent-centric internal planning dynamics for Kuhn Poker.
///
/// Models belief-state determinization and hypothetical lines against modeled opponent policies.
pub struct KuhnAgentDynamics {
    /// Private card held by the planning agent.
    pub my_card: u8,
    /// Probability with which opponent calls bets when holding Queen.
    pub opponent_call_rate: f32,
}

impl AgentDynamics for KuhnAgentDynamics {
    type State = KuhnObservation;
    type Action = KuhnAction;
    type Reward = [f32; 1];

    fn initial(&self) -> Self::State {
        KuhnObservation {
            my_card: self.my_card,
            history: Vec::new(),
            terminal: false,
        }
    }

    fn actions(&self, s: &Self::State) -> Vec<Self::Action> {
        if s.terminal {
            return Vec::new();
        }
        match s.history.as_slice() {
            [] => vec![KuhnAction::Check, KuhnAction::Bet],
            [KuhnAction::Check, KuhnAction::Bet] => vec![KuhnAction::Fold, KuhnAction::Call],
            _ => Vec::new(),
        }
    }

    fn step(&self, mut s: Self::State, action: &Self::Action) -> Transition<Self::State, Self::Reward> {
        s.history.push(*action);

        let reward;
        let terminated;

        match s.history.as_slice() {
            [KuhnAction::Check] => {
                s.history.push(KuhnAction::Check);
                terminated = true;
                reward = if s.my_card == 2 { 1.0 } else if s.my_card == 0 { -1.0 } else { 0.0 };
            }
            [KuhnAction::Bet] => {
                terminated = true;
                if self.opponent_call_rate > 0.5 {
                    s.history.push(KuhnAction::Call);
                    reward = if s.my_card == 2 { 2.0 } else { -2.0 };
                } else {
                    s.history.push(KuhnAction::Fold);
                    reward = 1.0;
                }
            }
            [KuhnAction::Check, KuhnAction::Bet, KuhnAction::Fold] => {
                terminated = true;
                reward = -1.0;
            }
            [KuhnAction::Check, KuhnAction::Bet, KuhnAction::Call] => {
                terminated = true;
                reward = if s.my_card == 2 { 2.0 } else { -2.0 };
            }
            _ => {
                terminated = true;
                reward = 0.0;
            }
        }

        s.terminal = terminated;
        Transition::new(s, [reward], terminated)
    }
}
