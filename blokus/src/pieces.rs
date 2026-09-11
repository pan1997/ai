//! Definition, geometric transformations, and canonical orientations of the 21 Blokus polyomino pieces.

/// Total number of unique polyomino pieces in Blokus per player.
pub const NUM_PIECES: usize = 21;

/// Total number of squares across all 21 polyomino pieces for a single player ($1\times 1 + 1\times 2 + 2\times 3 + 5\times 4 + 12\times 5 = 89$).
pub const TOTAL_SQUARES_PER_PLAYER: usize = 89;

/// Canonical representation of a polyomino shape in a specific orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolyominoShape {
    /// Number of squares forming this piece (1 to 5).
    pub num_squares: u8,
    /// Bounding box height in rows.
    pub height: u8,
    /// Bounding box width in columns.
    pub width: u8,
    /// Relative coordinates `(row, col)` normalized so that `min(row) == 0` and `min(col) == 0`.
    pub squares: [(u8, u8); 5],
}

impl PolyominoShape {
    /// Returns a slice of the active square coordinates `(row, col)`.
    #[inline]
    pub fn active_squares(&self) -> &[(u8, u8)] {
        &self.squares[..self.num_squares as usize]
    }
}

/// Base coordinate representations of the 21 free polyominoes.
const RAW_PIECES: [&[(i8, i8)]; NUM_PIECES] = [
    // 0: Monomino (1)
    &[(0, 0)],
    // 1: Domino (2)
    &[(0, 0), (0, 1)],
    // 2: I3 (3)
    &[(0, 0), (0, 1), (0, 2)],
    // 3: V3 / Corner (3)
    &[(0, 0), (0, 1), (1, 0)],
    // 4: I4 (4)
    &[(0, 0), (0, 1), (0, 2), (0, 3)],
    // 5: L4 (4)
    &[(0, 0), (1, 0), (2, 0), (2, 1)],
    // 6: T4 (4)
    &[(0, 0), (0, 1), (0, 2), (1, 1)],
    // 7: O4 / Square (4)
    &[(0, 0), (0, 1), (1, 0), (1, 1)],
    // 8: Z4 / Skew (4)
    &[(0, 0), (0, 1), (1, 1), (1, 2)],
    // 9: F (5)
    &[(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    // 10: I5 (5)
    &[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    // 11: L5 (5)
    &[(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],
    // 12: P (5)
    &[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    // 13: N (5)
    &[(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)],
    // 14: T5 (5)
    &[(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    // 15: U (5)
    &[(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)],
    // 16: V5 (5)
    &[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    // 17: W (5)
    &[(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
    // 18: X (5)
    &[(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
    // 19: Y (5)
    &[(0, 1), (1, 0), (1, 1), (2, 1), (3, 1)],
    // 20: Z5 (5)
    &[(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
];

/// Names of the 21 polyomino pieces.
pub const PIECE_NAMES: [&str; NUM_PIECES] = [
    "Monomino (1)",
    "Domino (2)",
    "I3",
    "V3",
    "I4",
    "L4",
    "T4",
    "O4",
    "Z4",
    "F5",
    "I5",
    "L5",
    "P5",
    "N5",
    "T5",
    "U5",
    "V5",
    "W5",
    "X5",
    "Y5",
    "Z5",
];

/// Returns the square count for a given piece index `0..21`.
#[inline]
pub fn piece_size(piece_id: u8) -> u8 {
    match piece_id {
        0 => 1,
        1 => 2,
        2..=3 => 3,
        4..=8 => 4,
        9..=20 => 5,
        _ => panic!("Invalid piece_id {piece_id}"),
    }
}

/// Computes all distinct canonical orientations for a raw piece coordinate set.
fn generate_orientations(raw: &[(i8, i8)]) -> Vec<PolyominoShape> {
    let mut unique_shapes: Vec<PolyominoShape> = Vec::new();
    let num_squares = raw.len() as u8;

    for &flip in &[false, true] {
        for rot in 0..4 {
            let mut transformed = [(0i8, 0i8); 5];
            for (i, &(r, c)) in raw.iter().enumerate() {
                let (mut tr, mut tc) = if flip { (r, -c) } else { (r, c) };
                for _ in 0..rot {
                    let next_r = tc;
                    let next_c = -tr;
                    tr = next_r;
                    tc = next_c;
                }
                transformed[i] = (tr, tc);
            }

            // Normalize to origin (min_r = 0, min_c = 0)
            let min_r = transformed[..num_squares as usize].iter().map(|&(r, _)| r).min().unwrap();
            let min_c = transformed[..num_squares as usize].iter().map(|&(_, c)| c).min().unwrap();

            let mut normalized = [(0u8, 0u8); 5];
            for (i, &(r, c)) in transformed[..num_squares as usize].iter().enumerate() {
                normalized[i] = ((r - min_r) as u8, (c - min_c) as u8);
            }

            // Sort lexicographically for canonical duplicate detection
            normalized[..num_squares as usize].sort_unstable();

            let max_r = normalized[..num_squares as usize].iter().map(|&(r, _)| r).max().unwrap();
            let max_c = normalized[..num_squares as usize].iter().map(|&(_, c)| c).max().unwrap();

            let shape = PolyominoShape {
                num_squares,
                height: max_r + 1,
                width: max_c + 1,
                squares: normalized,
            };

            if !unique_shapes.contains(&shape) {
                unique_shapes.push(shape);
            }
        }
    }

    unique_shapes
}

/// Lazily precomputed table of all distinct orientations for each of the 21 pieces.
///
/// Across all 21 pieces, there are exactly 91 distinct canonical orientations.
pub struct PieceRegistry {
    orientations: [Vec<PolyominoShape>; NUM_PIECES],
}

impl PieceRegistry {
    fn new() -> Self {
        let mut orientations: [Vec<PolyominoShape>; NUM_PIECES] = Default::default();
        for (i, &raw) in RAW_PIECES.iter().enumerate() {
            orientations[i] = generate_orientations(raw);
        }
        Self { orientations }
    }

    /// Returns the slice of canonical orientations for `piece_id` (0..21).
    #[inline]
    pub fn orientations_of(&self, piece_id: usize) -> &[PolyominoShape] {
        &self.orientations[piece_id]
    }
}

/// Global accessor for precomputed polyomino piece orientations.
pub fn registry() -> &'static PieceRegistry {
    static REGISTRY: std::sync::OnceLock<PieceRegistry> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(PieceRegistry::new)
}

